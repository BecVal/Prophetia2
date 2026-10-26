import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from scipy.stats import entropy
from scipy.special import softmax, i0e
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN BAYESIANA DE PESOS DE ESPECIALISTAS (OPTUNA)
# ==============================================================================
# Cambia RUN_OPTUNA a True si deseas volver a buscar los mejores pesos por submodelo
# e hiperparámetros de regularización/temperatura. De lo contrario (False), cargará
# los mejores guardados en optuna_params_stacker.json.
RUN_OPTUNA = False
OPTUNA_TRIALS = 40
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_stacker.json')
)
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_stacker')
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH_FUND = os.path.join(MODEL_SAVE_DIR, 'stacker_fundamental_model.pkl')
MODEL_SAVE_PATH_FINAL = os.path.join(MODEL_SAVE_DIR, 'stacker_final_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

DEFAULT_STACKER_PARAMS = {
    'w_quant': 0.18,
    'w_context': 0.38,
    'w_nn': 0.34,
    'w_draws': 0.10,
    'c_l1': 0.25,
    'tau_l1': 1.00,
    'w_fund': 0.38,
    'w_market': 0.24,
    'w_gbm': 0.20,
    'w_open': 0.18,
    'c_l2_odds': 0.08,
    'c_l2_noodds': 0.20,
    'tau_l2_odds': 1.00,
    'tau_l2_noodds': 1.00,
    'half_life_days': 600.0,
}


def get_time_weights(dates, half_life_days=600.0):
    """
    Calcula pesos de decaimiento exponencial temporal anclados a la fecha máxima
    del subconjunto recibido y normalizados a media unitaria (mean(w) == 1.0),
    evitando que la penalización L2 cambie de escala entre folds históricos.
    """
    if dates is None:
        return None
    s_dates = pd.to_datetime(pd.Series(dates))
    if s_dates.isna().all() or len(s_dates) == 0:
        return None
    max_date = s_dates.max()
    days_diff = (max_date - s_dates).dt.days.clip(lower=0).fillna(0).values
    w = np.exp(-np.log(2.0) * days_diff / max(float(half_life_days), 30.0))
    mean_w = np.mean(w)
    if mean_w > 0:
        w = w / mean_w
    return np.asarray(w, dtype=np.float64)


def clr_transform(prob_matrix):
    """
    Transformación isométrica Centered Log-Ratio (CLR) de Aitchison sobre el Símplex Δ^2:
        CLR(p)_k = ln(p_k) - (1/3) * sum_j ln(p_j)
    Proyecta vectores de probabilidad al espacio tangente real sin colinealidad perfecta.
    """
    p = np.asarray(prob_matrix, dtype=np.float64)
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / np.where(row_sums > 0, row_sums, 1.0)
    lp = np.log(np.clip(p, 1e-5, 1.0))
    return lp - lp.mean(axis=1, keepdims=True)


def jensen_shannon_divergence(p_mat, q_mat):
    """
    Divergencia simétrica y acotada de Jensen-Shannon entre dos distribuciones 1X2:
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),   M = 0.5 * (P + Q)
    """
    p = np.clip(np.asarray(p_mat, dtype=np.float64), 1e-6, 1.0)
    q = np.clip(np.asarray(q_mat, dtype=np.float64), 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    return np.clip(0.5 * (kl_pm + kl_qm), 0.0, np.log(3.0))


def opinion_pools(prob_list, weights):
    """
    Calcula simultáneamente el Linear Opinion Pool (convexo en el símplex) y el
    Logarithmic Opinion Pool (multiplicativo bayesiano en log-odds) a partir de
    una lista de matrices (N, 3) y sus pesos de autoridad Optuna.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-4)
    w = w / np.sum(w)

    lin_pool = np.zeros_like(prob_list[0], dtype=np.float64)
    log_pool_logits = np.zeros_like(prob_list[0], dtype=np.float64)

    for p_m, w_m in zip(prob_list, w):
        p_clean = np.clip(np.asarray(p_m, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        p_clean = p_clean / p_clean.sum(axis=1, keepdims=True)
        lin_pool += w_m * p_clean
        log_pool_logits += w_m * np.log(p_clean)

    lin_pool = lin_pool / lin_pool.sum(axis=1, keepdims=True)
    log_pool = softmax(log_pool_logits, axis=1)
    return lin_pool, log_pool


def build_level1_features(X_fund_raw, params=None):
    """
    Construye el espacio matemático de Nivel 1 (Consejo de Especialistas Fundamentales):
      1. Probabilidades directas de los especialistas (Quant, Context, NN, Draws).
      2. Coordenadas simpliciales CLR (Log-Odds centrados) por especialista.
      3. Invariantes estructurales de Skellam/Poisson sobre xG (ΔxG, xG_tot, ln(xG_H/xG_A), P_Pois(Draw)).
      4. Métricas de desacuerdo inter-especialista (dispersión y Divergencia de Jensen-Shannon).
      5. Consensos ponderados por Optuna: Linear Opinion Pool (LinOP) y Logarithmic Opinion Pool (LogOP).
    """
    p_cfg = DEFAULT_STACKER_PARAMS.copy()
    if params:
        p_cfg.update(params)

    df_in = X_fund_raw.reset_index(drop=True).fillna(0.0)
    out = df_in.copy()

    if {'quant_loss_prob', 'quant_draw_prob', 'quant_win_prob'}.issubset(df_in.columns):
        pq = df_in[['quant_loss_prob', 'quant_draw_prob', 'quant_win_prob']].values
    elif {'poisson_loss_prob', 'poisson_draw_prob', 'poisson_win_prob'}.issubset(df_in.columns):
        pq = df_in[['poisson_loss_prob', 'poisson_draw_prob', 'poisson_win_prob']].values
    else:
        pq = np.tile([0.30, 0.25, 0.45], (len(df_in), 1))

    pc = (
        df_in[['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx']].values
        if {'prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'}.issubset(df_in.columns)
        else pq.copy()
    )
    pn = (
        df_in[['prob_loss_nn', 'prob_draw_nn', 'prob_win_nn']].values
        if {'prob_loss_nn', 'prob_draw_nn', 'prob_win_nn'}.issubset(df_in.columns)
        else pc.copy()
    )

    p_draw_spec = (
        np.clip(df_in['prob_is_draw'].values, 1e-4, 1.0 - 1e-4)
        if 'prob_is_draw' in df_in.columns
        else 0.5 * (pc[:, 1] + pn[:, 1])
    )

    # Construir distribución 3-way inducida por el especialista de empates acoplado al consenso 1/2
    base_12 = 0.5 * (pc[:, [0, 2]] + pn[:, [0, 2]])
    base_12_sum = np.maximum(base_12.sum(axis=1, keepdims=True), 1e-5)
    rem_mass = np.clip(1.0 - p_draw_spec[:, None], 0.05, 0.95)
    pd_3way = np.column_stack([
        rem_mass[:, 0] * (base_12[:, 0] / base_12_sum[:, 0]),
        p_draw_spec,
        rem_mass[:, 0] * (base_12[:, 1] / base_12_sum[:, 0]),
    ])

    # 1. Transformaciones Simpliciales CLR (Log-Odds Centrados)
    for name, mat in [('q', pq), ('ctx', pc), ('nn', pn)]:
        clr_mat = clr_transform(mat)
        out[f'clr_{name}_loss'] = clr_mat[:, 0]
        out[f'clr_{name}_draw'] = clr_mat[:, 1]
        out[f'clr_{name}_win'] = clr_mat[:, 2]

    out['logit_is_draw'] = np.log(p_draw_spec / (1.0 - p_draw_spec))

    # 2. Invariantes de Skellam / Poisson sobre xG
    xg_s_col = 'predicted_xg_scored_quant' if 'predicted_xg_scored_quant' in df_in.columns else 'predicted_xg_scored'
    xg_c_col = 'predicted_xg_conceded_quant' if 'predicted_xg_conceded_quant' in df_in.columns else 'predicted_xg_conceded'
    xg_s = np.clip(df_in[xg_s_col].values if xg_s_col in df_in.columns else np.full(len(df_in), 1.35), 0.05, 8.0)
    xg_c = np.clip(df_in[xg_c_col].values if xg_c_col in df_in.columns else np.full(len(df_in), 1.10), 0.05, 8.0)

    out['skellam_xg_diff'] = xg_s - xg_c
    out['skellam_xg_total'] = xg_s + xg_c
    out['log_xg_ratio'] = np.log(xg_s / xg_c)
    # Probabilidad analítica exacta de empate bajo Poisson independiente: exp(-(lambda_H + lambda_A)) * I_0(2*sqrt(lambda_H*lambda_A))
    two_sqrt = 2.0 * np.sqrt(xg_s * xg_c)
    out['skellam_draw_prob'] = np.exp(-(xg_s + xg_c) + two_sqrt) * i0e(two_sqrt)

    # 3. Dispersión y Divergencia de Jensen-Shannon inter-especialista
    stack_3 = np.stack([pq, pc, pn], axis=0)
    std_3 = np.std(stack_3, axis=0)
    out['l1_std_loss'] = std_3[:, 0]
    out['l1_std_draw'] = std_3[:, 1]
    out['l1_std_win'] = std_3[:, 2]
    out['l1_js_ctx_nn'] = jensen_shannon_divergence(pc, pn)
    out['l1_js_quant_ctx'] = jensen_shannon_divergence(pq, pc)

    # 4. Opinion Pools Ponderados por Optuna (Linear Pool + Logarithmic Pool)
    w_l1 = [p_cfg['w_quant'], p_cfg['w_context'], p_cfg['w_nn'], p_cfg['w_draws']]
    lin_p, log_p = opinion_pools([pq, pc, pn, pd_3way], w_l1)
    for idx_k, lbl in enumerate(['loss', 'draw', 'win']):
        out[f'l1_linop_{lbl}'] = lin_p[:, idx_k]
        out[f'l1_logop_{lbl}'] = log_p[:, idx_k]

    clr_logop = clr_transform(log_p)
    out['l1_clr_logop_loss'] = clr_logop[:, 0]
    out['l1_clr_logop_draw'] = clr_logop[:, 1]
    out['l1_clr_logop_win'] = clr_logop[:, 2]

    return out.fillna(0.0)


def compute_meta_features(X_base, df_orig, prefix="meta_", params=None):
    """
    Construye el espacio matemático de Nivel 2 (Comité Ejecutivo Fundamental + Mercado):
      1. Descompone cuotas de apertura en Probabilidades Justas sin Vig (fair_open_*) y Overround (vig).
      2. Detecta explícitamente el régimen de observabilidad de cuotas (has_valid_odds) y evita
         contaminar con ceros (0.0 -> -3 sigma) cuando un partido carece de cuotas.
      3. Proyecta Fundamental, Market (Shin), GBM (Drift) y Fair Open al espacio tangente CLR Log-Odds.
      4. Calcula el Vector de Edge Fundamental vs. Mercado y la Divergencia de Jensen-Shannon.
      5. Calcula los Opinion Pools Ejecutivos (Linear + Logarithmic) con los pesos Optuna de Nivel 2.
    """
    p_cfg = DEFAULT_STACKER_PARAMS.copy()
    if params:
        p_cfg.update(params)

    X_base = X_base.reset_index(drop=True).fillna(0.0)
    meta = pd.DataFrame(index=X_base.index)

    # 1. Varianza inter-modelo base (únicamente sobre columnas de probabilidad directa [0, 1])
    prob_cols_loss = [
        c for c in X_base.columns
        if 'loss' in c.lower() and not any(k in c.lower() for k in ['clr_', 'std_', 'edge_', 'logop', 'linop'])
    ]
    prob_cols_draw = [
        c for c in X_base.columns
        if 'draw' in c.lower() and not any(k in c.lower() for k in ['clr_', 'std_', 'edge_', 'logop', 'linop', 'is_draw', 'skellam'])
    ]
    prob_cols_win = [
        c for c in X_base.columns
        if 'win' in c.lower() and not any(k in c.lower() for k in ['clr_', 'std_', 'edge_', 'logop', 'linop'])
    ]

    if prob_cols_loss:
        meta[f'{prefix}std_loss'] = X_base[prob_cols_loss].std(axis=1).fillna(0.0)
    if prob_cols_draw:
        meta[f'{prefix}std_draw'] = X_base[prob_cols_draw].std(axis=1).fillna(0.0)
    if prob_cols_win:
        meta[f'{prefix}std_win'] = X_base[prob_cols_win].std(axis=1).fillna(0.0)

    # 2. Entropía de Shannon del consenso medio
    mean_probs = pd.DataFrame(
        {
            'loss': X_base[prob_cols_loss].mean(axis=1) if prob_cols_loss else 0.30,
            'draw': X_base[prob_cols_draw].mean(axis=1) if prob_cols_draw else 0.25,
            'win': X_base[prob_cols_win].mean(axis=1) if prob_cols_win else 0.45,
        },
        index=X_base.index,
    )
    sums = mean_probs.sum(axis=1).values[:, None]
    p_cons = mean_probs.values / np.where(sums > 0, sums, 1.0)
    p_cons = np.clip(p_cons, 1e-9, 1.0)
    meta[f'{prefix}entropy'] = -np.sum(p_cons * np.log(p_cons), axis=1)

    # Extraer probabilidades del Nivel 1 Fundamental
    if {'fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'}.issubset(X_base.columns):
        p_fund = X_base[['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win']].values
    else:
        p_fund = p_cons.copy()
    p_fund = np.clip(p_fund, 1e-5, 1.0 - 1e-5)
    p_fund = p_fund / p_fund.sum(axis=1, keepdims=True)

    # 3. Extracción de Cuotas de Apertura y Régimen de Observabilidad (has_valid_odds)
    raw_ol = np.full(len(X_base), np.nan)
    raw_od = np.full(len(X_base), np.nan)
    raw_ow = np.full(len(X_base), np.nan)

    if df_orig is not None:
        if 'open_odds_loss' in df_orig.columns:
            s_ol = df_orig['open_odds_loss']
            if 'odds_loss' in df_orig.columns:
                s_ol = s_ol.fillna(df_orig['odds_loss'])
            raw_ol = pd.to_numeric(s_ol, errors='coerce').values
        elif 'odds_loss' in df_orig.columns:
            raw_ol = pd.to_numeric(df_orig['odds_loss'], errors='coerce').values

        if 'open_odds_draw' in df_orig.columns:
            s_od = df_orig['open_odds_draw']
            if 'odds_draw' in df_orig.columns:
                s_od = s_od.fillna(df_orig['odds_draw'])
            raw_od = pd.to_numeric(s_od, errors='coerce').values
        elif 'odds_draw' in df_orig.columns:
            raw_od = pd.to_numeric(df_orig['odds_draw'], errors='coerce').values

        if 'open_odds_win' in df_orig.columns:
            s_ow = df_orig['open_odds_win']
            if 'odds_win' in df_orig.columns:
                s_ow = s_ow.fillna(df_orig['odds_win'])
            raw_ow = pd.to_numeric(s_ow, errors='coerce').values
        elif 'odds_win' in df_orig.columns:
            raw_ow = pd.to_numeric(df_orig['odds_win'], errors='coerce').values

    has_valid_odds = (
        np.isfinite(raw_ol)
        & np.isfinite(raw_od)
        & np.isfinite(raw_ow)
        & (raw_ol > 1.01)
        & (raw_od > 1.01)
        & (raw_ow > 1.01)
    )
    meta['has_valid_odds'] = has_valid_odds.astype(np.float64)

    inv_ol = np.where(has_valid_odds, 1.0 / np.clip(raw_ol, 1.01, 100.0), p_fund[:, 0] * 1.05)
    inv_od = np.where(has_valid_odds, 1.0 / np.clip(raw_od, 1.01, 100.0), p_fund[:, 1] * 1.05)
    inv_ow = np.where(has_valid_odds, 1.0 / np.clip(raw_ow, 1.01, 100.0), p_fund[:, 2] * 1.05)

    meta['implied_open_loss'] = inv_ol
    meta['implied_open_draw'] = inv_od
    meta['implied_open_win'] = inv_ow

    vig_sum = np.maximum(inv_ol + inv_od + inv_ow, 1e-5)
    p_open_fair = np.column_stack([inv_ol / vig_sum, inv_od / vig_sum, inv_ow / vig_sum])
    p_open_fair = np.clip(p_open_fair, 1e-5, 1.0 - 1e-5)
    p_open_fair = p_open_fair / p_open_fair.sum(axis=1, keepdims=True)

    meta[f'{prefix}fair_open_loss'] = p_open_fair[:, 0]
    meta[f'{prefix}fair_open_draw'] = p_open_fair[:, 1]
    meta[f'{prefix}fair_open_win'] = p_open_fair[:, 2]
    meta[f'{prefix}open_overround'] = np.where(has_valid_odds, vig_sum - 1.0, 0.05)

    # Extraer probabilidades de Market (Shin) y GBM (Drift)
    p_mkt = (
        X_base[['prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt']].values
        if {'prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt'}.issubset(X_base.columns)
        else p_open_fair.copy()
    )
    p_gbm = (
        X_base[['prob_loss_gbm', 'prob_draw_gbm', 'prob_win_gbm']].values
        if {'prob_loss_gbm', 'prob_draw_gbm', 'prob_win_gbm'}.issubset(X_base.columns)
        else p_open_fair.copy()
    )
    # En filas sin cuotas válidas, alinear especialistas de mercado con p_fund para evitar ruido
    p_mkt = np.where(has_valid_odds[:, None], p_mkt, p_fund)
    p_gbm = np.where(has_valid_odds[:, None], p_gbm, p_fund)

    # 4. Transformaciones Simpliciales CLR (Log-Odds Centrados) en Nivel 2
    for name, mat in [('fund', p_fund), ('mkt', p_mkt), ('gbm', p_gbm), ('open', p_open_fair)]:
        clr_mat = clr_transform(mat)
        meta[f'{prefix}clr_{name}_loss'] = clr_mat[:, 0]
        meta[f'{prefix}clr_{name}_draw'] = clr_mat[:, 1]
        meta[f'{prefix}clr_{name}_win'] = clr_mat[:, 2]

    # 5. Vector de Edge Fundamental vs. Mercado y Divergencia de Jensen-Shannon
    p_mkt_comp = (0.40 * p_mkt) + (0.30 * p_gbm) + (0.30 * p_open_fair)
    p_mkt_comp = p_mkt_comp / p_mkt_comp.sum(axis=1, keepdims=True)

    meta[f'{prefix}edge_fund_mkt_loss'] = p_fund[:, 0] - p_mkt_comp[:, 0]
    meta[f'{prefix}edge_fund_mkt_draw'] = p_fund[:, 1] - p_mkt_comp[:, 1]
    meta[f'{prefix}edge_fund_mkt_win'] = p_fund[:, 2] - p_mkt_comp[:, 2]
    meta[f'{prefix}js_fund_mkt'] = jensen_shannon_divergence(p_fund, p_mkt_comp)
    meta[f'{prefix}js_mkt_gbm'] = jensen_shannon_divergence(p_mkt, p_gbm)

    # 6. Opinion Pools Ejecutivos de Nivel 2 (Ponderados por Optuna)
    w_l2 = [p_cfg['w_fund'], p_cfg['w_market'], p_cfg['w_gbm'], p_cfg['w_open']]
    lin_l2, log_l2 = opinion_pools([p_fund, p_mkt, p_gbm, p_open_fair], w_l2)
    for idx_k, lbl in enumerate(['loss', 'draw', 'win']):
        meta[f'{prefix}l2_linop_{lbl}'] = lin_l2[:, idx_k]
        meta[f'{prefix}l2_logop_{lbl}'] = log_l2[:, idx_k]

    clr_logop_l2 = clr_transform(log_l2)
    meta[f'{prefix}clr_logop_loss'] = clr_logop_l2[:, 0]
    meta[f'{prefix}clr_logop_draw'] = clr_logop_l2[:, 1]
    meta[f'{prefix}clr_logop_win'] = clr_logop_l2[:, 2]

    res = pd.concat([X_base, meta], axis=1).fillna(0.0)
    return res


class TemperatureScaledClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador lineal multinomial en espacio tangente con escalado de temperatura post-hoc (tau)
    para calibración exacta sobre el Símplex de Probabilidad.
    """

    def __init__(self, C=0.1, tau=1.0, max_iter=1000, random_state=42):
        self.C = C
        self.tau = tau
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_arr)
        # Guard contra columnas casi constantes en bloques de burn-in (ej. std ~ 1e-17)
        self.scaler_.scale_ = np.where(self.scaler_.scale_ < 1e-3, 1.0, self.scaler_.scale_)
        X_sc = self.scaler_.transform(X_arr)
        self.lr_ = LogisticRegression(C=float(self.C), max_iter=self.max_iter, random_state=self.random_state)
        if sample_weight is not None:
            self.lr_.fit(X_sc, y, sample_weight=np.asarray(sample_weight, dtype=np.float64))
        else:
            self.lr_.fit(X_sc, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_sc = self.scaler_.transform(X_arr)
        logits = self.lr_.decision_function(X_sc)
        tau_safe = max(float(self.tau), 0.1)
        probs = softmax(logits / tau_safe, axis=1)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RegimeGatedMetaStacker(BaseEstimator, ClassifierMixin):
    """
    Meta-Stacker de Nivel 2 con Compuerta de Observabilidad de Mercado:
      - En partidos CON cuotas válidas (has_valid_odds == 1), aplica el Comité Ejecutivo Completo
        (Fundamental + Market Shin + GBM Drift + Fair Open Probs + LogOP/LinOP Optuna).
      - En partidos SIN cuotas válidas (has_valid_odds == 0), enruta automáticamente al submodelo
        entrenado sobre las señales fundamentales puras (sin contaminarse con modelos de mercado ciegos).
    """

    def __init__(self, c_odds=0.08, c_noodds=0.20, tau_odds=1.0, tau_noodds=1.0, fund_cols=None, all_cols=None):
        self.c_odds = c_odds
        self.c_noodds = c_noodds
        self.tau_odds = tau_odds
        self.tau_noodds = tau_noodds
        self.fund_cols = fund_cols
        self.all_cols = all_cols

    def _extract_matrices(self, X):
        if isinstance(X, pd.DataFrame):
            has_odds = (
                (X['has_valid_odds'].values > 0.5)
                if 'has_valid_odds' in X.columns
                else np.ones(len(X), dtype=bool)
            )
            X_all = X[self.all_cols].values if self.all_cols is not None else X.values
            X_fund = X[self.fund_cols].values if self.fund_cols is not None else X.values
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            has_odds = np.ones(len(X_arr), dtype=bool)
            X_all = X_arr
            X_fund = X_arr
        return X_all, X_fund, has_odds

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            if self.all_cols is None:
                self.all_cols = [c for c in X.columns if c != 'has_valid_odds']
            if self.fund_cols is None:
                # En partidos SIN cuotas, usar exclusivamente especialistas libres de cuotas (Quant + Context + Draws + Skellam)
                preferred_no_odds = [
                    'quant_loss_prob', 'quant_draw_prob', 'quant_win_prob',
                    'prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx',
                    'clr_q_loss', 'clr_q_draw', 'clr_q_win',
                    'clr_ctx_loss', 'clr_ctx_draw', 'clr_ctx_win',
                    'prob_is_draw', 'logit_is_draw',
                    'skellam_xg_diff', 'skellam_xg_total', 'log_xg_ratio', 'skellam_draw_prob',
                ]
                self.fund_cols = [c for c in preferred_no_odds if c in X.columns]
                if not self.fund_cols:
                    self.fund_cols = [
                        c for c in X.columns
                        if c.startswith('fund_prob_') or c.startswith('meta_clr_fund_')
                    ] or self.all_cols

        X_all, X_fund, has_odds = self._extract_matrices(X)
        y_arr = np.asarray(y)
        w_arr = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None

        self.model_odds_ = TemperatureScaledClassifier(C=self.c_odds, tau=self.tau_odds)
        if np.any(has_odds):
            w_o = w_arr[has_odds] if w_arr is not None else None
            if w_o is not None and np.mean(w_o) > 0:
                w_o = w_o / np.mean(w_o)
            self.model_odds_.fit(X_all[has_odds], y_arr[has_odds], sample_weight=w_o)
        else:
            self.model_odds_.fit(X_all, y_arr, sample_weight=w_arr)

        self.model_noodds_ = TemperatureScaledClassifier(C=self.c_noodds, tau=self.tau_noodds)
        if np.sum(~has_odds) >= 50:
            w_no = w_arr[~has_odds] if w_arr is not None else None
            if w_no is not None and np.mean(w_no) > 0:
                w_no = w_no / np.mean(w_no)
            self.model_noodds_.fit(X_fund[~has_odds], y_arr[~has_odds], sample_weight=w_no)
        else:
            self.model_noodds_.fit(X_fund, y_arr, sample_weight=w_arr)

        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X):
        X_all, X_fund, has_odds = self._extract_matrices(X)
        probs = np.zeros((len(X_all), 3), dtype=np.float64)
        if np.any(has_odds):
            probs[has_odds] = self.model_odds_.predict_proba(X_all[has_odds])
        if np.any(~has_odds):
            probs[~has_odds] = self.model_noodds_.predict_proba(X_fund[~has_odds])
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def load_oof(model_name, split_idx, train_len=None, test_len=None):
    train_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_train.parquet')
    test_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_test.parquet')

    if os.path.exists(train_path) and os.path.exists(test_path):
        tr = pd.read_parquet(train_path, engine='fastparquet')
        ts = pd.read_parquet(test_path, engine='fastparquet')

        if 'lambda_total' in tr.columns:
            tr = tr.rename(columns={'lambda_total': f'lambda_total_{model_name}'})
            ts = ts.rename(columns={'lambda_total': f'lambda_total_{model_name}'})

        tr = tr.reset_index(drop=True).fillna(0.0)
        ts = ts.reset_index(drop=True).fillna(0.0)

        if train_len is not None and len(tr) != train_len:
            tr = tr.reindex(range(train_len)).fillna(0.0)
        if test_len is not None and len(ts) != test_len:
            ts = ts.reindex(range(test_len)).fillna(0.0)

        return tr, ts
    else:
        logger.warning(f"OOF para {model_name} no encontrado. Se omitirá.")
        return None, None


def _causal_consensus_prior(df_probs):
    if {'l1_logop_loss', 'l1_logop_draw', 'l1_logop_win'}.issubset(df_probs.columns):
        probs = df_probs[['l1_logop_loss', 'l1_logop_draw', 'l1_logop_win']].values
    elif {'meta_l2_logop_loss', 'meta_l2_logop_draw', 'meta_l2_logop_win'}.issubset(df_probs.columns):
        probs = df_probs[['meta_l2_logop_loss', 'meta_l2_logop_draw', 'meta_l2_logop_win']].values
    else:
        cols_l = [c for c in df_probs.columns if 'loss' in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower() and 'clr' not in c.lower() and 'edge' not in c.lower()]
        cols_d = [c for c in df_probs.columns if 'draw' in c.lower() and 'is_draw' not in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower() and 'clr' not in c.lower() and 'edge' not in c.lower() and 'skellam' not in c.lower()]
        cols_w = [c for c in df_probs.columns if 'win' in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower() and 'clr' not in c.lower() and 'edge' not in c.lower()]
        p_l = df_probs[cols_l].mean(axis=1).values if cols_l else np.full(len(df_probs), 0.30)
        p_d = df_probs[cols_d].mean(axis=1).values if cols_d else np.full(len(df_probs), 0.25)
        p_w = df_probs[cols_w].mean(axis=1).values if cols_w else np.full(len(df_probs), 0.45)
        probs = np.column_stack([p_l, p_d, p_w])
    probs = np.clip(probs, 1e-6, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def _generate_oof_level1(X_train_fund, y_train, train_dates, c_l1, tau_l1, half_life_days):
    """
    Genera predicciones OOF temporales estrictas para el Nivel 1 usando pesos temporales
    calculados y normalizados dentro de cada fold (sin colapso numérico en el bloque inicial).
    """
    fund_prob_train = np.zeros((len(X_train_fund), 3), dtype=np.float64)
    tscv_l1 = TimeSeriesSplit(n_splits=5)
    l1_splits = list(tscv_l1.split(X_train_fund))

    first_idx_l1 = l1_splits[0][0]
    sub_tscv = TimeSeriesSplit(n_splits=3)
    burn_in_cutoff = len(first_idx_l1) // 4
    for sub_tr, sub_va in sub_tscv.split(X_train_fund.iloc[first_idx_l1]):
        tr_pos = first_idx_l1[sub_tr]
        va_pos = first_idx_l1[sub_va]
        # Excluir filas de burn-in inicial donde Quant/Market/GBM aún son constantes
        valid_tr = tr_pos[tr_pos >= burn_in_cutoff]
        if len(valid_tr) < 500:
            continue
        sub_w = get_time_weights(train_dates.iloc[valid_tr], half_life_days) if train_dates is not None else None
        m_sub = TemperatureScaledClassifier(C=c_l1, tau=tau_l1)
        m_sub.fit(X_train_fund.iloc[valid_tr], y_train.iloc[valid_tr], sample_weight=sub_w)
        fund_prob_train[va_pos] = m_sub.predict_proba(X_train_fund.iloc[va_pos])

    unfilled_l1 = fund_prob_train.sum(axis=1) == 0
    if np.any(unfilled_l1):
        fund_prob_train[unfilled_l1] = _causal_consensus_prior(X_train_fund.iloc[unfilled_l1])

    for tr_idx, va_idx in l1_splits:
        fold_w = get_time_weights(train_dates.iloc[tr_idx], half_life_days) if train_dates is not None else None
        m_fold = TemperatureScaledClassifier(C=c_l1, tau=tau_l1)
        m_fold.fit(X_train_fund.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=fold_w)
        fund_prob_train[va_idx] = m_fold.predict_proba(X_train_fund.iloc[va_idx])

    return fund_prob_train / fund_prob_train.sum(axis=1, keepdims=True)


def _generate_oof_level2(X_train_meta, y_train, train_dates, c_odds, c_noodds, tau_odds, tau_noodds, half_life_days):
    """
    Genera predicciones OOF temporales estrictas para el Nivel 2 usando RegimeGatedMetaStacker
    y pesos temporales normalizados por fold.
    """
    y_prob_train = np.zeros((len(X_train_meta), 3), dtype=np.float64)
    cv = TimeSeriesSplit(n_splits=5)
    splits = list(cv.split(X_train_meta))

    first_idx = splits[0][0]
    sub_cv = TimeSeriesSplit(n_splits=3)
    burn_in_cutoff = len(first_idx) // 4
    for sub_tr, sub_va in sub_cv.split(X_train_meta.iloc[first_idx]):
        tr_pos = first_idx[sub_tr]
        va_pos = first_idx[sub_va]
        valid_tr = tr_pos[tr_pos >= burn_in_cutoff]
        if len(valid_tr) < 500:
            continue
        w_tr = get_time_weights(train_dates.iloc[valid_tr], half_life_days) if train_dates is not None else None
        m = RegimeGatedMetaStacker(c_odds=c_odds, c_noodds=c_noodds, tau_odds=tau_odds, tau_noodds=tau_noodds)
        m.fit(X_train_meta.iloc[valid_tr], y_train.iloc[valid_tr], sample_weight=w_tr)
        y_prob_train[va_pos] = m.predict_proba(X_train_meta.iloc[va_pos])

    unfilled = y_prob_train.sum(axis=1) == 0
    if np.any(unfilled):
        y_prob_train[unfilled] = _causal_consensus_prior(X_train_meta.iloc[unfilled])

    for tr_idx, va_idx in splits:
        w_tr = get_time_weights(train_dates.iloc[tr_idx], half_life_days) if train_dates is not None else None
        m = RegimeGatedMetaStacker(c_odds=c_odds, c_noodds=c_noodds, tau_odds=tau_odds, tau_noodds=tau_noodds)
        m.fit(X_train_meta.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=w_tr)
        y_prob_train[va_idx] = m.predict_proba(X_train_meta.iloc[va_idx])

    return y_prob_train / y_prob_train.sum(axis=1, keepdims=True)


def optimize_stacker_with_optuna(X_train_fund_raw, mkt_train_raw, df_train_orig, y_train, train_dates, n_trials=40):
    """
    Optimiza mediante Optuna (TPE Bayesiano) los pesos de autoridad asignados a cada modelo
    especialista en el Nivel 1 (Quant, Context, NN, Draws) y en el Nivel 2 (Fundamental,
    Market, GBM, Fair Open Odds), junto con las constantes de regularización C y temperaturas tau.
    Evalúa en los últimos 3 folds temporales de expansión (donde todos los especialistas son maduros).
    """
    logger.info(f"Optimizando Pesos de Especialistas y Meta-Parámetros con Optuna ({n_trials} Trials)...")
    tscv = TimeSeriesSplit(n_splits=5)
    eval_splits = list(tscv.split(X_train_fund_raw))[-2:]

    def objective(trial):
        cand_params = {
            'w_quant': trial.suggest_float('w_quant', 0.05, 0.50),
            'w_context': trial.suggest_float('w_context', 0.20, 0.80),
            'w_nn': trial.suggest_float('w_nn', 0.15, 0.80),
            'w_draws': trial.suggest_float('w_draws', 0.02, 0.30),
            'c_l1': trial.suggest_float('c_l1', 0.02, 1.50, log=True),
            'tau_l1': trial.suggest_float('tau_l1', 0.90, 1.15),
            'w_fund': trial.suggest_float('w_fund', 0.20, 0.80),
            'w_market': trial.suggest_float('w_market', 0.10, 0.65),
            'w_gbm': trial.suggest_float('w_gbm', 0.08, 0.55),
            'w_open': trial.suggest_float('w_open', 0.08, 0.55),
            'c_l2_odds': trial.suggest_float('c_l2_odds', 0.01, 0.50, log=True),
            'c_l2_noodds': trial.suggest_float('c_l2_noodds', 0.02, 0.80, log=True),
            'tau_l2_odds': trial.suggest_float('tau_l2_odds', 0.90, 1.15),
            'tau_l2_noodds': trial.suggest_float('tau_l2_noodds', 0.90, 1.15),
            'half_life_days': trial.suggest_float('half_life_days', 365.0, 900.0),
        }

        X_f = build_level1_features(X_train_fund_raw, cand_params)
        fold_losses = []

        for tr_idx, va_idx in eval_splits:
            w_tr = get_time_weights(train_dates.iloc[tr_idx], cand_params['half_life_days']) if train_dates is not None else None
            # Ajustar Nivel 1 en tr_idx
            m1 = TemperatureScaledClassifier(C=cand_params['c_l1'], tau=cand_params['tau_l1'])
            m1.fit(X_f.iloc[tr_idx], y_train.iloc[tr_idx], sample_weight=w_tr)

            # Generar probabilidades de Nivel 1 (usando consenso LogOP/predicción en train para rapidez y pureza)
            p1_tr = 0.5 * m1.predict_proba(X_f.iloc[tr_idx]) + 0.5 * X_f.iloc[tr_idx][['l1_logop_loss', 'l1_logop_draw', 'l1_logop_win']].values
            p1_va = m1.predict_proba(X_f.iloc[va_idx])

            df_f_tr = pd.DataFrame(p1_tr, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'])
            df_f_va = pd.DataFrame(p1_va, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'])
            f_sub_tr = X_f.iloc[tr_idx].reset_index(drop=True)
            f_sub_va = X_f.iloc[va_idx].reset_index(drop=True)

            if mkt_train_raw is not None:
                base_tr = pd.concat([df_f_tr, f_sub_tr, mkt_train_raw.iloc[tr_idx].reset_index(drop=True)], axis=1)
                base_va = pd.concat([df_f_va, f_sub_va, mkt_train_raw.iloc[va_idx].reset_index(drop=True)], axis=1)
            else:
                base_tr = pd.concat([df_f_tr, f_sub_tr], axis=1)
                base_va = pd.concat([df_f_va, f_sub_va], axis=1)

            meta_tr = compute_meta_features(base_tr, df_train_orig.iloc[tr_idx].reset_index(drop=True), prefix="meta_", params=cand_params)
            meta_va = compute_meta_features(base_va, df_train_orig.iloc[va_idx].reset_index(drop=True), prefix="meta_", params=cand_params)

            m2 = RegimeGatedMetaStacker(
                c_odds=cand_params['c_l2_odds'],
                c_noodds=cand_params['c_l2_noodds'],
                tau_odds=cand_params['tau_l2_odds'],
                tau_noodds=cand_params['tau_l2_noodds'],
            )
            m2.fit(meta_tr, y_train.iloc[tr_idx], sample_weight=w_tr)
            p2_va = m2.predict_proba(meta_va)
            fold_losses.append(log_loss(y_train.iloc[va_idx], np.clip(p2_va, 1e-6, 1.0 - 1e-6)))

        return float(np.mean(fold_losses))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = DEFAULT_STACKER_PARAMS.copy()
    best_params.update(study.best_params)

    with open(OPTUNA_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Mejores pesos y parámetros guardados en {OPTUNA_PARAMS_FILE}")
    return best_params


def train_stacker():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)

    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    train_len = len(y_train)
    test_len = len(y_test)

    df_train_orig = df.iloc[:split_idx].reset_index(drop=True)
    df_test_orig = df.iloc[split_idx:].reset_index(drop=True)

    train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx]).reset_index(drop=True) if 'match_date' in df.columns else None

    logger.info("Cargando predicciones OOF de los modelos 1X2 especialistas y de mercado...")

    oof_data = {}
    for mod in ['quant', 'poisson', 'context', 'nn', 'draws', 'market', 'gbm']:
        tr, ts = load_oof(mod, split_idx, train_len=train_len, test_len=test_len)
        if tr is not None:
            oof_data[mod] = (tr, ts)

    # ====== PREPARACIÓN DE MATRICES BASE ======
    fundamental_train_list = []
    fundamental_test_list = []

    if 'quant' in oof_data:
        logger.info("Modelo 'quant' encontrado. Se usará como especialista base fundamental.")
        base_models = ['quant', 'context', 'nn', 'draws']
    else:
        logger.info("Modelo 'quant' no encontrado. Se usará 'poisson' como plan B.")
        base_models = ['poisson', 'context', 'nn', 'draws']

    for mod in base_models:
        if mod in oof_data:
            fundamental_train_list.append(oof_data[mod][0])
            fundamental_test_list.append(oof_data[mod][1])

    if not fundamental_train_list:
        raise ValueError("No se encontraron modelos fundamentales.")

    X_train_fund_raw = pd.concat(fundamental_train_list, axis=1).reset_index(drop=True).fillna(0.0)
    X_test_fund_raw = pd.concat(fundamental_test_list, axis=1).reset_index(drop=True).fillna(0.0)

    market_models = []
    if 'market' in oof_data:
        market_models.append(oof_data['market'])
    if 'gbm' in oof_data:
        market_models.append(oof_data['gbm'])

    mkt_train_raw = pd.concat([m[0] for m in market_models], axis=1).reset_index(drop=True) if market_models else None
    mkt_test_raw = pd.concat([m[1] for m in market_models], axis=1).reset_index(drop=True) if market_models else None

    # ====== OPTIMIZACIÓN DE PESOS POR ESPECIALISTA CON OPTUNA ======
    if RUN_OPTUNA or not os.path.exists(OPTUNA_PARAMS_FILE):
        best_params = optimize_stacker_with_optuna(
            X_train_fund_raw, mkt_train_raw, df_train_orig, y_train, train_dates, n_trials=OPTUNA_TRIALS
        )
    else:
        logger.info(f"Cargando pesos óptimos de especialistas desde {OPTUNA_PARAMS_FILE}...")
        with open(OPTUNA_PARAMS_FILE, 'r') as f:
            loaded_p = json.load(f)
        if 'w_quant' not in loaded_p:
            logger.info("Archivo Optuna heredado detectado. Re-optimizando pesos de especialistas con Optuna...")
            best_params = optimize_stacker_with_optuna(
                X_train_fund_raw, mkt_train_raw, df_train_orig, y_train, train_dates, n_trials=OPTUNA_TRIALS
            )
        else:
            best_params = DEFAULT_STACKER_PARAMS.copy()
            best_params.update(loaded_p)

    # Normalizar y mostrar los pesos de autoridad por especialista en los logs
    w1_sum = best_params['w_quant'] + best_params['w_context'] + best_params['w_nn'] + best_params['w_draws']
    w2_sum = best_params['w_fund'] + best_params['w_market'] + best_params['w_gbm'] + best_params['w_open']
    logger.info(
        f"Pesos Optuna Nivel 1 (Especialistas Fundamentales) -> "
        f"Quant: {best_params['w_quant']/w1_sum*100:.1f}% | Context: {best_params['w_context']/w1_sum*100:.1f}% | "
        f"NN: {best_params['w_nn']/w1_sum*100:.1f}% | Draws: {best_params['w_draws']/w1_sum*100:.1f}%"
    )
    logger.info(
        f"Pesos Optuna Nivel 2 (Comité Ejecutivo) -> "
        f"Fundamental L1: {best_params['w_fund']/w2_sum*100:.1f}% | Market (Shin): {best_params['w_market']/w2_sum*100:.1f}% | "
        f"GBM (Drift): {best_params['w_gbm']/w2_sum*100:.1f}% | Fair Open Odds: {best_params['w_open']/w2_sum*100:.1f}%"
    )

    half_life = best_params['half_life_days']

    # ====== ETAPA 1: FUNDAMENTAL STACKER (NIVEL 1) ======
    logger.info("Entrenando Stacker Fundamental (Nivel 1) con Geometría Simplicial CLR + Invariantes Skellam...")
    X_train_fund = build_level1_features(X_train_fund_raw, best_params)
    X_test_fund = build_level1_features(X_test_fund_raw, best_params)

    fund_prob_train = _generate_oof_level1(
        X_train_fund, y_train, train_dates, best_params['c_l1'], best_params['tau_l1'], half_life
    )

    w_train_full = get_time_weights(train_dates, half_life) if train_dates is not None else None
    fund_pipeline = TemperatureScaledClassifier(C=best_params['c_l1'], tau=best_params['tau_l1'])
    fund_pipeline.fit(X_train_fund, y_train, sample_weight=w_train_full)

    fund_prob_test = fund_pipeline.predict_proba(X_test_fund)

    df_fund_train = pd.DataFrame(
        fund_prob_train, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_train_fund.index
    )
    df_fund_test = pd.DataFrame(
        fund_prob_test, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_test_fund.index
    )

    joblib.dump(
        {
            'model': fund_pipeline,
            'features': X_train_fund.columns.tolist(),
            'raw_features': X_train_fund_raw.columns.tolist(),
            'optuna_params': best_params,
        },
        MODEL_SAVE_PATH_FUND,
    )

    # ====== ETAPA 2: FINAL (MARKET-GATED) STACKER (NIVEL 2) ======
    logger.info("Entrenando Stacker Ejecutivo Final (Nivel 2) con Compuerta de Observabilidad de Mercado...")

    if mkt_train_raw is not None:
        X_train_meta_base = pd.concat([df_fund_train, X_train_fund, mkt_train_raw], axis=1).reset_index(drop=True).fillna(0.0)
        X_test_meta_base = pd.concat([df_fund_test, X_test_fund, mkt_test_raw], axis=1).reset_index(drop=True).fillna(0.0)
    else:
        logger.warning("No se encontró OOF de Market. El Stacker Final usará probabilidades fundamentales.")
        X_train_meta_base = pd.concat([df_fund_train, X_train_fund], axis=1).reset_index(drop=True).fillna(0.0)
        X_test_meta_base = pd.concat([df_fund_test, X_test_fund], axis=1).reset_index(drop=True).fillna(0.0)

    X_train_meta = compute_meta_features(X_train_meta_base, df_train_orig, prefix="meta_", params=best_params).fillna(0.0)
    X_test_meta = compute_meta_features(X_test_meta_base, df_test_orig, prefix="meta_", params=best_params).fillna(0.0)

    fit_features = X_train_meta.columns.tolist()

    final_model = RegimeGatedMetaStacker(
        c_odds=best_params['c_l2_odds'],
        c_noodds=best_params['c_l2_noodds'],
        tau_odds=best_params['tau_l2_odds'],
        tau_noodds=best_params['tau_l2_noodds'],
    )
    final_model.fit(X_train_meta, y_train, sample_weight=w_train_full)

    y_prob_train = _generate_oof_level2(
        X_train_meta,
        y_train,
        train_dates,
        best_params['c_l2_odds'],
        best_params['c_l2_noodds'],
        best_params['tau_l2_odds'],
        best_params['tau_l2_noodds'],
        half_life,
    )
    y_prob_test = final_model.predict_proba(X_test_meta)

    y_pred = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob_test)
    y_prob_train_norm = np.clip(y_prob_train, 1e-6, 1.0 - 1e-6)
    y_prob_train_norm = y_prob_train_norm / y_prob_train_norm.sum(axis=1, keepdims=True)
    train_oof_loss = log_loss(y_train, y_prob_train_norm)

    has_odds_ts = X_test_meta['has_valid_odds'].values > 0.5
    loss_with_odds = log_loss(y_test[has_odds_ts], y_prob_test[has_odds_ts])
    acc_with_odds = accuracy_score(y_test[has_odds_ts], y_pred[has_odds_ts])
    loss_no_odds = log_loss(y_test[~has_odds_ts], y_prob_test[~has_odds_ts])
    acc_no_odds = accuracy_score(y_test[~has_odds_ts], y_pred[~has_odds_ts])

    logger.info("=== RESULTADOS META-MODELO FINAL (REGIME-GATED BAYESIAN OPINION POOL + CLR LOG-ODDS) ===")
    logger.info(f"Train OOF Log-Loss: {train_oof_loss:.4f}")
    logger.info(f"Test Accuracy Global: {acc:.4f} | Test Log-Loss Global: {loss:.4f}")
    logger.info(f" - Partidos CON Cuotas ({has_odds_ts.sum()}): Log-Loss = {loss_with_odds:.4f} | Acc = {acc_with_odds:.4f}")
    logger.info(f" - Partidos SIN Cuotas ({(~has_odds_ts).sum()}): Log-Loss = {loss_no_odds:.4f} | Acc = {acc_no_odds:.4f}")

    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    y_test_oh = np.zeros_like(y_prob_test)
    y_test_oh[np.arange(len(y_test_arr)), y_test_arr] = 1

    brier_loss = brier_score_loss(y_test_oh[:, 0], y_prob_test[:, 0])
    brier_draw = brier_score_loss(y_test_oh[:, 1], y_prob_test[:, 1])
    brier_win = brier_score_loss(y_test_oh[:, 2], y_prob_test[:, 2])
    brier_global = np.mean([brier_loss, brier_draw, brier_win])

    logger.info(f"Brier Score Global (Promedio de Clases): {brier_global:.4f}")
    logger.info(f"Brier Score por Clase -> Loss: {brier_loss:.4f} | Draw: {brier_draw:.4f} | Win: {brier_win:.4f}")

    logger.info("=== AUDITORÍA ESTADÍSTICA DEL ENSAMBLE FINAL ===")
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()

    pred_loss = y_prob_train[:, 0].mean()
    pred_draw = y_prob_train[:, 1].mean()
    pred_win = y_prob_train[:, 2].mean()

    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Train Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Train Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Train Dataset: {real_win*100:.1f}%")

    # Preservar pred_clv_* previos si ya existían en test_predictions.parquet / train_predictions.parquet
    prev_clv_test = {}
    test_preds_file = os.path.join(PROCESSED_DIR, 'test_predictions.parquet')
    if os.path.exists(test_preds_file):
        try:
            old_ts = pd.read_parquet(test_preds_file, engine='fastparquet')
            if len(old_ts) == len(y_test):
                for c_clv in ['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win']:
                    if c_clv in old_ts.columns:
                        prev_clv_test[c_clv] = old_ts[c_clv].values
        except Exception:
            pass

    df_train_save = pd.DataFrame(
        {
            'match_date': df_train_orig['match_date'].values if 'match_date' in df.columns else np.array([None] * len(y_train)),
            'competition': df_train_orig['competition'].values if 'competition' in df.columns else np.array([None] * len(y_train)),
            'team': df_train_orig['team'].values,
            'opponent': df_train_orig['opponent'].values,
            'is_home': df_train_orig['is_home'].values,
            'prob_loss': y_prob_train[:, 0],
            'prob_draw': y_prob_train[:, 1],
            'prob_win': y_prob_train[:, 2],
            'fund_prob_loss': fund_prob_train[:, 0],
            'fund_prob_draw': fund_prob_train[:, 1],
            'fund_prob_win': fund_prob_train[:, 2],
            'meta_std_loss': X_train_meta['meta_std_loss'].values,
            'meta_std_draw': X_train_meta['meta_std_draw'].values,
            'meta_std_win': X_train_meta['meta_std_win'].values,
            'meta_js_fund_mkt': X_train_meta['meta_js_fund_mkt'].values,
            'outcome': y_train.values,
            'odds_win': df_train_orig['open_odds_win'].fillna(df_train_orig['odds_win']).values if 'open_odds_win' in df.columns else df_train_orig['odds_win'].values,
            'odds_draw': df_train_orig['open_odds_draw'].fillna(df_train_orig['odds_draw']).values if 'open_odds_draw' in df.columns else df_train_orig['odds_draw'].values,
            'odds_loss': df_train_orig['open_odds_loss'].fillna(df_train_orig['odds_loss']).values if 'open_odds_loss' in df.columns else df_train_orig['odds_loss'].values,
            'closing_odds_win': df_train_orig['closing_odds_win'].values if 'closing_odds_win' in df.columns else df_train_orig['odds_win'].values,
            'closing_odds_draw': df_train_orig['closing_odds_draw'].values if 'closing_odds_draw' in df.columns else df_train_orig['odds_draw'].values,
            'closing_odds_loss': df_train_orig['closing_odds_loss'].values if 'closing_odds_loss' in df.columns else df_train_orig['odds_loss'].values,
        }
    )

    has_odds = 'odds_win' in df.columns
    if has_odds:
        df_test_save = pd.DataFrame(
            {
                'match_date': df_test_orig['match_date'].values if 'match_date' in df.columns else np.array([None] * len(y_test)),
                'competition': df_test_orig['competition'].values if 'competition' in df.columns else np.array([None] * len(y_test)),
                'team': df_test_orig['team'].values,
                'opponent': df_test_orig['opponent'].values,
                'is_home': df_test_orig['is_home'].values,
                'prob_loss': y_prob_test[:, 0],
                'prob_draw': y_prob_test[:, 1],
                'prob_win': y_prob_test[:, 2],
                'fund_prob_loss': fund_prob_test[:, 0],
                'fund_prob_draw': fund_prob_test[:, 1],
                'fund_prob_win': fund_prob_test[:, 2],
                'meta_std_loss': X_test_meta['meta_std_loss'].values,
                'meta_std_draw': X_test_meta['meta_std_draw'].values,
                'meta_std_win': X_test_meta['meta_std_win'].values,
                'meta_js_fund_mkt': X_test_meta['meta_js_fund_mkt'].values,
                'outcome': y_test.values,
                'odds_win': df_test_orig['open_odds_win'].fillna(df_test_orig['odds_win']).values if 'open_odds_win' in df.columns else df_test_orig['odds_win'].values,
                'odds_draw': df_test_orig['open_odds_draw'].fillna(df_test_orig['odds_draw']).values if 'open_odds_draw' in df.columns else df_test_orig['odds_draw'].values,
                'odds_loss': df_test_orig['open_odds_loss'].fillna(df_test_orig['odds_loss']).values if 'open_odds_loss' in df.columns else df_test_orig['odds_loss'].values,
                'closing_odds_win': df_test_orig['closing_odds_win'].values if 'closing_odds_win' in df.columns else df_test_orig['odds_win'].values,
                'closing_odds_draw': df_test_orig['closing_odds_draw'].values if 'closing_odds_draw' in df.columns else df_test_orig['odds_draw'].values,
                'closing_odds_loss': df_test_orig['closing_odds_loss'].values if 'closing_odds_loss' in df.columns else df_test_orig['odds_loss'].values,
            }
        )
        for c_clv, vals_clv in prev_clv_test.items():
            df_test_save[c_clv] = vals_clv

        df_test_save.to_parquet(test_preds_file, engine='fastparquet')
        logger.info("Guardado test_predictions.parquet")

    df_train_save.to_parquet(os.path.join(PROCESSED_DIR, 'train_predictions.parquet'), engine='fastparquet')

    if 'fair_loss' in df.columns:
        for col in ['fair_loss', 'fair_draw', 'fair_win']:
            X_train_meta[col] = df_train_orig[col].values
            X_test_meta[col] = df_test_orig[col].values

    if 'open_fair_loss' in df.columns:
        for col in ['open_fair_loss', 'open_fair_draw', 'open_fair_win']:
            X_train_meta[col] = df_train_orig[col].values
            X_test_meta[col] = df_test_orig[col].values

    X_train_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_train.parquet'), engine='fastparquet')
    X_test_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_test.parquet'), engine='fastparquet')

    joblib.dump(
        {'model': final_model, 'features': fit_features, 'optuna_params': best_params},
        MODEL_SAVE_PATH_FINAL,
    )
    logger.info(f"Modelo Stacker Ejecutivo Final (Regime-Gated Log-Odds + Opinion Pool) guardado en {MODEL_SAVE_PATH_FINAL}")


if __name__ == "__main__":
    from core.models.train_stacker import train_stacker as _run_main

    _run_main()
