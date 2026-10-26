import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss, accuracy_score
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson, nbinom
import optuna

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = False
OPTUNA_TRIALS = 30
# ==============================================================================

logger = get_logger(__name__, 'train_shots_on_goal')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_PARAMS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_shots.json')
)
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'shots_on_goal_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))


def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"🚀 [CUDA ENABLED] GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ [CUDA DISABLED] Usando CPU. El entrenamiento será más lento.")
    return device


def get_time_weights(dates, half_life_days=365):
    if dates is None or len(dates) == 0:
        return None
    s_dates = pd.to_datetime(pd.Series(dates))
    max_date = s_dates.max()
    days_diff = (max_date - s_dates).dt.days.clip(lower=0).fillna(0)
    return np.asarray(np.exp(-np.log(2) * days_diff / half_life_days), dtype=np.float32).copy()


class PyTorchMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout_rate=0.25, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return torch.clamp(out, min=1e-4)


class SklearnPyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim,
        hidden_dims=(64, 32),
        dropout_rate=0.25,
        weight_decay=1e-2,
        epochs=60,
        batch_size=256,
        lr=1e-3,
        device='cpu',
        patience=6,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.output_dim_ = 1

    def _init_model(self, output_dim=1):
        self.output_dim_ = output_dim
        self.model = PyTorchMLPRegressor(
            self.input_dim, self.hidden_dims, self.dropout_rate, output_dim=output_dim
        ).to(self.device)

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float32).copy()
        y_arr = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float32).copy()
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        self._init_model(output_dim=y_arr.shape[1])

        val_idx = int(len(X_arr) * 0.85)
        if val_idx < 2 or val_idx >= len(X_arr) - 1:
            val_idx = len(X_arr)

        X_tr_raw = X_arr[:val_idx]
        X_val_raw = X_arr[val_idx:] if val_idx < len(X_arr) else X_arr[:val_idx]

        X_tr = np.asarray(self.scaler.fit_transform(self.imputer.fit_transform(X_tr_raw)), dtype=np.float32).copy()
        X_val = np.asarray(self.scaler.transform(self.imputer.transform(X_val_raw)), dtype=np.float32).copy()

        y_tr = y_arr[:val_idx].copy()
        y_val = (y_arr[val_idx:] if val_idx < len(y_arr) else y_arr[:val_idx]).copy()

        if sample_weight is not None:
            w_arr = np.asarray(
                sample_weight.values if hasattr(sample_weight, 'values') else sample_weight,
                dtype=np.float32,
            ).copy()
            w_tr = w_arr[:val_idx].copy()
            w_val = (w_arr[val_idx:] if val_idx < len(w_arr) else w_arr[:val_idx]).copy()
        else:
            w_tr = np.ones(len(X_tr), dtype=np.float32)
            w_val = np.ones(len(X_val), dtype=np.float32)

        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)
        w_tr_t = torch.from_numpy(w_tr).to(self.device)

        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)
        w_val_t = torch.from_numpy(w_val).to(self.device)

        dataset = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
        eff_bs = min(self.batch_size, max(len(X_tr_t), 1))
        # Evitar lote final de tamaño 1 que falla en BatchNorm1d
        drop_last = (len(X_tr_t) > eff_bs) and (len(X_tr_t) % eff_bs == 1)
        loader = DataLoader(dataset, batch_size=eff_bs, shuffle=True, drop_last=drop_last)

        criterion = nn.PoissonNLLLoss(log_input=False, reduction='none')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for _ in range(self.epochs):
            self.model.train()
            for bx, by, bw in loader:
                if len(bx) < 2:
                    continue
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                bw_unsqueeze = bw.unsqueeze(1)
                loss = (loss * bw_unsqueeze).sum() / (bw_unsqueeze.sum() * self.output_dim_ + 1e-8)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = criterion(val_preds, y_val_t)
                w_val_unsq = w_val_t.unsqueeze(1)
                val_loss = ((val_loss * w_val_unsq).sum() / (w_val_unsq.sum() * self.output_dim_ + 1e-8)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        return self

    def predict(self, X):
        self.model.eval()
        X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float32).copy()
        X_imputed = self.imputer.transform(X_arr)
        X_scaled = np.asarray(self.scaler.transform(X_imputed), dtype=np.float32).copy()
        X_tensor = torch.from_numpy(X_scaled).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds[:, 0] if self.output_dim_ == 1 else preds


class PerspectiveDistributionalEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensamble Distributivo para conteo de Tiros al Arco en una perspectiva ('home' o 'away'):
      1. XGBoost Poisson (count:poisson) - Captura no-linealidades e interacciones tácticas.
      2. PyTorch MLP Poisson (Softplus + GELU + BatchNorm) - Aproximador suave continuo.
      3. GLM Poisson Regularizado (PoissonRegressor L2 con enlace canónico log) - Estabilidad lineal.
      4. Meta-Combinador en Espacio Logarítmico (Log-Linear Geometric Pool via Ridge en log-espacio):
         ln(lambda_final) = w_0 + w_1 ln(lambda_xgb) + w_2 ln(lambda_mlp) + w_3 ln(lambda_glm)
    """

    def __init__(self, mlp_params, xgb_params, ridge_alpha=1.0, meta_alpha=1.0, cv_splits=3):
        self.mlp_params = mlp_params
        self.xgb_params = xgb_params
        self.ridge_alpha = ridge_alpha
        self.meta_alpha = meta_alpha
        self.cv_splits = cv_splits

        self.mlp_model = SklearnPyTorchRegressorWrapper(**mlp_params)
        self.xgb_model = XGBRegressor(**xgb_params)
        self.glm_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.glm_scaler = StandardScaler()
        self.glm_model = PoissonRegressor(alpha=ridge_alpha, max_iter=1000)
        self.meta_model = Ridge(alpha=meta_alpha, positive=True)

    @staticmethod
    def _to_log_features(p_xgb, p_mlp, p_glm):
        return np.column_stack([
            np.log(np.clip(p_xgb, 0.15, 25.0)),
            np.log(np.clip(p_mlp, 0.15, 25.0)),
            np.log(np.clip(p_glm, 0.15, 25.0)),
        ])

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float64).copy()
        y_arr = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float64).ravel().copy()
        w_arr = (
            np.asarray(sample_weight.values if hasattr(sample_weight, 'values') else sample_weight, dtype=np.float64).ravel().copy()
            if sample_weight is not None
            else None
        )

        xgb_oof = np.full(len(y_arr), np.nan, dtype=np.float64)
        mlp_oof = np.full(len(y_arr), np.nan, dtype=np.float64)
        glm_oof = np.full(len(y_arr), np.nan, dtype=np.float64)

        tscv_inner = TimeSeriesSplit(n_splits=self.cv_splits)
        for tr_idx, val_idx in tscv_inner.split(X_arr):
            X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
            X_va = X_arr[val_idx]
            w_tr = w_arr[tr_idx] if w_arr is not None else None

            # 1. XGBoost Poisson
            fold_xgb = XGBRegressor(**self.xgb_params)
            fold_xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            xgb_oof[val_idx] = fold_xgb.predict(X_va)

            # 2. PyTorch MLP Poisson
            fold_mlp = SklearnPyTorchRegressorWrapper(**self.mlp_params)
            fold_mlp.fit(X_tr, y_tr, sample_weight=w_tr)
            mlp_oof[val_idx] = fold_mlp.predict(X_va)

            # 3. GLM Poisson Regularizado
            fold_imp = SimpleImputer(strategy='median', keep_empty_features=True)
            fold_scl = StandardScaler()
            X_tr_scl = fold_scl.fit_transform(fold_imp.fit_transform(X_tr))
            X_va_scl = fold_scl.transform(fold_imp.transform(X_va))
            fold_glm = PoissonRegressor(alpha=self.ridge_alpha, max_iter=1000)
            fold_glm.fit(X_tr_scl, y_tr, sample_weight=w_tr)
            glm_oof[val_idx] = fold_glm.predict(X_va_scl)

        valid_mask = ~np.isnan(xgb_oof) & ~np.isnan(mlp_oof) & ~np.isnan(glm_oof)
        meta_X_log = self._to_log_features(xgb_oof[valid_mask], mlp_oof[valid_mask], glm_oof[valid_mask])
        # Target suavizado en espacio logarítmico con corrección de sesgo de Jensen
        # Para preservar E[Y] = exp(eta), entrenamos un PoissonRegressor sobre los log-predictores
        self.meta_poisson_ = PoissonRegressor(alpha=max(self.meta_alpha * 0.05, 1e-3), max_iter=1000)
        meta_w = w_arr[valid_mask] if w_arr is not None else None
        self.meta_poisson_.fit(meta_X_log, y_arr[valid_mask], sample_weight=meta_w)

        # Entrenar modelos base finales sobre el 100% de las muestras observadas de este bloque
        self.xgb_model.fit(X_arr, y_arr, sample_weight=w_arr, verbose=False)
        self.mlp_model.fit(X_arr, y_arr, sample_weight=w_arr)
        X_full_scl = self.glm_scaler.fit_transform(self.glm_imputer.fit_transform(X_arr))
        self.glm_model.fit(X_full_scl, y_arr, sample_weight=w_arr)
        return self

    def predict(self, X):
        X_arr = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float64).copy()
        p_xgb = self.xgb_model.predict(X_arr)
        p_mlp = self.mlp_model.predict(X_arr)
        X_scl = self.glm_scaler.transform(self.glm_imputer.transform(X_arr))
        p_glm = self.glm_model.predict(X_scl)

        meta_X_log = self._to_log_features(p_xgb, p_mlp, p_glm)
        preds = self.meta_poisson_.predict(meta_X_log)
        return np.clip(preds, 0.25, 18.0)


def enrich_clean_causal_shot_dynamics(df, df_all):
    """
    Calcula de forma estrictamente causal (.shift(1) cronológico por equipo y por condición Local/Visita)
    las medias exponenciales limpias de Tiros al Arco y Tiros Totales CREADOS y CONCEDIDOS,
    enmascarando como NaN los partidos históricos donde la liga no registró estadísticas de tiros
    para evitar contaminar las EMAs con ceros falsos.
    """
    work = df_all[['match_id', 'match_date', 'team', 'competition', 'is_home', 'shots_on_target', 'shots_total']].copy()
    work['match_date'] = pd.to_datetime(work['match_date'])
    work = work.sort_values(['match_date', 'match_id', 'is_home']).reset_index(drop=True)

    # Emparejar los tiros del rival en el mismo partido (= tiros concedidos por la defensa del equipo)
    opp_view = work[['match_id', 'is_home', 'shots_on_target', 'shots_total']].copy()
    opp_view['is_home'] = 1 - opp_view['is_home']
    opp_view = opp_view.rename(
        columns={'shots_on_target': 'sot_conceded_raw', 'shots_total': 'shots_conceded_raw'}
    ).drop_duplicates(subset=['match_id', 'is_home'])
    work = work.merge(opp_view, on=['match_id', 'is_home'], how='left')

    obs = (
        ((work['shots_total'].fillna(0) + work['shots_conceded_raw'].fillna(0)) > 0)
        | ((work['shots_on_target'].fillna(0) + work['sot_conceded_raw'].fillna(0)) > 0)
    )
    work['sot_clean'] = np.where(obs, work['shots_on_target'], np.nan)
    work['sot_conc_clean'] = np.where(obs, work['sot_conceded_raw'], np.nan)
    work['st_clean'] = np.where(obs, work['shots_total'], np.nan)
    work['st_conc_clean'] = np.where(obs, work['shots_conceded_raw'], np.nan)

    # 1. EMAs globales por equipo (sin importar localía)
    g_team = work.groupby('team', sort=False)
    work['clean_sot_c5'] = g_team['sot_clean'].transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
    work['clean_sot_c10'] = g_team['sot_clean'].transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    work['clean_sot_allow5'] = g_team['sot_conc_clean'].transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
    work['clean_sot_allow10'] = g_team['sot_conc_clean'].transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    work['clean_st_c8'] = g_team['st_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
    work['clean_st_allow8'] = g_team['st_conc_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())

    # 2. EMAs específicas de condición (Home-only / Away-only) por equipo
    g_venue = work.groupby(['team', 'is_home'], sort=False)
    work['venue_sot_c8'] = g_venue['sot_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
    work['venue_sot_allow8'] = g_venue['sot_conc_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
    work['venue_st_c8'] = g_venue['st_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
    work['venue_st_allow8'] = g_venue['st_conc_clean'].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())

    # 3. Baseline causal de la liga por condición (Home/Away)
    g_comp = work.groupby(['competition', 'is_home'], sort=False)
    work['comp_venue_sot_prior'] = g_comp['sot_clean'].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())

    cols_stat = [
        'clean_sot_c5', 'clean_sot_c10', 'clean_sot_allow5', 'clean_sot_allow10',
        'clean_st_c8', 'clean_st_allow8', 'venue_sot_c8', 'venue_sot_allow8',
        'venue_st_c8', 'venue_st_allow8', 'comp_venue_sot_prior',
    ]

    home_stats = work[work['is_home'] == 1][['match_id'] + cols_stat].drop_duplicates('match_id').rename(
        columns={c: f'h_{c}' for c in cols_stat}
    )
    away_stats = work[work['is_home'] == 0][['match_id'] + cols_stat].drop_duplicates('match_id').rename(
        columns={c: f'a_{c}' for c in cols_stat}
    )

    df = df.merge(home_stats, on='match_id', how='left').merge(away_stats, on='match_id', how='left')
    return df


def build_perspective_shots_features(df, perspective='home'):
    """
    Construye el espacio de características simétrico desde la perspectiva del equipo ('home' o 'away'),
    incorporando las EMAs limpias de tiros al arco creados y concedidos (globales y por condición Local/Visita),
    el Prior Multiplicativo de Intensidad de Poisson, la eficiencia de puntería Bayesiana (Beta-Binomial Shrinkage)
    y el enlace canónico log-xG.
    """
    feat = pd.DataFrame(index=df.index)

    if perspective == 'home':
        p_t, p_o = 'h_', 'a_'
        mu0_sot, mu0_st = 4.75, 13.2
        sot_ema3 = df['shots_on_target_ema3'].fillna(4.5) if 'shots_on_target_ema3' in df.columns else pd.Series(4.5, index=df.index)
        sot_ema5 = df['shots_on_target_ema5'].fillna(4.5) if 'shots_on_target_ema5' in df.columns else pd.Series(4.5, index=df.index)
        opp_sot_ema3 = df['opp_shots_on_target_ema3'].fillna(3.9) if 'opp_shots_on_target_ema3' in df.columns else pd.Series(3.9, index=df.index)
        opp_sot_ema5 = df['opp_shots_on_target_ema5'].fillna(3.9) if 'opp_shots_on_target_ema5' in df.columns else pd.Series(3.9, index=df.index)

        st_ema3 = df['shots_total_ema3'].fillna(13.2)
        st_ema5 = df['shots_total_ema5'].fillna(13.2)
        opp_st_ema3 = df['opp_shots_total_ema3'].fillna(10.8)
        opp_st_ema5 = df['opp_shots_total_ema5'].fillna(10.8)

        poss_ema3 = df['possession_pct_ema3'].fillna(51.0)
        poss_ema5 = df['possession_pct_ema5'].fillna(51.0)
        opp_poss_ema3 = df['opp_possession_pct_ema3'].fillna(49.0)
        opp_poss_ema5 = df['opp_possession_pct_ema5'].fillna(49.0)

        pass_ema3 = df['pass_accuracy_ema3'].fillna(78.0)
        pass_ema5 = df['pass_accuracy_ema5'].fillna(78.0)
        opp_pass_ema3 = df['opp_pass_accuracy_ema3'].fillna(78.0)
        opp_pass_ema5 = df['opp_pass_accuracy_ema5'].fillna(78.0)

        att_rtg = df['team_att_rating'].fillna(1.2) if 'team_att_rating' in df.columns else pd.Series(1.2, index=df.index)
        def_rtg = df['team_def_rating'].fillna(1.2) if 'team_def_rating' in df.columns else pd.Series(1.2, index=df.index)
        opp_att_rtg = df['opp_att_rating'].fillna(1.2) if 'opp_att_rating' in df.columns else pd.Series(1.2, index=df.index)
        opp_def_rtg = df['opp_def_rating'].fillna(1.2) if 'opp_def_rating' in df.columns else pd.Series(1.2, index=df.index)

        elo_diff = df['elo_diff'].fillna(0.0) if 'elo_diff' in df.columns else pd.Series(0.0, index=df.index)
        t_val = df['team_squad_value'].fillna(50.0).clip(lower=1.0)
        o_val = df['opp_squad_value'].fillna(50.0).clip(lower=1.0)

        q_xg_team = df['predicted_xg_scored_quant'].fillna(1.35).clip(lower=0.05)
        q_xg_opp = df['predicted_xg_conceded_quant'].fillna(1.10).clip(lower=0.05)

        imp_team = df['implied_prob_home'].fillna(0.45)
        imp_opp = df['implied_prob_away'].fillna(0.30)
        odds_ratio = df['odds_ratio_home_away'].fillna(0.0)
        is_home_val = 1.0
    else:
        p_t, p_o = 'a_', 'h_'
        mu0_sot, mu0_st = 3.90, 10.8
        sot_ema3 = df['opp_shots_on_target_ema3'].fillna(3.9) if 'opp_shots_on_target_ema3' in df.columns else pd.Series(3.9, index=df.index)
        sot_ema5 = df['opp_shots_on_target_ema5'].fillna(3.9) if 'opp_shots_on_target_ema5' in df.columns else pd.Series(3.9, index=df.index)
        opp_sot_ema3 = df['shots_on_target_ema3'].fillna(4.5) if 'shots_on_target_ema3' in df.columns else pd.Series(4.5, index=df.index)
        opp_sot_ema5 = df['shots_on_target_ema5'].fillna(4.5) if 'shots_on_target_ema5' in df.columns else pd.Series(4.5, index=df.index)

        st_ema3 = df['opp_shots_total_ema3'].fillna(10.8)
        st_ema5 = df['opp_shots_total_ema5'].fillna(10.8)
        opp_st_ema3 = df['shots_total_ema3'].fillna(13.2)
        opp_st_ema5 = df['shots_total_ema5'].fillna(13.2)

        poss_ema3 = df['opp_possession_pct_ema3'].fillna(49.0)
        poss_ema5 = df['opp_possession_pct_ema5'].fillna(49.0)
        opp_poss_ema3 = df['possession_pct_ema3'].fillna(51.0)
        opp_poss_ema5 = df['possession_pct_ema5'].fillna(51.0)

        pass_ema3 = df['opp_pass_accuracy_ema3'].fillna(78.0)
        pass_ema5 = df['opp_pass_accuracy_ema5'].fillna(78.0)
        opp_pass_ema3 = df['pass_accuracy_ema3'].fillna(78.0)
        opp_pass_ema5 = df['pass_accuracy_ema5'].fillna(78.0)

        att_rtg = df['opp_att_rating'].fillna(1.2) if 'opp_att_rating' in df.columns else pd.Series(1.2, index=df.index)
        def_rtg = df['opp_def_rating'].fillna(1.2) if 'opp_def_rating' in df.columns else pd.Series(1.2, index=df.index)
        opp_att_rtg = df['team_att_rating'].fillna(1.2) if 'team_att_rating' in df.columns else pd.Series(1.2, index=df.index)
        opp_def_rtg = df['team_def_rating'].fillna(1.2) if 'team_def_rating' in df.columns else pd.Series(1.2, index=df.index)

        elo_diff = (-df['elo_diff']).fillna(0.0) if 'elo_diff' in df.columns else pd.Series(0.0, index=df.index)
        t_val = df['opp_squad_value'].fillna(50.0).clip(lower=1.0)
        o_val = df['team_squad_value'].fillna(50.0).clip(lower=1.0)

        q_xg_team = df['predicted_xg_conceded_quant'].fillna(1.10).clip(lower=0.05)
        q_xg_opp = df['predicted_xg_scored_quant'].fillna(1.35).clip(lower=0.05)

        imp_team = df['implied_prob_away'].fillna(0.30)
        imp_opp = df['implied_prob_home'].fillna(0.45)
        odds_ratio = -df['odds_ratio_home_away'].fillna(0.0)
        is_home_val = 0.0

    # Extraer EMAs causales limpias de creación y concesión de tiros
    c_sot5 = df.get(f'{p_t}clean_sot_c5', pd.Series(mu0_sot, index=df.index)).fillna(mu0_sot)
    c_sot10 = df.get(f'{p_t}clean_sot_c10', pd.Series(mu0_sot, index=df.index)).fillna(mu0_sot)
    o_allow5 = df.get(f'{p_o}clean_sot_allow5', pd.Series(mu0_sot, index=df.index)).fillna(mu0_sot)
    o_allow10 = df.get(f'{p_o}clean_sot_allow10', pd.Series(mu0_sot, index=df.index)).fillna(mu0_sot)
    t_allow10 = df.get(f'{p_t}clean_sot_allow10', pd.Series(4.3, index=df.index)).fillna(4.3)
    o_sot10 = df.get(f'{p_o}clean_sot_c10', pd.Series(4.3, index=df.index)).fillna(4.3)

    c_st8 = df.get(f'{p_t}clean_st_c8', pd.Series(mu0_st, index=df.index)).fillna(mu0_st)
    o_st_allow8 = df.get(f'{p_o}clean_st_allow8', pd.Series(mu0_st, index=df.index)).fillna(mu0_st)

    v_sot8 = df.get(f'{p_t}venue_sot_c8', c_sot10).fillna(c_sot10)
    v_o_allow8 = df.get(f'{p_o}venue_sot_allow8', o_allow10).fillna(o_allow10)
    v_st8 = df.get(f'{p_t}venue_st_c8', c_st8).fillna(c_st8)
    v_o_st_allow8 = df.get(f'{p_o}venue_st_allow8', o_st_allow8).fillna(o_st_allow8)
    comp_prior = df.get(f'{p_t}comp_venue_sot_prior', pd.Series(mu0_sot, index=df.index)).fillna(mu0_sot).clip(lower=2.5, upper=7.0)

    feat['clean_sot_c5'] = c_sot5
    feat['clean_sot_c10'] = c_sot10
    feat['opp_sot_allow5'] = o_allow5
    feat['opp_sot_allow10'] = o_allow10
    feat['team_sot_allow10'] = t_allow10
    feat['opp_clean_sot_c10'] = o_sot10

    feat['venue_sot_c8'] = v_sot8
    feat['venue_opp_sot_allow8'] = v_o_allow8
    feat['clean_st_c8'] = c_st8
    feat['opp_st_allow8'] = o_st_allow8
    feat['venue_st_c8'] = v_st8
    feat['venue_opp_st_allow8'] = v_o_st_allow8
    feat['comp_venue_sot_prior'] = comp_prior

    # Prior Multiplicativo de Poisson (Global y Específico de Condición Local/Visita)
    feat['poisson_mult_sot_global'] = (c_sot10 * o_allow10) / comp_prior
    feat['poisson_mult_sot_venue'] = (v_sot8 * v_o_allow8) / comp_prior
    feat['poisson_mult_st_venue'] = (v_st8 * v_o_st_allow8) / mu0_st
    feat['log_poisson_mult_venue'] = np.log(feat['poisson_mult_sot_venue'].clip(lower=0.25))

    feat['team_sot_ema3'] = np.where(sot_ema3 > 0, sot_ema3, c_sot5)
    feat['team_sot_ema5'] = np.where(sot_ema5 > 0, sot_ema5, c_sot5)
    feat['opp_sot_ema3'] = np.where(opp_sot_ema3 > 0, opp_sot_ema3, o_sot10)
    feat['opp_sot_ema5'] = np.where(opp_sot_ema5 > 0, opp_sot_ema5, o_sot10)

    feat['team_shots_ema3'] = np.where(st_ema3 > 0, st_ema3, c_st8)
    feat['team_shots_ema5'] = np.where(st_ema5 > 0, st_ema5, c_st8)
    feat['opp_shots_ema3'] = np.where(opp_st_ema3 > 0, opp_st_ema3, o_st_allow8)
    feat['opp_shots_ema5'] = np.where(opp_st_ema5 > 0, opp_st_ema5, o_st_allow8)

    feat['team_poss_ema3'] = poss_ema3
    feat['team_poss_ema5'] = poss_ema5
    feat['opp_poss_ema3'] = opp_poss_ema3
    feat['opp_poss_ema5'] = opp_poss_ema5

    feat['team_pass_ema3'] = pass_ema3
    feat['team_pass_ema5'] = pass_ema5
    feat['opp_pass_ema3'] = opp_pass_ema3
    feat['opp_pass_ema5'] = opp_pass_ema5

    feat['team_att_rating'] = att_rtg
    feat['team_def_rating'] = def_rtg
    feat['opp_att_rating'] = opp_att_rtg
    feat['opp_def_rating'] = opp_def_rtg

    feat['elo_diff'] = elo_diff
    feat['log_squad_ratio'] = np.log(t_val / o_val)
    feat['rest_days'] = df['rest_days'].fillna(7.0) if 'rest_days' in df.columns else 7.0

    # Variables de acoplamiento matemático y priors cuantitativos
    feat['pred_xg_team_quant'] = q_xg_team
    feat['pred_xg_opp_quant'] = q_xg_opp
    feat['log_xg_team_quant'] = np.log(q_xg_team + 0.10)

    # Eficiencia de Puntería con Suavizado Empirical Bayes (Beta-Binomial Shrinkage: prior pi_0 = 0.345, m = 6.0)
    feat['sot_precision_bayes'] = (c_sot10 + 6.0 * 0.345) / (c_st8.clip(lower=0.0) + 6.0)
    feat['analytic_sot_prior'] = feat['poisson_mult_st_venue'] * feat['sot_precision_bayes']

    # Índice de Presión Ofensiva acoplado a la permeabilidad de tiros concedidos del rival
    feat['attack_pressure_index'] = np.sqrt(c_sot5 * o_allow5) * (q_xg_team / 1.25)

    feat['implied_prob_team'] = imp_team
    feat['implied_prob_opp'] = imp_opp
    feat['odds_log_ratio'] = odds_ratio
    feat['market_drift_home'] = df['market_drift_home'].fillna(1.0)
    feat['is_home'] = is_home_val

    return feat


class DualPerspectiveShotsModel(BaseEstimator, RegressorMixin):
    """
    Empaqueta los dos ensambles distributivos simétricos (Home y Away) manteniendo
    la interfaz .predict(df_or_dict) -> np.ndarray de forma (N, 2) [pred_home, pred_away].
    """

    def __init__(self, home_ensemble, away_ensemble, feature_cols):
        self.home_ensemble = home_ensemble
        self.away_ensemble = away_ensemble
        self.feature_cols = feature_cols

    def predict_from_perspectives(self, X_home, X_away):
        pred_h = self.home_ensemble.predict(X_home[self.feature_cols])
        pred_a = self.away_ensemble.predict(X_away[self.feature_cols])
        return np.column_stack([pred_h, pred_a])


def estimate_nb2_params(y_true_h, lam_h, y_true_a, lam_a):
    """
    Estima los parámetros de sobredispersión cuadrática NB2 de Cameron-Trivedi (alpha_H, alpha_A)
    y la correlación residual condicional (rho_res) estrictamente sobre partidos observados.
      Var(Y | lambda) = lambda + alpha * lambda^2
    """
    lam_h_c = np.clip(lam_h, 0.5, None)
    lam_a_c = np.clip(lam_a, 0.5, None)

    res_h = y_true_h - lam_h_c
    res_a = y_true_a - lam_a_c

    # Estimador de momentos ponderado por intensidad de Cameron & Trivedi (2013)
    alpha_h = float(np.clip(np.sum((res_h ** 2) - lam_h_c) / (np.sum(lam_h_c ** 2) + 1e-8), 0.01, 0.35))
    alpha_a = float(np.clip(np.sum((res_a ** 2) - lam_a_c) / (np.sum(lam_a_c ** 2) + 1e-8), 0.01, 0.35))

    if len(res_h) > 2 and np.std(res_h) > 1e-6 and np.std(res_a) > 1e-6:
        rho_res = float(np.clip(np.corrcoef(res_h, res_a)[0, 1], -0.35, 0.35))
    else:
        rho_res = -0.05

    return alpha_h, alpha_a, rho_res


def calc_over_probs_bivariate(lambda_home, lambda_away, lines, alpha_h=0.06, alpha_a=0.06, rho_res=-0.05):
    """
    Calcula probabilidades de superación (Over) para el total del partido T = Y_H + Y_A
    utilizando la ley de varianza condicional cuadrática NB2 de Cameron-Trivedi y la
    correlación condicional de residuales rho_res.
    """
    probs = {}
    lam_h = np.maximum(lambda_home, 0.1)
    lam_a = np.maximum(lambda_away, 0.1)
    lambda_total = lam_h + lam_a

    var_h_i = lam_h + alpha_h * (lam_h ** 2)
    var_a_i = lam_a + alpha_a * (lam_a ** 2)
    cov_i = rho_res * np.sqrt(var_h_i * var_a_i)

    var_tot_i = np.maximum(var_h_i + var_a_i + 2.0 * cov_i, lambda_total * 1.02)

    for line in lines:
        k = int(np.floor(line))
        mean_val = np.maximum(lambda_total, 0.1)
        var_val = np.maximum(var_tot_i, mean_val * 1.02)

        p = np.clip(mean_val / var_val, 1e-5, 0.999)
        n = np.maximum((mean_val ** 2) / (var_val - mean_val), 1e-4)
        prob_over = 1.0 - nbinom.cdf(k, n, p)
        probs[f'prob_over_{line}'] = prob_over
    return probs


def calc_over_probs_team(lambda_vals, lines, alpha_team=0.06):
    """
    Calcula probabilidades de superación (Over) individuales del equipo mediante NB2 condicional.
    """
    probs = {}
    lam = np.maximum(lambda_vals, 0.1)
    var_i = np.maximum(lam + alpha_team * (lam ** 2), lam * 1.02)

    for line in lines:
        k = int(np.floor(line))
        p = np.clip(lam / var_i, 1e-5, 0.999)
        n = np.maximum((lam ** 2) / (var_i - lam), 1e-4)
        probs[f'prob_over_{line}'] = 1.0 - nbinom.cdf(k, n, p)
    return probs


def train_shots_on_goal():
    active_device = check_cuda()

    df = get_base_dataset()

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATASET_PATH = os.path.join(BASE_DIR, 'data/processed/matches_with_referees.parquet')
    FALLBACK_DATASET = os.path.join(BASE_DIR, 'data/processed/matches_with_odds.parquet')
    path_to_load = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET

    df_all = pd.read_parquet(path_to_load, engine='fastparquet')

    # 1. Extraer target y tiros totales del visitante (Away Team) emparejado por match_id
    away_cols = ['match_id', 'shots_on_target']
    if 'shots_total' in df_all.columns:
        away_cols.append('shots_total')
    away_data = df_all[df_all['is_home'] == 0][away_cols].copy()
    rename_map = {'shots_on_target': 'away_shots_on_target'}
    if 'shots_total' in away_cols:
        rename_map['shots_total'] = 'away_shots_total'
    away_data = away_data.rename(columns=rename_map).drop_duplicates(subset=['match_id'])

    df = df.merge(away_data, on='match_id', how='left')
    if 'away_shots_total' not in df.columns:
        df['away_shots_total'] = 0.0

    # 2. INGENIERÍA DE CARACTERÍSTICAS DE MERCADO (Cuotas de Apuestas Implícitas)
    logger.info("Calculando probabilidades e indicadores predictivos del mercado de cuotas...")
    if 'odds_win' in df.columns and 'odds_loss' in df.columns:
        inv_win = 1.0 / df['odds_win'].clip(lower=1.01)
        inv_draw = (1.0 / df['odds_draw'].clip(lower=1.01)) if 'odds_draw' in df.columns else 0.0
        inv_loss = 1.0 / df['odds_loss'].clip(lower=1.01)
        margin = inv_win + inv_draw + inv_loss

        df['implied_prob_home'] = inv_win / margin
        df['implied_prob_away'] = inv_loss / margin
        df['odds_ratio_home_away'] = np.log(df['odds_win'].clip(lower=1.01) / df['odds_loss'].clip(lower=1.01))

        if 'closing_odds_win' in df.columns and 'open_odds_win' in df.columns and df['open_odds_win'].notna().any():
            raw_drift = (df['closing_odds_win'] / df['open_odds_win'].clip(lower=1.01)).fillna(1.0)
            df['market_drift_home'] = (
                raw_drift.groupby(df['team'], sort=False).shift(1).fillna(1.0)
                if 'team' in df.columns
                else raw_drift.shift(1).fillna(1.0)
            )
        else:
            df['market_drift_home'] = 1.0
    else:
        df['implied_prob_home'] = 0.5
        df['implied_prob_away'] = 0.5
        df['odds_ratio_home_away'] = 0.0
        df['market_drift_home'] = 1.0

    df['shots_on_target'] = df['shots_on_target'].fillna(0.0)
    df['away_shots_on_target'] = df['away_shots_on_target'].fillna(0.0)
    df['shots_total'] = df['shots_total'].fillna(0.0)
    df['away_shots_total'] = df['away_shots_total'].fillna(0.0)

    # Máscara de observabilidad: identifica partidos donde realmente se registraron estadísticas de tiros
    obs_mask = ((df['shots_total'] + df['away_shots_total']) > 0) | (
        (df['shots_on_target'] + df['away_shots_on_target']) > 0
    )

    split_idx = get_train_test_split(df)

    # === CARGAR xG DEL MODELO QUANT ADVANCED ===
    quant_train_path = os.path.join(PROCESSED_DIR, 'oof_quant_train.parquet')
    quant_test_path = os.path.join(PROCESSED_DIR, 'oof_quant_test.parquet')

    if os.path.exists(quant_train_path) and os.path.exists(quant_test_path):
        logger.info("Cargando variables xG del modelo Quant Advanced...")
        q_tr = pd.read_parquet(quant_train_path)
        q_ts = pd.read_parquet(quant_test_path)
        q_full = pd.concat([q_tr, q_ts]).reset_index(drop=True)

        if len(q_full) == len(df):
            df['predicted_xg_scored_quant'] = q_full['predicted_xg_scored_quant'].values
            df['predicted_xg_conceded_quant'] = q_full['predicted_xg_conceded_quant'].values
        elif 'match_id' in q_full.columns:
            df = df.merge(
                q_full[['match_id', 'predicted_xg_scored_quant', 'predicted_xg_conceded_quant']],
                on='match_id',
                how='left',
            )
            df['predicted_xg_scored_quant'] = df['predicted_xg_scored_quant'].fillna(1.30)
            df['predicted_xg_conceded_quant'] = df['predicted_xg_conceded_quant'].fillna(1.10)
        else:
            logger.warning("No se pudo alinear oof_quant por diferencia de longitud. Usando fallback.")
            df['predicted_xg_scored_quant'] = 1.30
            df['predicted_xg_conceded_quant'] = 1.10
    else:
        logger.warning("No se encontraron predicciones Quant. Ejecuta train_quant_advanced.py primero.")
        df['predicted_xg_scored_quant'] = 1.30
        df['predicted_xg_conceded_quant'] = 1.10

    # Enriquecer dinámicas causales limpias de tiros creados y concedidos (globales y por condición Local/Visita)
    logger.info("Calculando dinámicas causales limpias de tiros creados y concedidos (enmascarando ceros no registrados)...")
    df = enrich_clean_causal_shot_dynamics(df, df_all)

    # Construir matrices simétricas por perspectiva (Home y Away)
    X_home_all = build_perspective_shots_features(df, perspective='home')
    X_away_all = build_perspective_shots_features(df, perspective='away')
    feature_cols = list(X_home_all.columns)

    y_home_all = df['shots_on_target'].copy()
    y_away_all = df['away_shots_on_target'].copy()

    X_home_tr, X_home_ts = X_home_all.iloc[:split_idx].copy(), X_home_all.iloc[split_idx:].copy()
    X_away_tr, X_away_ts = X_away_all.iloc[:split_idx].copy(), X_away_all.iloc[split_idx:].copy()
    y_home_tr, y_home_ts = y_home_all.iloc[:split_idx].copy(), y_home_all.iloc[split_idx:].copy()
    y_away_tr, y_away_ts = y_away_all.iloc[:split_idx].copy(), y_away_all.iloc[split_idx:].copy()
    obs_tr = obs_mask.iloc[:split_idx].values
    obs_ts = obs_mask.iloc[split_idx:].values

    logger.info(
        f"Partidos con estadísticas reales de tiros -> Train: {obs_tr.sum()}/{len(obs_tr)} ({obs_tr.mean()*100:.1f}%) | "
        f"Test: {obs_ts.sum()}/{len(obs_ts)} ({obs_ts.mean()*100:.1f}%)"
    )

    train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx]) if 'match_date' in df.columns else None
    input_dim = len(feature_cols)

    logger.info("=== ENTRENANDO ENSAMBLE DISTRIBUTIVO DUAL (HOME + AWAY) DE TIROS AL ARCO ===")

    # Filtrar solo partidos observados para Optuna
    X_h_obs = X_home_tr.iloc[obs_tr].reset_index(drop=True)
    X_a_obs = X_away_tr.iloc[obs_tr].reset_index(drop=True)
    y_h_obs = y_home_tr.iloc[obs_tr].reset_index(drop=True)
    y_a_obs = y_away_tr.iloc[obs_tr].reset_index(drop=True)
    dates_obs = train_dates.iloc[obs_tr].reset_index(drop=True) if train_dates is not None else None

    opt_split = int(len(X_h_obs) * 0.80)
    w_opt_tr = get_time_weights(dates_obs.iloc[:opt_split]) if dates_obs is not None else None

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 2)
        hidden_dims = [trial.suggest_categorical(f'n_units_l{i}', [32, 64, 128]) for i in range(n_layers)]
        dropout_rate = trial.suggest_float('dropout_rate', 0.15, 0.35)
        lr = trial.suggest_float('lr', 5e-4, 3e-3, log=True)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.5, 50.0, log=True)
        meta_alpha = trial.suggest_float('meta_alpha', 0.1, 10.0, log=True)

        xgb_n_est = trial.suggest_int('xgb_n_estimators', 120, 320)
        xgb_lr = trial.suggest_float('xgb_lr', 0.02, 0.08, log=True)
        xgb_depth = trial.suggest_int('xgb_max_depth', 3, 6)
        xgb_sub = trial.suggest_float('xgb_subsample', 0.65, 0.90)
        xgb_col = trial.suggest_float('xgb_colsample', 0.65, 0.90)

        mlp_params = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'lr': lr,
            'epochs': 35,
            'batch_size': 256,
            'patience': 4,
            'device': active_device,
        }
        xgb_params = {
            'objective': 'count:poisson',
            'n_estimators': xgb_n_est,
            'learning_rate': xgb_lr,
            'max_depth': xgb_depth,
            'subsample': xgb_sub,
            'colsample_bytree': xgb_col,
            'min_child_weight': 5,
            'random_state': 42,
            'n_jobs': -1,
        }

        ens_h = PerspectiveDistributionalEnsemble(mlp_params, xgb_params, ridge_alpha, meta_alpha, cv_splits=3)
        ens_a = PerspectiveDistributionalEnsemble(mlp_params, xgb_params, ridge_alpha, meta_alpha, cv_splits=3)

        ens_h.fit(X_h_obs.iloc[:opt_split], y_h_obs.iloc[:opt_split], sample_weight=w_opt_tr)
        ens_a.fit(X_a_obs.iloc[:opt_split], y_a_obs.iloc[:opt_split], sample_weight=w_opt_tr)

        pred_h = np.clip(ens_h.predict(X_h_obs.iloc[opt_split:]), 1e-4, None)
        pred_a = np.clip(ens_a.predict(X_a_obs.iloc[opt_split:]), 1e-4, None)

        yv_h = y_h_obs.iloc[opt_split:].values
        yv_a = y_a_obs.iloc[opt_split:].values

        nll_h = np.mean(pred_h - yv_h * np.log(pred_h))
        nll_a = np.mean(pred_a - yv_a * np.log(pred_a))
        return 0.5 * (nll_h + nll_a)

    if RUN_OPTUNA or not os.path.exists(OPTUNA_PARAMS_FILE):
        logger.info(f"Optimizando Ensamble Distributivo Dual con Optuna ({OPTUNA_TRIALS} Trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Mejores parámetros guardados en {OPTUNA_PARAMS_FILE}")
    else:
        logger.info("Cargando mejores parámetros guardados...")
        with open(OPTUNA_PARAMS_FILE, 'r') as f:
            best_params = json.load(f)

    best_mlp = {
        'input_dim': input_dim,
        'hidden_dims': [best_params.get(f'n_units_l{i}', 64) for i in range(best_params.get('n_layers', 2))],
        'dropout_rate': best_params.get('dropout_rate', 0.25),
        'lr': best_params.get('lr', 1e-3),
        'epochs': 65,
        'batch_size': 256,
        'patience': 6,
        'device': active_device,
    }
    best_xgb = {
        'objective': 'count:poisson',
        'n_estimators': best_params.get('xgb_n_estimators', 220),
        'learning_rate': best_params.get('xgb_lr', 0.04),
        'max_depth': best_params.get('xgb_max_depth', 5),
        'subsample': best_params.get('xgb_subsample', 0.80),
        'colsample_bytree': best_params.get('xgb_colsample', 0.80),
        'min_child_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
    }
    r_alpha = best_params.get('ridge_alpha', 2.0)
    m_alpha = best_params.get('meta_alpha', 1.0)

    # OUT-OF-FOLD PREDICTIONS TEMPORALES (Entrenando solo en partidos observados, prediciendo el 100% de filas)
    tscv = get_cv_strategy(n_splits=5)
    preds_train = np.full((len(X_home_tr), 2), np.nan, dtype=np.float64)
    splits = list(tscv.split(X_home_tr))

    first_train_idx = splits[0][0]
    inner_tscv = TimeSeriesSplit(n_splits=3)
    logger.info(f"  -> Fold Inicial TimeSeriesSplit ({len(first_train_idx)} muestras totales)...")
    for tr_in, val_in in inner_tscv.split(first_train_idx):
        tr_idx_orig = first_train_idx[tr_in]
        val_idx_orig = first_train_idx[val_in]
        tr_obs_idx = tr_idx_orig[obs_tr[tr_idx_orig]]

        if len(tr_obs_idx) < 50:
            continue

        dates_in_tr = train_dates.iloc[tr_obs_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_in_tr) if dates_in_tr is not None else None

        ens_h = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
        ens_a = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
        ens_h.fit(X_home_tr.iloc[tr_obs_idx], y_home_tr.iloc[tr_obs_idx], sample_weight=w_tr)
        ens_a.fit(X_away_tr.iloc[tr_obs_idx], y_away_tr.iloc[tr_obs_idx], sample_weight=w_tr)

        preds_train[val_idx_orig, 0] = ens_h.predict(X_home_tr.iloc[val_idx_orig])
        preds_train[val_idx_orig, 1] = ens_a.predict(X_away_tr.iloc[val_idx_orig])

    # Imputar el bloque inicial de burn-in con media expansiva histórica previa sobre partidos observados
    first_unpredicted = first_train_idx[np.isnan(preds_train[first_train_idx, 0])]
    if len(first_unpredicted) > 0:
        yh_masked = y_home_tr.iloc[first_unpredicted].where(obs_tr[first_unpredicted], np.nan)
        ya_masked = y_away_tr.iloc[first_unpredicted].where(obs_tr[first_unpredicted], np.nan)
        burn_h = yh_masked.shift(1).expanding(min_periods=1).mean().fillna(4.75).values
        burn_a = ya_masked.shift(1).expanding(min_periods=1).mean().fillna(3.90).values
        preds_train[first_unpredicted, 0] = burn_h
        preds_train[first_unpredicted, 1] = burn_a

    # Expanding Windows CV principal
    for i, (train_idx, val_idx) in enumerate(splits):
        tr_obs_idx = train_idx[obs_tr[train_idx]]
        logger.info(f"  -> Fold Temporal {i+1}/{len(splits)} ({len(tr_obs_idx)} partidos observados en train)...")
        w_tr = get_time_weights(train_dates.iloc[tr_obs_idx]) if train_dates is not None else None

        ens_h = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
        ens_a = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
        ens_h.fit(X_home_tr.iloc[tr_obs_idx], y_home_tr.iloc[tr_obs_idx], sample_weight=w_tr)
        ens_a.fit(X_away_tr.iloc[tr_obs_idx], y_away_tr.iloc[tr_obs_idx], sample_weight=w_tr)

        preds_train[val_idx, 0] = ens_h.predict(X_home_tr.iloc[val_idx])
        preds_train[val_idx, 1] = ens_a.predict(X_away_tr.iloc[val_idx])

    logger.info("Entrenando Modelo Final Dual en todos los partidos observados de Train...")
    final_weights = get_time_weights(dates_obs)
    final_ens_h = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
    final_ens_a = PerspectiveDistributionalEnsemble(best_mlp, best_xgb, r_alpha, m_alpha, cv_splits=3)
    final_ens_h.fit(X_h_obs, y_h_obs, sample_weight=final_weights)
    final_ens_a.fit(X_a_obs, y_a_obs, sample_weight=final_weights)

    final_dual_model = DualPerspectiveShotsModel(final_ens_h, final_ens_a, feature_cols)
    preds_test = final_dual_model.predict_from_perspectives(X_home_ts, X_away_ts)

    # --- PARAMETRIZACIÓN NB2 DE CAMERON-TRIVEDI Y LÍNEAS DE APUESTAS ---
    logger.info("=== CALCULANDO VARIABILIDAD CONDICIONAL NB2 (CAMERON-TRIVEDI) Y LÍNEAS OVER/UNDER ===")

    df_train_tmp = df.iloc[:split_idx].copy()
    df_train_tmp['pred_shots'] = preds_train[:, 0]
    df_train_tmp['opp_pred_shots'] = preds_train[:, 1]
    df_train_tmp['lambda_total'] = df_train_tmp['pred_shots'] + df_train_tmp['opp_pred_shots']
    df_train_tmp['opp_shots'] = df_train_tmp['away_shots_on_target']
    df_train_tmp['true_total_shots'] = df_train_tmp['shots_on_target'] + df_train_tmp['opp_shots']

    df_test_tmp = df.iloc[split_idx:].copy()
    df_test_tmp['pred_shots'] = preds_test[:, 0]
    df_test_tmp['opp_pred_shots'] = preds_test[:, 1]
    df_test_tmp['lambda_total'] = df_test_tmp['pred_shots'] + df_test_tmp['opp_pred_shots']
    df_test_tmp['opp_shots'] = df_test_tmp['away_shots_on_target']
    df_test_tmp['true_total_shots'] = df_test_tmp['shots_on_target'] + df_test_tmp['opp_shots']

    team_lines = [3.5, 4.5, 5.5]
    total_lines = [7.5, 8.5, 9.5]

    alpha_h, alpha_a, rho_res = estimate_nb2_params(
        df_train_tmp['shots_on_target'].values[obs_tr],
        df_train_tmp['pred_shots'].values[obs_tr],
        df_train_tmp['opp_shots'].values[obs_tr],
        df_train_tmp['opp_pred_shots'].values[obs_tr],
    )
    logger.info(
        f"Parámetros Condicionales NB2 (Partidos Observados) -> "
        f"alpha_Home: {alpha_h:.4f} | alpha_Away: {alpha_a:.4f} | rho_residual(H,A): {rho_res:.4f}"
    )

    train_team_probs = calc_over_probs_team(df_train_tmp['pred_shots'].values, team_lines, alpha_h)
    test_team_probs = calc_over_probs_team(df_test_tmp['pred_shots'].values, team_lines, alpha_h)
    for col, vals in train_team_probs.items():
        df_train_tmp[col + '_team'] = vals
    for col, vals in test_team_probs.items():
        df_test_tmp[col + '_team'] = vals

    train_total_probs = calc_over_probs_bivariate(
        df_train_tmp['pred_shots'].values,
        df_train_tmp['opp_pred_shots'].values,
        total_lines,
        alpha_h,
        alpha_a,
        rho_res,
    )
    test_total_probs = calc_over_probs_bivariate(
        df_test_tmp['pred_shots'].values,
        df_test_tmp['opp_pred_shots'].values,
        total_lines,
        alpha_h,
        alpha_a,
        rho_res,
    )
    for col, vals in train_total_probs.items():
        df_train_tmp[col + '_total'] = vals
    for col, vals in test_total_probs.items():
        df_test_tmp[col + '_total'] = vals

    # Calibración Isotónica sobre OOF temporal (ajustada EXCLUSIVAMENTE en partidos observados)
    calibrators = {}
    obs_indices_tr = np.where(obs_tr)[0]
    cal_splits = list(TimeSeriesSplit(n_splits=5).split(obs_indices_tr))

    for line in team_lines:
        col = f'prob_over_{line}_team'
        true_over_train = (df_train_tmp['shots_on_target'] > line).astype(int).values
        raw_p = df_train_tmp[col].values.copy()
        oof_cal_p = raw_p.copy()
        for c_tr, c_va in cal_splits:
            tr_pos = obs_indices_tr[c_tr]
            va_pos = obs_indices_tr[c_va]
            ir_fold = IsotonicRegression(out_of_bounds='clip')
            ir_fold.fit(raw_p[tr_pos], true_over_train[tr_pos])
            oof_cal_p[va_pos] = ir_fold.predict(raw_p[va_pos])

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_p[obs_tr], true_over_train[obs_tr])
        oof_cal_p[~obs_tr] = ir.predict(raw_p[~obs_tr])
        df_train_tmp[col] = oof_cal_p
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir

    for line in total_lines:
        col = f'prob_over_{line}_total'
        true_over_train = (df_train_tmp['true_total_shots'] > line).astype(int).values
        raw_p = df_train_tmp[col].values.copy()
        oof_cal_p = raw_p.copy()
        for c_tr, c_va in cal_splits:
            tr_pos = obs_indices_tr[c_tr]
            va_pos = obs_indices_tr[c_va]
            ir_fold = IsotonicRegression(out_of_bounds='clip')
            ir_fold.fit(raw_p[tr_pos], true_over_train[tr_pos])
            oof_cal_p[va_pos] = ir_fold.predict(raw_p[va_pos])

        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_p[obs_tr], true_over_train[obs_tr])
        oof_cal_p[~obs_tr] = ir.predict(raw_p[~obs_tr])
        df_train_tmp[col] = oof_cal_p
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir

    # AUDITORÍA CONTINUA SOBRE PARTIDOS CON ESTADÍSTICAS REALES REGISTRADAS
    logger.info("--- REGRESIÓN PURA (MSE/MAE) EN PARTIDOS OBSERVADOS ---")
    mse_h_ts = mean_squared_error(df_test_tmp['shots_on_target'].values[obs_ts], df_test_tmp['pred_shots'].values[obs_ts])
    mae_h_ts = mean_absolute_error(df_test_tmp['shots_on_target'].values[obs_ts], df_test_tmp['pred_shots'].values[obs_ts])
    mse_a_ts = mean_squared_error(df_test_tmp['opp_shots'].values[obs_ts], df_test_tmp['opp_pred_shots'].values[obs_ts])
    mae_a_ts = mean_absolute_error(df_test_tmp['opp_shots'].values[obs_ts], df_test_tmp['opp_pred_shots'].values[obs_ts])

    mse_tr = mean_squared_error(df_train_tmp['true_total_shots'].values[obs_tr], df_train_tmp['lambda_total'].values[obs_tr])
    mae_tr = mean_absolute_error(df_train_tmp['true_total_shots'].values[obs_tr], df_train_tmp['lambda_total'].values[obs_tr])
    mse_ts = mean_squared_error(df_test_tmp['true_total_shots'].values[obs_ts], df_test_tmp['lambda_total'].values[obs_ts])
    mae_ts = mean_absolute_error(df_test_tmp['true_total_shots'].values[obs_ts], df_test_tmp['lambda_total'].values[obs_ts])

    logger.info(f"Test Home Shots  - MSE: {mse_h_ts:.3f} | MAE: {mae_h_ts:.3f}")
    logger.info(f"Test Away Shots  - MSE: {mse_a_ts:.3f} | MAE: {mae_a_ts:.3f}")
    logger.info(f"Train Total Shots - MSE: {mse_tr:.3f} | MAE: {mae_tr:.3f}")
    logger.info(f"Test  Total Shots - MSE: {mse_ts:.3f} | MAE: {mae_ts:.3f}")

    logger.info("--- PROBABILIDADES TOTAL MATCH BIVARIADAS NB2 (Over/Under en Test Observado) ---")
    for line in total_lines:
        true_over_ts = (df_test_tmp['true_total_shots'].values[obs_ts] > line).astype(int)
        prob_ts = df_test_tmp[f'prob_over_{line}_total'].values[obs_ts]
        ll_ts = log_loss(true_over_ts, prob_ts)
        bs_ts = brier_score_loss(true_over_ts, prob_ts)
        acc_ts = accuracy_score(true_over_ts, (prob_ts > 0.5).astype(int))
        logger.info(
            f"OVER {line} Total | Dist: {true_over_ts.mean()*100:.1f}% | "
            f"LogLoss: {ll_ts:.4f} | Brier: {bs_ts:.4f} | Acc: {acc_ts:.4f}"
        )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cols_to_save = (
        ['pred_shots', 'opp_pred_shots', 'lambda_total']
        + [f'prob_over_{L}_team' for L in team_lines]
        + [f'prob_over_{L}_total' for L in total_lines]
    )

    oof_train = df_train_tmp[cols_to_save].copy()
    oof_test = df_test_tmp[cols_to_save].copy()

    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_shots_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_shots_test.parquet'), engine='fastparquet')

    joblib.dump(
        {
            'model_shots': final_dual_model,
            'features': feature_cols,
            'team_lines': team_lines,
            'total_lines': total_lines,
            'alpha_h': alpha_h,
            'alpha_a': alpha_a,
            'rho_res': rho_res,
            'calibrators': calibrators,
        },
        MODEL_SAVE_PATH,
    )
    logger.info(f"=== MODELO TIROS AL ARCO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    from core.models.train_shots_on_goal import train_shots_on_goal as _run_main

    _run_main()
