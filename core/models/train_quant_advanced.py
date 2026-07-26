import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import poisson
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Asegurar import de data_splitter y logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_quant_advanced')

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'quant_advanced_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))


def calc_dixon_coles_probabilities_vectorized(mu_home, mu_away, rho, max_goals=10):
    """
    Calcula las probabilidades vectorizadas de Victoria, Empate y Derrota usando
    distribución Poisson bivariada con ajuste de verosimilitud exacta Dixon-Coles.
    """
    mu_h = np.asarray(mu_home).reshape(-1, 1, 1)
    mu_a = np.asarray(mu_away).reshape(-1, 1, 1)
    
    x = np.arange(max_goals + 1).reshape(1, max_goals + 1, 1)
    y = np.arange(max_goals + 1).reshape(1, 1, max_goals + 1)
    
    # PMFs Poisson marginales
    pmf_home = poisson.pmf(x, mu_h)
    pmf_away = poisson.pmf(y, mu_a)
    
    # Probabilidad conjunta independiente
    prob_matrix = pmf_home * pmf_away
    
    # Ajuste Dixon-Coles para marcadores bajos (0,0), (1,0), (0,1), (1,1)
    if rho != 0.0:
        mh = mu_h.squeeze(axis=-1)
        ma = mu_a.squeeze(axis=-1)
        
        tau_00 = np.clip(1.0 - mh * ma * rho, 0.0, None)
        tau_10 = np.clip(1.0 + ma * rho, 0.0, None)
        tau_01 = np.clip(1.0 + mh * rho, 0.0, None)
        tau_11 = np.clip(1.0 - rho, 0.0, None)
        
        prob_matrix[:, 0, 0] *= tau_00.squeeze(axis=-1)
        prob_matrix[:, 1, 0] *= tau_10.squeeze(axis=-1)
        prob_matrix[:, 0, 1] *= tau_01.squeeze(axis=-1)
        prob_matrix[:, 1, 1] *= tau_11.squeeze(axis=-1)
        
        # Renormalizar constante Z para que las probabilidades sumen exactamente 1.0
        sums = prob_matrix.sum(axis=(1, 2), keepdims=True)
        prob_matrix /= np.where(sums > 0, sums, 1.0)
        
    win_prob = np.tril(prob_matrix, -1).sum(axis=(1, 2))
    draw_prob = np.diagonal(prob_matrix, axis1=1, axis2=2).sum(axis=1)
    loss_prob = np.triu(prob_matrix, 1).sum(axis=(1, 2))
    
    return win_prob, draw_prob, loss_prob


class KalmanDixonColesQuantModel(nn.Module):
    """
    Modelo Bayesiano Cuantitativo Dixon-Coles con Filtro de Kalman Smoother (Estado-Espacio Dinámico).
    Ajusta dinámicamente el decaimiento temporal y la varianza de transición de fuerzas.
    """
    def __init__(self, n_teams):
        super().__init__()
        self.n_teams = n_teams
        self.intercept = nn.Parameter(torch.tensor([np.log(1.2)], dtype=torch.float32))
        self.home_adv = nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
        self.att_star = nn.Parameter(torch.zeros(n_teams, dtype=torch.float32))
        self.def_star = nn.Parameter(torch.zeros(n_teams, dtype=torch.float32))
        self.rho = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.w_elo = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.w_xg = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.kalman_gamma_logit = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32))

    def get_att_def(self):
        # Restricción suma cero para identificabilidad de parámetros
        att = self.att_star - self.att_star.mean()
        def_ = self.def_star - self.def_star.mean()
        return att, def_

    def forward(self, h_idx, a_idx, elo_diff=None, xg_diff=None):
        att, def_ = self.get_att_def()
        
        # Categoría dummy para equipos no vistos en train (fuerza 0.0)
        att_ext = torch.cat([att, torch.zeros(1, dtype=torch.float32)])
        def_ext = torch.cat([def_, torch.zeros(1, dtype=torch.float32)])
        
        h_idx_safe = torch.where((h_idx >= 0) & (h_idx < self.n_teams), h_idx, torch.tensor(self.n_teams))
        a_idx_safe = torch.where((a_idx >= 0) & (a_idx < self.n_teams), a_idx, torch.tensor(self.n_teams))
        
        feat_h = 0.0
        feat_a = 0.0
        if elo_diff is not None:
            feat_h = feat_h + self.w_elo * elo_diff
            feat_a = feat_a - self.w_elo * elo_diff
        if xg_diff is not None:
            feat_h = feat_h + self.w_xg * xg_diff
            feat_a = feat_a - self.w_xg * xg_diff
            
        log_mu_h = self.intercept + self.home_adv + att_ext[h_idx_safe] + def_ext[a_idx_safe] + feat_h
        log_mu_a = self.intercept + att_ext[a_idx_safe] + def_ext[h_idx_safe] + feat_a
        
        mu_h = torch.exp(log_mu_h)
        mu_a = torch.exp(log_mu_a)
        return mu_h, mu_a

    def compute_loss(self, h_idx, a_idx, h_goals, a_goals, days_ago, elo_diff=None, xg_diff=None):
        mu_h, mu_a = self.forward(h_idx, a_idx, elo_diff, xg_diff)
        
        # Ponderación dinámica de Kalman Smoother: w(t) = 1 / (1 + gamma * days_ago)
        gamma = 0.0001 + 0.0099 * torch.sigmoid(self.kalman_gamma_logit)
        kalman_weights = 1.0 / (1.0 + gamma * days_ago)
        kalman_weights = kalman_weights / kalman_weights.mean()
        
        # Log-Likelihood Poisson marginales
        log_p_h = h_goals.float() * torch.log(mu_h + 1e-8) - mu_h - torch.lgamma(h_goals.float() + 1.0)
        log_p_a = a_goals.float() * torch.log(mu_a + 1e-8) - mu_a - torch.lgamma(a_goals.float() + 1.0)
        
        # Ajuste Dixon-Coles tau
        is_00 = (h_goals == 0) & (a_goals == 0)
        is_10 = (h_goals == 1) & (a_goals == 0)
        is_01 = (h_goals == 0) & (a_goals == 1)
        is_11 = (h_goals == 1) & (a_goals == 1)
        
        tau = torch.ones_like(mu_h)
        tau = torch.where(is_00, 1.0 - mu_h * mu_a * self.rho, tau)
        tau = torch.where(is_10, 1.0 + mu_a * self.rho, tau)
        tau = torch.where(is_01, 1.0 + mu_h * self.rho, tau)
        tau = torch.where(is_11, 1.0 - self.rho, tau)
        tau = torch.clamp(tau, min=1e-5)
        
        # Factor de normalización exacta Z(mu_h, mu_a, rho)
        p_h0 = torch.exp(-mu_h)
        p_h1 = mu_h * p_h0
        p_a0 = torch.exp(-mu_a)
        p_a1 = mu_a * p_a0
        
        Z = 1.0 + self.rho * (
            - mu_h * mu_a * (p_h0 * p_a0)
            + mu_a * (p_h1 * p_a0)
            + mu_h * (p_h0 * p_a1)
            - (p_h1 * p_a1)
        )
        Z = torch.clamp(Z, min=1e-5)
        
        log_likelihood = log_p_h + log_p_a + torch.log(tau) - torch.log(Z)
        weighted_nll = - torch.mean(kalman_weights * log_likelihood)
        
        # Priors Bayesianos (Regularización L2)
        att, def_ = self.get_att_def()
        prior_att = 0.5 * torch.sum(att**2) / (0.4**2)
        prior_def = 0.5 * torch.sum(def_**2) / (0.4**2)
        prior_home = 0.5 * ((self.home_adv - 0.25)**2) / (0.1**2)
        prior_rho = 0.5 * (self.rho**2) / (0.1**2)
        prior_feats = 0.5 * (self.w_elo**2 + self.w_xg**2) / (0.5**2)
        
        n_obs = len(h_goals)
        loss = weighted_nll + (prior_att + prior_def + prior_home + prior_rho + prior_feats) / n_obs
        return loss


def fit_quant_model(df_sub, team_mapping, n_teams, max_iter=100):
    """
    Ajusta el modelo cuantitativo Kalman Dixon-Coles en un subconjunto de datos usando L-BFGS optimizado.
    """
    h_idx = torch.tensor(df_sub['team'].map(lambda t: team_mapping.get(t, -1)).values, dtype=torch.long)
    a_idx = torch.tensor(df_sub['opponent'].map(lambda t: team_mapping.get(t, -1)).values, dtype=torch.long)
    h_goals = torch.tensor(df_sub['goals_scored'].values, dtype=torch.long)
    a_goals = torch.tensor(df_sub['goals_conceded'].values, dtype=torch.long)

    dates = pd.to_datetime(df_sub['match_date'])
    max_date = dates.max()
    days_ago = torch.tensor((max_date - dates).dt.days.values, dtype=torch.float32)

    elo_diff = torch.tensor(df_sub['elo_diff'].fillna(0.0).values / 100.0, dtype=torch.float32) if 'elo_diff' in df_sub.columns else None
    
    if 'xg_created_ema5' in df_sub.columns and 'xg_conceded_ema5' in df_sub.columns:
        xg_diff = torch.tensor((df_sub['xg_created_ema5'].fillna(0.0) - df_sub['xg_conceded_ema5'].fillna(0.0)).values, dtype=torch.float32)
    else:
        xg_diff = None

    model = KalmanDixonColesQuantModel(n_teams)
    optimizer = optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = model.compute_loss(h_idx, a_idx, h_goals, a_goals, days_ago, elo_diff, xg_diff)
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def predict_from_model(model, team_mapping, df_test):
    """
    Realiza inferencia vectorizada out-of-fold o out-of-sample dada una instancia ajustada.
    """
    h_idx = torch.tensor(df_test['team'].map(lambda t: team_mapping.get(t, -1)).values, dtype=torch.long)
    a_idx = torch.tensor(df_test['opponent'].map(lambda t: team_mapping.get(t, -1)).values, dtype=torch.long)
    
    elo_diff = torch.tensor(df_test['elo_diff'].fillna(0.0).values / 100.0, dtype=torch.float32) if 'elo_diff' in df_test.columns else None
    
    if 'xg_created_ema5' in df_test.columns and 'xg_conceded_ema5' in df_test.columns:
        xg_diff = torch.tensor((df_test['xg_created_ema5'].fillna(0.0) - df_test['xg_conceded_ema5'].fillna(0.0)).values, dtype=torch.float32)
    else:
        xg_diff = None

    with torch.no_grad():
        mu_h, mu_a = model(h_idx, a_idx, elo_diff, xg_diff)
        
    mu_h_np = mu_h.numpy()
    mu_a_np = mu_a.numpy()
    rho_val = model.rho.item()
    
    w, d, l = calc_dixon_coles_probabilities_vectorized(mu_h_np, mu_a_np, rho_val)
    return w, d, l, mu_h_np, mu_a_np


def train_quant_advanced():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    teams = pd.concat([df_train['team'], df_train['opponent']]).unique()
    team_mapping = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    
    logger.info(f"=== ENTRENANDO MODELO QUANT AVANZADO (Kalman Dixon-Coles Bayesiano Vectorizado) ===")
    logger.info(f" - Muestras Train: {len(df_train)}, Test: {len(df_test)}, Equipos Unicos: {n_teams}")
    
    t0 = time.time()
    
    # === GENERACIÓN DE PREDICCIONES OUT-OF-FOLD (OOF) REALES CON TimeSeriesSplit ===
    tscv = get_cv_strategy(n_splits=5)
    oof_win = np.zeros(len(df_train))
    oof_draw = np.zeros(len(df_train))
    oof_loss = np.zeros(len(df_train))
    oof_scored = np.zeros(len(df_train))
    oof_conceded = np.zeros(len(df_train))
    
    logger.info("Generando predicciones OOF reales con TimeSeriesSplit(5)...")
    splits = list(tscv.split(df_train))
    
    # 1. Procesar primer bloque inicial con KFold interno sin leakage
    first_train_idx = splits[0][0]
    df_first = df_train.iloc[first_train_idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_tr, kf_val in kf.split(df_first):
        df_tr_kf, df_val_kf = df_first.iloc[kf_tr], df_first.iloc[kf_val]
        m_kf = fit_quant_model(df_tr_kf, team_mapping, n_teams, max_iter=60)
        w_v, d_v, l_v, s_v, c_v = predict_from_model(m_kf, team_mapping, df_val_kf)
        
        orig_indices = first_train_idx[kf_val]
        oof_win[orig_indices] = w_v
        oof_draw[orig_indices] = d_v
        oof_loss[orig_indices] = l_v
        oof_scored[orig_indices] = s_v
        oof_conceded[orig_indices] = c_v
        
    # 2. Procesar folds temporales expandidos
    for i, (train_idx, val_idx) in enumerate(splits):
        df_tr_fold = df_train.iloc[train_idx]
        df_val_fold = df_train.iloc[val_idx]
        
        m_fold = fit_quant_model(df_tr_fold, team_mapping, n_teams, max_iter=60)
        w_v, d_v, l_v, s_v, c_v = predict_from_model(m_fold, team_mapping, df_val_fold)
        
        oof_win[val_idx] = w_v
        oof_draw[val_idx] = d_v
        oof_loss[val_idx] = l_v
        oof_scored[val_idx] = s_v
        oof_conceded[val_idx] = c_v

    # === ENTRENAMIENTO FINAL SOBRE TODO EL TRAIN SET Y TEST INFERENCIA ===
    logger.info("Ajustando modelo cuantitativo final sobre todo el Train set...")
    final_model = fit_quant_model(df_train, team_mapping, n_teams, max_iter=100)
    
    win_ts, draw_ts, loss_ts, scored_ts, conceded_ts = predict_from_model(final_model, team_mapping, df_test)
    
    t1 = time.time()
    logger.info(f"Entrenamiento e Inferencia completados en {t1 - t0:.2f} segundos!")
    
    # === AUDITORÍA LOG-LOSS Y CALIBRACIÓN ===
    y_train = df_train['outcome'].replace({-1: 0, 0: 1, 1: 2})
    pred_probs_train_oof = np.column_stack((oof_loss, oof_draw, oof_win))
    
    y_test = df_test['outcome'].replace({-1: 0, 0: 1, 1: 2})
    pred_probs_test = np.column_stack((loss_ts, draw_ts, win_ts))
    
    oof_logloss = log_loss(y_train, pred_probs_train_oof)
    test_logloss = log_loss(y_test, pred_probs_test)
    
    learned_gamma = (0.0001 + 0.0099 * torch.sigmoid(final_model.kalman_gamma_logit)).item()
    
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO QUANT KALMAN DIXON-COLES ===")
    logger.info(f" - Log-Loss (Train/REAL OOF): {oof_logloss:.4f}")
    logger.info(f" - Log-Loss (Test/OOS): {test_logloss:.4f}")
    
    logger.info(f" - Media xG Scored Predicha (Test): {scored_ts.mean():.3f} (Real: {df_test['goals_scored'].mean():.3f})")
    logger.info(f" - Media xG Conceded Predicha (Test): {conceded_ts.mean():.3f} (Real: {df_test['goals_conceded'].mean():.3f})")
    logger.info(f" - Parámetro Dixon-Coles Rho: {final_model.rho.item():.4f}")
    logger.info(f" - Kalman Drift Ratio Gamma: {learned_gamma:.6f}")
    logger.info(f" - Peso Form / Elo: {final_model.w_elo.item():.4f}, Peso Form / xG: {final_model.w_xg.item():.4f}")
    
    # Comparación con modelo Poisson anterior
    poisson_oof_path = os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet')
    if os.path.exists(poisson_oof_path):
        poisson_oof = pd.read_parquet(poisson_oof_path)
        poisson_probs = poisson_oof[['poisson_loss_prob', 'poisson_draw_prob', 'poisson_win_prob']].values
        poisson_y = df.iloc[:len(poisson_probs)]['outcome'].replace({-1: 0, 0: 1, 1: 2})
        poisson_logloss = log_loss(poisson_y, poisson_probs)
        logger.info(f" -> REFERENCIA: Modelo Poisson Antiguo Log-Loss (Train OOF): {poisson_logloss:.4f}")
        if oof_logloss < poisson_logloss:
            logger.info(" -> ¡ÉXITO! El nuevo Modelo Quant Kalman Dixon-Coles ha SUPERADO al Modelo Poisson Clásico en Log-Loss OOF.")
        else:
            logger.info(" -> El modelo Dixon-Coles se mantiene competitivo frente a Poisson.")
            
    # Guardar Resultados OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    res_train = pd.DataFrame({
        'predicted_xg_scored_quant': oof_scored,
        'predicted_xg_conceded_quant': oof_conceded,
        'quant_win_prob': oof_win,
        'quant_draw_prob': oof_draw,
        'quant_loss_prob': oof_loss
    }, index=df_train.index)
    
    res_test = pd.DataFrame({
        'predicted_xg_scored_quant': scored_ts,
        'predicted_xg_conceded_quant': conceded_ts,
        'quant_win_prob': win_ts,
        'quant_draw_prob': draw_ts,
        'quant_loss_prob': loss_ts
    }, index=df_test.index)
    
    res_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_quant_train.parquet'), engine='fastparquet')
    res_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_quant_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({
        'model_state': final_model.state_dict(), 
        'team_mapping': team_mapping,
        'n_teams': n_teams,
        'kalman_gamma': learned_gamma
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO QUANT KALMAN AVANZADO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_quant_advanced()
