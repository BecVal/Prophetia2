import os
import sys
import pandas as pd
import numpy as np
import joblib
import pymc as pm
import pytensor
import pytensor.tensor as pt
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import nbinom
import arviz as az

# Permitimos backend flexible, recomendando 'c' o default en lugar de forzar NUMBA que puede dar problemas.
pytensor.config.mode = 'FAST_RUN'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_quant_advanced')

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'quant_advanced_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def calc_zinb_dc_probabilities_vectorized(mu_home, mu_away, alpha_h, alpha_a, psi, rho, max_goals=10):
    """
    Calcula las probabilidades de Victoria, Empate y Derrota usando Zero-Inflated Negative Binomial (ZINB)
    con ajuste Dixon-Coles aprendido.
    """
    mu_h = np.asarray(mu_home).reshape(-1, 1, 1)
    mu_a = np.asarray(mu_away).reshape(-1, 1, 1)
    
    # Parametrización para scipy.stats.nbinom: n = alpha, p = alpha / (alpha + mu)
    p_h = alpha_h / (alpha_h + mu_h)
    p_a = alpha_a / (alpha_a + mu_a)
    
    x = np.arange(max_goals + 1).reshape(1, max_goals + 1, 1)
    y = np.arange(max_goals + 1).reshape(1, 1, max_goals + 1)
    
    # PMF Base Negative Binomial
    nb_pmf_home = nbinom.pmf(x, alpha_h, p_h)
    nb_pmf_away = nbinom.pmf(y, alpha_a, p_a)
    
    # Ajuste Zero-Inflation
    pmf_home = psi * nb_pmf_home
    pmf_home[:, 0, :] += (1 - psi)  # Solo sumamos (1-psi) a x=0
    
    pmf_away = psi * nb_pmf_away
    pmf_away[:, :, 0] += (1 - psi)  # Solo sumamos (1-psi) a y=0
    
    # Probabilidad Conjunta Independiente ZINB
    prob_matrix = pmf_home * pmf_away
    
    # Ajuste Dixon-Coles
    if rho != 0.0:
        mh = mu_h.reshape(-1)
        ma = mu_a.reshape(-1)
        
        tau_00 = np.clip(1 - rho * mh * ma, 0, None)
        tau_10 = np.clip(1 + rho * ma, 0, None)
        tau_01 = np.clip(1 + rho * mh, 0, None)
        tau_11 = np.clip(1 - rho, 0, None)
        
        prob_matrix[:, 0, 0] *= tau_00
        prob_matrix[:, 1, 0] *= tau_10
        prob_matrix[:, 0, 1] *= tau_01
        prob_matrix[:, 1, 1] *= tau_11
        
        sums = prob_matrix.sum(axis=(1, 2), keepdims=True)
        prob_matrix /= np.where(sums > 0, sums, 1.0)
        
    draw_prob = np.diagonal(prob_matrix, axis1=1, axis2=2).sum(axis=1)
    win_prob = np.tril(prob_matrix, -1).sum(axis=(1, 2))
    loss_prob = np.triu(prob_matrix, 1).sum(axis=(1, 2))
    
    return win_prob, draw_prob, loss_prob

def build_and_fit_pymc_model(df_train):
    teams = pd.concat([df_train['team'], df_train['opponent']]).unique()
    team_mapping = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    home_teams = df_train['team'].map(team_mapping).values
    away_teams = df_train['opponent'].map(team_mapping).values
    
    home_goals = df_train['goals_scored'].values
    away_goals = df_train['goals_conceded'].values
    
    is_home = df_train.get('is_home', np.ones(len(df_train))).values
    
    # Decaimiento del tiempo (Manteniendo compatibilidad y eficiencia para MAP)
    if 'match_date' in df_train.columns:
        dates = pd.to_datetime(df_train['match_date'])
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        half_life = 600.0
        decay_rate = np.log(2) / half_life
        weights = np.exp(-decay_rate * days_ago)
        weights = weights / np.mean(weights) 
    else:
        weights = np.ones(len(df_train))
        
    with pm.Model() as quant_model:
        home_idx = pm.Data('home_idx', home_teams)
        away_idx = pm.Data('away_idx', away_teams)
        w_obs = pm.Data('w_obs', weights)
        h_obs = pm.Data('h_obs', home_goals)
        a_obs = pm.Data('a_obs', away_goals)
        is_h = pm.Data('is_h', is_home)
        
        # --- PARÁMETROS BASE ---
        home_advantage = pm.Normal('home_advantage', mu=0.2, sigma=0.1)
        intercept = pm.Normal('intercept', mu=np.log(1.5), sigma=0.5)
        
        att_star = pm.Normal('att_star', mu=0, sigma=0.5, shape=n_teams)
        def_star = pm.Normal('def_star', mu=0, sigma=0.5, shape=n_teams)
        
        att = pm.Deterministic('att', att_star - pt.mean(att_star))
        def_ = pm.Deterministic('def', def_star - pt.mean(def_star))
        
        # --- PARÁMETROS ZINB ---
        # Dispersión (Alpha) de la Binomial Negativa. Valores altos -> Comportamiento Poisson.
        alpha_h = pm.HalfNormal('alpha_h', sigma=5.0)
        alpha_a = pm.HalfNormal('alpha_a', sigma=5.0)
        
        # Zero-Inflation (Psi) - Probabilidad global de 0 estructural
        # En PyMC, psi es la probabilidad de la distribución BASE (Poisson/NB), no la prob. de cero.
        # Por tanto, priorizamos probabilidades altas (ej. 88% por defecto)
        psi_logit = pm.Normal('psi_logit', mu=2.0, sigma=1.0)
        psi = pm.Deterministic('psi', pm.math.sigmoid(psi_logit))
        
        # Parámetro Dixon-Coles (Rho) real, aprendido de los datos
        rho = pm.Normal('rho', mu=0.0, sigma=0.1)
        
        # --- EXPECTED GOALS (MU) ---
        log_theta_home = intercept + (home_advantage * is_h) + att[home_idx] + def_[away_idx]
        log_theta_away = intercept + att[away_idx] + def_[home_idx]
        
        mu_home = pm.math.exp(log_theta_home)
        mu_away = pm.math.exp(log_theta_away)
        
        # --- LIKELIHOOD (ZINB Marginals) ---
        home_dist = pm.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu_home, alpha=alpha_h)
        away_dist = pm.ZeroInflatedNegativeBinomial.dist(psi=psi, mu=mu_away, alpha=alpha_a)
        
        logp_h = pm.logp(home_dist, h_obs)
        logp_a = pm.logp(away_dist, a_obs)
        
        # Ajuste por Decaimiento Exponencial 
        pm.Potential('weighted_home_logp', logp_h * w_obs)
        pm.Potential('weighted_away_logp', logp_a * w_obs)
        
        # --- LIKELIHOOD (Dixon-Coles Joint Correction) ---
        tau_obs = pt.switch(pt.eq(h_obs, 0) & pt.eq(a_obs, 0), 1 - rho * mu_home * mu_away,
                  pt.switch(pt.eq(h_obs, 1) & pt.eq(a_obs, 0), 1 + rho * mu_away,
                  pt.switch(pt.eq(h_obs, 0) & pt.eq(a_obs, 1), 1 + rho * mu_home,
                  pt.switch(pt.eq(h_obs, 1) & pt.eq(a_obs, 1), 1 - rho, 1.0))))
        
        # Clip para asegurar positividad absoluta de la verosimilitud en PyMC
        tau_obs_safe = pt.maximum(tau_obs, 1e-5)
        pm.Potential('dc_correction', pt.log(tau_obs_safe) * w_obs)
                                        
        logger.info(f"Iniciando Optimización MAP (ZINB + Dixon-Coles + Time Decay) para {len(home_goals)} partidos...")
        
        try:
            map_estimate = pm.find_MAP(method='L-BFGS-B', progressbar=True)
        except Exception as e:
            logger.warning(f"find_MAP falló ({e}). Intentando optimización de respaldo (Powell)...")
            try:
                map_estimate = pm.find_MAP(method='Powell', progressbar=True)
            except Exception as e2:
                logger.warning(f"Powell falló ({e2}). Intentando Inferencia Variacional (ADVI)...")
                with quant_model:
                    mean_field = pm.fit(n=30000, method='advi', progressbar=True)
                    trace = mean_field.sample(200)
                    map_estimate = {var: np.asarray(trace.posterior[var].mean(dim=["chain", "draw"])) for var in trace.posterior.data_vars}
        
    return map_estimate, team_mapping, quant_model

def predict_from_map(map_estimate, team_mapping, df_test):
    def safe_map(team):
        return team_mapping.get(team, -1)
        
    home_teams = df_test['team'].map(safe_map).values
    away_teams = df_test['opponent'].map(safe_map).values
    is_home = df_test.get('is_home', np.ones(len(df_test))).values
    
    intercept = float(map_estimate['intercept'])
    home_adv = float(map_estimate['home_advantage'])
    
    # Extraer parámetros ZINB y Dixon-Coles
    alpha_h = float(map_estimate.get('alpha_h', 10.0))
    alpha_a = float(map_estimate.get('alpha_a', 10.0))
    psi = float(map_estimate.get('psi', 0.0))
    rho = float(map_estimate.get('rho', 0.0))
    
    att_post = np.array(map_estimate['att'])
    def_post = np.array(map_estimate['def'])
    
    att_ext = np.append(att_post, 0.0)
    def_ext = np.append(def_post, 0.0)
    
    # Reemplazamos -1 por el índice del dummy de fuerza 0.0
    idx_unseen_att = len(att_post)
    h_idx = np.where(home_teams == -1, idx_unseen_att, home_teams)
    a_idx = np.where(away_teams == -1, idx_unseen_att, away_teams)
    
    h_att, h_def = att_ext[h_idx], def_ext[h_idx]
    a_att, a_def = att_ext[a_idx], def_ext[a_idx]
    
    # Cálculos vectorizados para las medias (Expected Goals latentes)
    mu_h = np.exp(intercept + (home_adv * is_home) + h_att + a_def)
    mu_a = np.exp(intercept + a_att + h_def)
    
    # Expected Goals Reales (Ajustado por Zero-Inflation)
    # En PyMC, psi es la probabilidad de la distribución base (Negative Binomial)
    # por lo que la media teórica empírica es psi * mu
    pred_scored = psi * mu_h
    pred_conceded = psi * mu_a
    
    # Probabilidades de victoria, empate y derrota usando ZINB + DC
    w, d, l = calc_zinb_dc_probabilities_vectorized(mu_h, mu_a, alpha_h, alpha_a, psi, rho)
        
    return w, d, l, pred_scored, pred_conceded

def train_quant_advanced():
    df = get_base_dataset()
    
    split_idx = get_train_test_split(df)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    logger.info("=== ENTRENANDO MODELO QUANT AVANZADO (PyMC: ZINB + Dixon-Coles Aprendido + Time Decay) ===")
    
    map_estimate, team_mapping, quant_model = build_and_fit_pymc_model(df_train)
    
    logger.info("Calculando probabilidades vectorizadas en Train (In-Sample)...")
    win_tr, draw_tr, loss_tr, scored_tr, conceded_tr = predict_from_map(map_estimate, team_mapping, df_train)
    
    logger.info("Calculando probabilidades vectorizadas en Test (Out-Of-Sample)...")
    win_ts, draw_ts, loss_ts, scored_ts, conceded_ts = predict_from_map(map_estimate, team_mapping, df_test)
    
    # === AUDITORÍA LOG-LOSS Y CALIBRACIÓN ===
    y_train = df_train['outcome'].replace({-1: 0, 0: 1, 1: 2})
    pred_probs_train = np.column_stack((loss_tr, draw_tr, win_tr))
    
    y_test = df_test['outcome'].replace({-1: 0, 0: 1, 1: 2})
    pred_probs_test = np.column_stack((loss_ts, draw_ts, win_ts))
    
    try:
        oof_logloss = log_loss(y_train, pred_probs_train)
        test_logloss = log_loss(y_test, pred_probs_test)
    except Exception as e:
        logger.error(f"Error al calcular log-loss: {e}")
        oof_logloss, test_logloss = np.nan, np.nan
        
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO QUANT ZINB ===")
    logger.info(f" - Log-Loss (Train/In-Sample): {oof_logloss:.4f}")
    logger.info(f" - Log-Loss (Test/OOS): {test_logloss:.4f}")
    
    logger.info(f" - Media xG Scored Predicha (Test): {scored_ts.mean():.3f} (Real: {df_test['goals_scored'].mean():.3f})")
    logger.info(f" - Media xG Conceded Predicha (Test): {conceded_ts.mean():.3f} (Real: {df_test['goals_conceded'].mean():.3f})")
    
    # Comparación
    poisson_oof_path = os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet')
    if os.path.exists(poisson_oof_path):
        poisson_oof = pd.read_parquet(poisson_oof_path)
        poisson_probs = poisson_oof[['poisson_loss_prob', 'poisson_draw_prob', 'poisson_win_prob']].values
        poisson_y = df.iloc[:len(poisson_probs)]['outcome'].replace({-1: 0, 0: 1, 1: 2})
        poisson_logloss = log_loss(poisson_y, poisson_probs)
        logger.info(f" -> REFERENCIA: Modelo Poisson Antiguo Log-Loss (Train OOF): {poisson_logloss:.4f}")
        if oof_logloss < poisson_logloss:
            logger.info(" -> ¡ÉXITO! El nuevo Modelo Quant ZINB ha SUPERADO al Modelo Poisson Clásico en Log-Loss.")
        else:
            logger.info(" -> El modelo PyMC in-sample tiene un Log-Loss marginalmente mayor. Se requiere afinar priors o la regresión ZINB.")
            
    # Guardar Resultados
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    res_train = pd.DataFrame({
        'predicted_xg_scored_quant': scored_tr,
        'predicted_xg_conceded_quant': conceded_tr,
        'quant_win_prob': win_tr,
        'quant_draw_prob': draw_tr,
        'quant_loss_prob': loss_tr
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
        'map_estimate': map_estimate, 
        'team_mapping': team_mapping,
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO QUANT AVANZADO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_quant_advanced()

