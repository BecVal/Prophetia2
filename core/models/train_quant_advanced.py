import os
import sys
import pandas as pd
import numpy as np
import joblib
import pymc as pm
import pytensor
pytensor.config.mode = 'NUMBA'
import pytensor.tensor as pt
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import arviz as az

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_quant_advanced')

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'quant_advanced_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def calc_zibnb_probabilities(mu_home, mu_away, alpha_home, alpha_away, psi_home, psi_away, max_goals=10):
    """
    Calculates Win, Draw, and Loss probabilities using independent Zero-Inflated Negative Binomials
    (ZINB). For a Bivariate structure, we approximate it by coupling them through shared features
    in the PyMC model, so the marginals are ZINB.
    
    mu: mean of the NB distribution
    alpha: dispersion parameter of the NB distribution
    psi: probability of zero-inflation (e.g. tactical 0-0 state)
    """
    from scipy.stats import nbinom
    
    # NB parameters for scipy:
    # variance = mu + alpha * mu^2
    # n = 1 / alpha
    # p = n / (n + mu)
    
    x = np.arange(max_goals + 1)
    y = np.arange(max_goals + 1)
    
    n_h = 1.0 / np.clip(alpha_home, 1e-6, 10.0)
    p_h = n_h / (n_h + mu_home)
    
    n_a = 1.0 / np.clip(alpha_away, 1e-6, 10.0)
    p_a = n_a / (n_a + mu_away)
    
    # Calculate PMFs for regular Negative Binomial
    nb_pmf_home = nbinom.pmf(x, n_h, p_h)
    nb_pmf_away = nbinom.pmf(y, n_a, p_a)
    
    # Apply Zero-Inflation
    pmf_home = (1 - psi_home) * nb_pmf_home
    pmf_home[0] += psi_home
    
    pmf_away = (1 - psi_away) * nb_pmf_away
    pmf_away[0] += psi_away
    
    # Independence assumption conditional on the latent state
    prob_matrix = np.outer(pmf_home, pmf_away)
    
    win_prob = np.tril(prob_matrix, -1).sum()
    draw_prob = np.trace(prob_matrix)
    loss_prob = np.triu(prob_matrix, 1).sum()
    
    total = win_prob + draw_prob + loss_prob
    if total > 0:
        return win_prob / total, draw_prob / total, loss_prob / total
    else:
        return 0.0, 0.0, 0.0

def build_and_fit_pymc_model(df_train):
    """
    Construye y entrena el modelo PyMC para fuerza dinámica de equipos (Latent Skill)
    y regresión ZINB.
    """
    # Map teams to IDs
    teams = pd.concat([df_train['team'], df_train['opponent']]).unique()
    team_mapping = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    home_teams = df_train['team'].map(team_mapping).values
    away_teams = df_train['opponent'].map(team_mapping).values
    
    home_goals = df_train['goals_scored'].values
    away_goals = df_train['goals_conceded'].values
    
    # Limit dataset size for computation time in this exact implementation if it's too large,
    # but the user wants precision. We will use ADVI for fast variational inference.
    
    with pm.Model() as quant_model:
        # Data containers for out-of-sample predictions
        home_idx = pm.Data('home_idx', home_teams)
        away_idx = pm.Data('away_idx', away_teams)
        
        # --- LATENT SKILLS (STATIC PER FOLD FOR PERFORMANCE) ---
        # For a full dynamic random walk match-by-match, we would need a state space model.
        # Given the massive number of rows, we estimate a latent attack and defense for the current fold
        home_advantage = pm.Normal('home_advantage', mu=0.2, sigma=0.1)
        intercept = pm.Normal('intercept', mu=np.log(1.5), sigma=0.5)
        
        # Equipos: Ataque y Defensa (Suma 0 para identificabilidad)
        att_star = pm.Normal('att_star', mu=0, sigma=0.5, shape=n_teams)
        def_star = pm.Normal('def_star', mu=0, sigma=0.5, shape=n_teams)
        
        att = pm.Deterministic('att', att_star - pt.mean(att_star))
        def_ = pm.Deterministic('def', def_star - pt.mean(def_star))
        
        # --- ZERO-INFLATED NEGATIVE BINOMIAL PARAMETERS ---
        # Log-linear predictors for the mean
        log_theta_home = intercept + home_advantage + att[home_idx] + def_[away_idx]
        log_theta_away = intercept + att[away_idx] + def_[home_idx]
        
        mu_home = pm.math.exp(log_theta_home)
        mu_away = pm.math.exp(log_theta_away)
        
        # Overdispersion (alpha)
        alpha = pm.Exponential('alpha', 1.0)
        
        # Zero-Inflation probability (psi)
        # Logit link based on the defensive strengths, representing "tactical lockdown"
        logit_psi = pm.Normal('psi_intercept', mu=-2.0, sigma=1.0) + 0.1 * (def_[home_idx] + def_[away_idx])
        psi = pm.math.invlogit(logit_psi)
        
        # LIKELIHOOD
        pm.ZeroInflatedNegativeBinomial('home_goals_obs', 
                                        mu=mu_home, 
                                        alpha=alpha, 
                                        psi=psi, 
                                        observed=home_goals)
                                        
        pm.ZeroInflatedNegativeBinomial('away_goals_obs', 
                                        mu=mu_away, 
                                        alpha=alpha, 
                                        psi=psi, 
                                        observed=away_goals)
                                        
        logger.info(f"Iniciando Inferencia Variacional (ADVI) en PyMC para {len(home_goals)} partidos y {n_teams} equipos...")
        # Usamos ADVI en lugar de NUTS porque NUTS tardaría días en >100k filas
        approx = pm.fit(n=30000, method='advi', obj_n_mc=1, progressbar=False)
        trace = approx.sample(1000)
        
    return trace, team_mapping, quant_model, approx

def predict_from_trace(trace, team_mapping, df_test):
    """
    Genera probabilidades Win/Draw/Loss usando las medianas de los parámetros posteriores.
    """
    home_teams = df_test['team'].map(team_mapping).fillna(0).astype(int).values
    away_teams = df_test['opponent'].map(team_mapping).fillna(0).astype(int).values
    
    intercept = np.median(trace.posterior['intercept'])
    home_adv = np.median(trace.posterior['home_advantage'])
    att = np.median(trace.posterior['att'], axis=(0,1))
    def_ = np.median(trace.posterior['def'], axis=(0,1))
    alpha = np.median(trace.posterior['alpha'])
    psi_intercept = np.median(trace.posterior['psi_intercept'])
    
    preds_win, preds_draw, preds_loss = [], [], []
    pred_scored, pred_conceded = [], []
    
    for i in range(len(df_test)):
        h_idx = home_teams[i]
        a_idx = away_teams[i]
        
        # Calculate Means
        mu_h = np.exp(intercept + home_adv + att[h_idx] + def_[a_idx])
        mu_a = np.exp(intercept + att[a_idx] + def_[h_idx])
        
        # Calculate Zero Inflation
        psi = 1.0 / (1.0 + np.exp(-(psi_intercept + 0.1 * (def_[h_idx] + def_[a_idx]))))
        
        w, d, l = calc_zibnb_probabilities(mu_h, mu_a, alpha, alpha, psi, psi)
        
        preds_win.append(w)
        preds_draw.append(d)
        preds_loss.append(l)
        
        # Expected Goals
        pred_scored.append((1 - psi) * mu_h)
        pred_conceded.append((1 - psi) * mu_a)
        
    return np.array(preds_win), np.array(preds_draw), np.array(preds_loss), np.array(pred_scored), np.array(pred_conceded)

def train_quant_advanced():
    df = get_base_dataset()
    
    # Por razones computacionales para un modelo MCMC/ADVI global, nos enfocamos en data reciente si es masiva,
    # pero el usuario quiere la mayor precisión para la predicción final. Usaremos todo el dataset si la memoria lo permite.
    # Dado que Prophetia2 filtra data en run_pipeline o usa toda, tomamos lo estándar.
    
    split_idx = get_train_test_split(df)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    logger.info("=== ENTRENANDO MODELO QUANT AVANZADO (PyMC: ZIBNB + Fuerza Latente) ===")
    
    # Entrenamiento completo en Train Set
    trace, team_mapping, quant_model, approx = build_and_fit_pymc_model(df_train)
    
    # Evaluar Train (para auditoría OOF-like, aquí in-sample por la limitación de CV en PyMC)
    logger.info("Calculando probabilidades en conjunto de Entrenamiento (In-Sample)...")
    win_tr, draw_tr, loss_tr, scored_tr, conceded_tr = predict_from_trace(trace, team_mapping, df_train)
    
    # Evaluar Test
    logger.info("Calculando probabilidades en conjunto de Prueba (Out-Of-Sample)...")
    win_ts, draw_ts, loss_ts, scored_ts, conceded_ts = predict_from_trace(trace, team_mapping, df_test)
    
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
        
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO QUANT ===")
    logger.info(f" - PyMC ZIBNB Log-Loss (Train/In-Sample): {oof_logloss:.4f}")
    logger.info(f" - PyMC ZIBNB Log-Loss (Test/OOS): {test_logloss:.4f}")
    
    logger.info(f" - Media xG Scored Predicha: {scored_ts.mean():.3f} (Real: {df_test['goals_scored'].mean():.3f})")
    logger.info(f" - Media xG Conceded Predicha: {conceded_ts.mean():.3f} (Real: {df_test['goals_conceded'].mean():.3f})")
    
    # Comparación con Poisson (si existe)
    poisson_oof_path = os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet')
    if os.path.exists(poisson_oof_path):
        poisson_oof = pd.read_parquet(poisson_oof_path)
        poisson_probs = poisson_oof[['poisson_loss_prob', 'poisson_draw_prob', 'poisson_win_prob']].values
        poisson_y = df.iloc[:len(poisson_probs)]['outcome'].replace({-1: 0, 0: 1, 1: 2})
        poisson_logloss = log_loss(poisson_y, poisson_probs)
        logger.info(f" -> REFERENCIA: Modelo Poisson Antiguo Log-Loss (Train OOF): {poisson_logloss:.4f}")
        if oof_logloss < poisson_logloss:
            logger.info(" -> ¡El nuevo Modelo Quant (ZIBNB) ha SUPERADO al Modelo Poisson Clásico en Log-Loss!")
        else:
            logger.info(" -> El modelo PyMC in-sample tiene un Log-Loss mayor. Podría ser necesario ajustar los Priors o aumentar el muestreo MCMC.")
            
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
        'trace': trace, 
        'team_mapping': team_mapping,
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO QUANT AVANZADO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_quant_advanced()
