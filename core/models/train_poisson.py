import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_poisson')


MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'poisson_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def calc_match_probabilities(lam_scored, lam_conceded, rho=-0.15, max_goals=10):
    """
    Calculates Win, Draw, and Loss probabilities using a bivariate Poisson distribution 
    with Dixon-Coles adjustment for low-scoring matches.
    """
    # Matrices of goals
    x = np.arange(max_goals + 1)
    y = np.arange(max_goals + 1)
    
    # PMF for scored and conceded
    pmf_scored = poisson.pmf(x, lam_scored)
    pmf_conceded = poisson.pmf(y, lam_conceded)
    
    # Outer product to get the independent bivariate Poisson matrix
    prob_matrix = np.outer(pmf_scored, pmf_conceded)
    
    # Apply Dixon-Coles adjustment
    # tau(x, y) = 
    # 0,0 -> 1 - lambda*mu*rho
    # 0,1 -> 1 + lambda*rho
    # 1,0 -> 1 + mu*rho
    # 1,1 -> 1 - rho
    
    # Calculate terms for adjustment
    tau_00 = max(0, 1 - (lam_scored * lam_conceded * rho))
    tau_01 = max(0, 1 + (lam_scored * rho))
    tau_10 = max(0, 1 + (lam_conceded * rho))
    tau_11 = max(0, 1 - rho)
    
    # Apply to matrix (x is scored (rows), y is conceded (cols))
    prob_matrix[0, 0] *= tau_00
    prob_matrix[0, 1] *= tau_01
    prob_matrix[1, 0] *= tau_10
    prob_matrix[1, 1] *= tau_11
    
    # Win = Lower triangle (scored > conceded)
    # Draw = Diagonal (scored == conceded)
    # Loss = Upper triangle (scored < conceded)
    win_prob = np.tril(prob_matrix, -1).sum()
    draw_prob = np.trace(prob_matrix)
    loss_prob = np.triu(prob_matrix, 1).sum()
    
    # Normalize in case adjustments pushed sum slightly away from 1.0
    total = win_prob + draw_prob + loss_prob
    if total > 0:
        return win_prob / total, draw_prob / total, loss_prob / total
    else:
        return 0.0, 0.0, 0.0

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def get_expanding_predictions(estimator_factory, X, y, dates):
    """
    Generates Out-Of-Fold predictions using Time Series Split.
    For the very first fold, uses a standard KFold internally to prevent data leakage.
    """
    tscv = get_cv_strategy(n_splits=5)
    preds = np.zeros(len(X))
    preds[:] = np.nan
    
    splits = list(tscv.split(X))
    
    # First chunk (Fold 0 training data)
    first_train_idx = splits[0][0]
    X_first = X.iloc[first_train_idx]
    y_first = y.iloc[first_train_idx]
    dates_first = dates.iloc[first_train_idx] if dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con TimeSeriesSplit(5) para evitar Leakage...")
    kf = TimeSeriesSplit(n_splits=5)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train) if dates_kf_train is not None else None
        
        # Instantiate a fresh model for this internal split
        kf_estimator = estimator_factory()
        kf_estimator.fit(X_kf_train, y_kf_train, sample_weight=w_tr)
        
        val_indices_in_original = first_train_idx[kf_val]
        preds[val_indices_in_original] = kf_estimator.predict(X_kf_val)

    # Subsequent expanding window folds
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        w_tr = get_time_weights(dates.iloc[train_idx]) if dates is not None else None
        
        fold_estimator = estimator_factory()
        fold_estimator.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=w_tr)
        preds[val_idx] = fold_estimator.predict(X.iloc[val_idx])
        
    return preds

def get_xgb_model():
    return XGBRegressor(
        objective='count:poisson',
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        device='cuda'
    )

def train_poisson():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Modelo A: Poisson Regression (Goles, Fuerza, ELO)
    feature_cols = [
        'relative_attack_strength', 
        'team_att_rating', 'team_def_rating', 'opp_att_rating', 'opp_def_rating',
        'team_elo', 'opp_elo', 'elo_diff',
        'xg_created_ema3', 'xg_conceded_ema3',
        'xg_created_ema5', 'xg_conceded_ema5'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas en Poisson: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    y_scored = df['goals_scored'].fillna(0)
    y_conceded = df['goals_conceded'].fillna(0)
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_scored_train, y_conceded_train = y_scored.iloc[:split_idx], y_conceded.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
        
    logger.info("=== ENTRENANDO MODELO A (POISSON) OOF ===")
    
    logger.info("Entrenando objetivo: Goles Anotados (Scored)...")
    pred_scored_train = get_expanding_predictions(get_xgb_model, X_train, y_scored_train, train_dates)
    
    logger.info("Entrenando objetivo: Goles Recibidos (Conceded)...")
    pred_conceded_train = get_expanding_predictions(get_xgb_model, X_train, y_conceded_train, train_dates)
    
    logger.info("Entrenando modelos finales (Scored y Conceded) sobre todo el Train Set...")
    final_train_weights = get_time_weights(train_dates)
    
    xgb_poisson_scored_model = get_xgb_model()
    xgb_poisson_scored_model.fit(X_train, y_scored_train, sample_weight=final_train_weights)
    pred_scored_test = xgb_poisson_scored_model.predict(X_test)
    
    xgb_poisson_conceded_model = get_xgb_model()
    xgb_poisson_conceded_model.fit(X_train, y_conceded_train, sample_weight=final_train_weights)
    pred_conceded_test = xgb_poisson_conceded_model.predict(X_test)
    
    logger.info("=== CALCULANDO PROBABILIDADES MATEMÁTICAS (DIXON-COLES) ===")
    
    # Vectorizar la función para aplicarla rápido a los arreglos
    v_calc_probs = np.vectorize(calc_match_probabilities)
    
    # Train Probs
    win_prob_train, draw_prob_train, loss_prob_train = v_calc_probs(pred_scored_train, pred_conceded_train)
    # Test Probs
    win_prob_test, draw_prob_test, loss_prob_test = v_calc_probs(pred_scored_test, pred_conceded_test)
    
    # LOGS: Verificacion de los resultados de Poisson
    logger.info("Estadísticas del Entrenamiento:")
    logger.info(f" - Media xG Scored predicho: {pred_scored_train.mean():.3f} (Real: {y_scored_train.mean():.3f})")
    logger.info(f" - Media xG Conceded predicho: {pred_conceded_train.mean():.3f} (Real: {y_conceded_train.mean():.3f})")
    logger.info(f" - Promedio Prob Victoria: {win_prob_train.mean()*100:.1f}%")
    logger.info(f" - Promedio Prob Empate: {draw_prob_train.mean()*100:.1f}%")
    logger.info(f" - Promedio Prob Derrota: {loss_prob_train.mean()*100:.1f}%")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame({
        'predicted_xg_scored': pred_scored_train,
        'predicted_xg_conceded': pred_conceded_train,
        'poisson_win_prob': win_prob_train,
        'poisson_draw_prob': draw_prob_train,
        'poisson_loss_prob': loss_prob_train
    }, index=X_train.index)
    
    oof_test = pd.DataFrame({
        'predicted_xg_scored': pred_scored_test,
        'predicted_xg_conceded': pred_conceded_test,
        'poisson_win_prob': win_prob_test,
        'poisson_draw_prob': draw_prob_test,
        'poisson_loss_prob': loss_prob_test
    }, index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_test.parquet'), engine='fastparquet')
    
    logger.info(f"Archivos OOF guardados exitosamente. (Rows Train: {len(oof_train)}, Rows Test: {len(oof_test)})")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({
        'model_scored': xgb_poisson_scored_model, 
        'model_conceded': xgb_poisson_conceded_model, 
        'features': feature_cols
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO POISSON FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_poisson()
