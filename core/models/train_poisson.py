import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
from xgboost import XGBRegressor
from scipy.stats import poisson

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'poisson_model.pkl')
PROCESSED_DIR = '../data/processed'

def calc_dixon_coles_draw(lam_scored, lam_conceded, rho=-0.15):
    prob = 0
    for i in range(6):
        p_scored = poisson.pmf(i, lam_scored)
        p_conceded = poisson.pmf(i, lam_conceded)
        base_prob = p_scored * p_conceded
        
        tau = 1.0
        if i == 0: # 0-0
            tau = 1 - (lam_scored * lam_conceded * rho)
        elif i == 1: # 1-1
            tau = 1 - rho
            
        tau = max(0, tau)
        prob += base_prob * tau
    return prob

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
        
    def get_time_weights(dates, half_life_days=365):
        if dates is None:
            return None
        max_date = dates.max()
        days_diff = (max_date - dates).dt.days.clip(lower=0)
        return np.exp(-np.log(2) * days_diff / half_life_days)

    xgb_poisson = XGBRegressor(objective='count:poisson', n_estimators=100, learning_rate=0.05, random_state=42, device='cuda')
    
    def get_expanding_predictions(estimator, X, y, dates):
        tscv = get_cv_strategy(n_splits=5)
        preds = np.zeros(len(X))
        preds[:] = np.nan
        for train_idx, val_idx in tscv.split(X):
            w_tr = get_time_weights(dates.iloc[train_idx]) if dates is not None else None
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=w_tr)
            preds[val_idx] = estimator.predict(X.iloc[val_idx])
        
        first_train_idx = next(tscv.split(X))[0]
        w_first = get_time_weights(dates.iloc[first_train_idx]) if dates is not None else None
        estimator.fit(X.iloc[first_train_idx], y.iloc[first_train_idx], sample_weight=w_first)
        preds[first_train_idx] = estimator.predict(X.iloc[first_train_idx])
        return preds

    logger.info("Entrenando Modelo A (Poisson) OOF...")
    pred_scored_train = get_expanding_predictions(xgb_poisson, X_train, y_scored_train, train_dates)
    pred_conceded_train = get_expanding_predictions(xgb_poisson, X_train, y_conceded_train, train_dates)
    
    final_train_weights = get_time_weights(train_dates)
    xgb_poisson.fit(X_train, y_scored_train, sample_weight=final_train_weights)
    pred_scored_test = xgb_poisson.predict(X_test)
    xgb_poisson_scored_model = XGBRegressor(**xgb_poisson.get_params()).fit(X_train, y_scored_train, sample_weight=final_train_weights)
    
    xgb_poisson.fit(X_train, y_conceded_train, sample_weight=final_train_weights)
    pred_conceded_test = xgb_poisson.predict(X_test)
    xgb_poisson_conceded_model = XGBRegressor(**xgb_poisson.get_params()).fit(X_train, y_conceded_train, sample_weight=final_train_weights)
    
    draw_prob_train = np.vectorize(calc_dixon_coles_draw)(pred_scored_train, pred_conceded_train)
    draw_prob_test = np.vectorize(calc_dixon_coles_draw)(pred_scored_test, pred_conceded_test)
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame({
        'predicted_xg_scored': pred_scored_train,
        'predicted_xg_conceded': pred_conceded_train,
        'poisson_draw_prob': draw_prob_train
    }, index=X_train.index)
    
    oof_test = pd.DataFrame({
        'predicted_xg_scored': pred_scored_test,
        'predicted_xg_conceded': pred_conceded_test,
        'poisson_draw_prob': draw_prob_test
    }, index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({
        'model_scored': xgb_poisson_scored_model, 
        'model_conceded': xgb_poisson_conceded_model, 
        'features': feature_cols
    }, MODEL_SAVE_PATH)
    logger.info(f"Modelo Poisson guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_poisson()
