import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'context_model.pkl')
PROCESSED_DIR = '../data/processed'

def objective(trial, X_train, y_train, cv_strategy):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        xgb_eval = XGBClassifier(**param)
        xgb_eval.fit(X_tr, y_tr)
        
        y_prob = xgb_eval.predict_proba(X_val)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

def train_context():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Modelo B: Contexto y Táctica
    base_stats = [
        'shots_total', 'shots_on_target',
        'passes_total', 'passes_completed', 'pass_accuracy', 'possession_pct',
        'crosses', 'corners', 'through_balls', 'key_passes',
        'dribbles_completed', 'pressures', 'interceptions', 'clearances',
        'blocks', 'ball_recoveries', 'actions_under_pressure',
        'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
        'aerials_won'
    ]
    
    feature_cols = [
        'is_home', 'rest_days', 'rest_diff',
        'team_squad_value', 'opp_squad_value', 'squad_value_diff',
        'h2h_games_played', 'h2h_points_last_5', 'h2h_win_rate_hist', 'h2h_draw_rate_hist', 'is_european_hangover',
        'win_streak_3', 'loss_streak_3', 'xg_momentum_macd', 
        'opp_win_streak_3', 'opp_loss_streak_3', 'opp_xg_momentum_macd',
        'fatigue_index', 'fatigue_diff', 'xg_volatility_5', 'opp_xg_volatility_5', 'volatility_diff'
    ]
    
    for stat in base_stats:
        feature_cols.append(f"{stat}_ema3")
        feature_cols.append(f"{stat}_ema5")
        
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas en Contexto: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    logger.info("Optimizando Modelo B (Contexto) con Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, cv_strategy), n_trials=30)
    
    logger.info(f"Mejores parámetros XGBoost Contexto: {study.best_params}")
    
    xgb_best = XGBClassifier(
        **study.best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    
    logger.info("Calculando predicciones OOF para Train (Contexto)...")
    # Para Train OOF, usamos TimeSeriesSplit manualmente para evitar leakage, como en poisson
    # pero aquí podemos usar un loop sencillo.
    pred_probs_train = np.zeros((len(X_train), 3))
    
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        xgb_best.fit(X_tr, y_tr)
        pred_probs_train[val_idx] = xgb_best.predict_proba(X_val)
        
    # Rellenar la primera partición (que no se predice en OOF temporal) con una evaluación en sí misma (subóptimo pero evita NaNs)
    first_train_idx = next(cv_strategy.split(X_train))[0]
    xgb_best.fit(X_train.iloc[first_train_idx], y_train.iloc[first_train_idx])
    pred_probs_train[first_train_idx] = xgb_best.predict_proba(X_train.iloc[first_train_idx])
    
    logger.info("Entrenando Modelo B final y prediciendo Test...")
    xgb_best.fit(X_train, y_train)
    pred_probs_test = xgb_best.predict_proba(X_test)
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': xgb_best, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"Modelo Contexto guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_context()
