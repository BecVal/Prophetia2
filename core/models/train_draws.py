import os

import json

RUN_OPTUNA = False
OPTUNA_TRIALS = 20
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger


OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_draws.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_draws')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'draws_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr)
        
        xgb_eval = XGBClassifier(**param)
        xgb_eval.fit(X_tr, y_tr, sample_weight=w_tr)
        
        y_prob = xgb_eval.predict_proba(X_val)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

def train_draws():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Features específicos para cazar empates: 
    # Diferencias pequeñas de valor, rachas parecidas, volatilidad baja/alta.
    feature_cols = [
        'is_home', 'rest_days', 'rest_diff',
        'team_squad_value', 'opp_squad_value', 'squad_value_diff',
        'h2h_games_played', 'h2h_draw_rate_hist',
        'win_streak_3', 'loss_streak_3', 'xg_momentum_macd', 
        'opp_win_streak_3', 'opp_loss_streak_3', 'opp_xg_momentum_macd',
        'fatigue_index', 'fatigue_diff', 'xg_volatility_5', 'opp_xg_volatility_5', 'volatility_diff'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas en Draws: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    
    # TARGET BINARIO: 1 si es Empate, 0 si no.
    # Original target mapping: -1: 0 (Loss), 0: 1 (Draw), 1: 2 (Win)
    y_multi = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y = (y_multi == 1).astype(int)
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo Binario Caza-Empates (Draws) con Optuna ({OPTUNA_TRIALS} Trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, train_dates, cv_strategy), n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Mejores parámetros guardados en {OPTUNA_PARAMS_FILE}")
    else:
        logger.info("Cargando mejores parámetros de Optuna guardados...")
        if os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
        else:
            logger.warning(f"Archivo de parámetros {OPTUNA_PARAMS_FILE} no encontrado. Ejecutando Optuna como fallback.")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, train_dates, cv_strategy), n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params, f, indent=4)
                
    logger.info(f"Mejores parámetros XGBoost Draws: {best_params}")
    
    xgb_best = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        random_state=42,
        device='cuda'
    )
    
    logger.info("Calculando predicciones OOF para Train (Draws)...")
    # Es binario, predict_proba devuelve [prob_not_draw, prob_draw]
    pred_probs_train = np.zeros((len(X_train), 2))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando KFold
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para OOF completo...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train)
        
        kf_estimator_base = XGBClassifier(**xgb_best.get_params())
        kf_estimator = CalibratedClassifierCV(estimator=kf_estimator_base, method='isotonic', cv=3)
        try:
            kf_estimator.fit(X_kf_train, y_kf_train, sample_weight=w_tr) if w_tr is not None else kf_estimator.fit(X_kf_train, y_kf_train)
        except TypeError:
            kf_estimator.fit(X_kf_train, y_kf_train)
        
        val_indices_in_original = first_train_idx[kf_val]
        pred_probs_train[val_indices_in_original] = kf_estimator.predict_proba(X_kf_val)

    # 2. Expanding Windows estándar para el resto
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        fold_estimator_base = XGBClassifier(**xgb_best.get_params())
        fold_estimator = CalibratedClassifierCV(estimator=fold_estimator_base, method='isotonic', cv=3)
        try:
            fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr) if w_tr is not None else fold_estimator.fit(X_tr, y_tr)
        except TypeError:
            fold_estimator.fit(X_tr, y_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo Draws final (Isotonic Calibration) y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    final_model = CalibratedClassifierCV(estimator=xgb_best, method='isotonic', cv=get_cv_strategy(n_splits=5))
    try:
        final_model.fit(X_train, y_train, sample_weight=final_w_tr) if final_w_tr is not None else final_model.fit(X_train, y_train)
    except TypeError:
        final_model.fit(X_train, y_train)
    pred_probs_test = final_model.predict_proba(X_test)
    
    # Nos interesa solo la probabilidad de la clase positiva (Empate)
    # Extraemos la columna 1 (prob_draw)
    prob_draw_train = pred_probs_train[:, 1]
    prob_draw_test = pred_probs_test[:, 1]
    
    # LOGS: Verificacion
    real_draw = y_train.mean()
    pred_draw = prob_draw_train.mean()
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO DRAWS ===")
    logger.info(f" - Empate (Draw) | Predicha: {pred_draw*100:.1f}% | Real en Dataset: {real_draw*100:.1f}%")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame({'prob_is_draw': prob_draw_train}, index=X_train.index)
    oof_test = pd.DataFrame({'prob_is_draw': prob_draw_test}, index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_draws_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_draws_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': final_model, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO DRAWS FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_draws()
