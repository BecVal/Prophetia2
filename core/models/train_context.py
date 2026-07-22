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


OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_context.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_context')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'context_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'enable_categorical': True
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr)
        
        xgb_eval = XGBClassifier(**param)
        calibrated_eval = CalibratedClassifierCV(estimator=xgb_eval, method='isotonic', cv=3)
        if w_tr is not None:
            calibrated_eval.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            calibrated_eval.fit(X_tr, y_tr)
        
        y_prob = calibrated_eval.predict_proba(X_val)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

def train_context():
    df = get_base_dataset()
    
    if 'competition' in df.columns:
        df['competition_id'] = pd.factorize(df['competition'])[0]
        df['competition_id'] = df['competition_id'].astype('category')
    else:
        df['competition_id'] = 0
        df['competition_id'] = df['competition_id'].astype('category')
        
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
        'competition_id',
        'team_elo', 'opp_elo', 'elo_diff',
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
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo B (Contexto) con Optuna ({OPTUNA_TRIALS} Trials)...")
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
                
    logger.info(f"Mejores parámetros XGBoost Contexto: {best_params}")
    
    xgb_best = XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda',
        enable_categorical=True
    )
    
    logger.info("Calculando predicciones OOF para Train (Contexto)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando KFold
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    from sklearn.model_selection import KFold
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para obtener OOF completos...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train)
        
        base_kf = XGBClassifier(**xgb_best.get_params())
        kf_estimator = CalibratedClassifierCV(estimator=base_kf, method='isotonic', cv=3)
        if w_tr is not None:
            kf_estimator.fit(X_kf_train, y_kf_train, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
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
        
        base_fold = XGBClassifier(**xgb_best.get_params())
        fold_estimator = CalibratedClassifierCV(estimator=base_fold, method='isotonic', cv=3)
        if w_tr is not None:
            fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            fold_estimator.fit(X_tr, y_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo B final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    base_final = XGBClassifier(**xgb_best.get_params())
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='isotonic', cv=3)
    if final_w_tr is not None:
        final_estimator.fit(X_train, y_train, sample_weight=final_w_tr.values if isinstance(final_w_tr, pd.Series) else final_w_tr)
    else:
        final_estimator.fit(X_train, y_train)
    pred_probs_test = final_estimator.predict_proba(X_test)
    
    # LOGS: Verificacion de calibracion
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO B ===")
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = pred_probs_train[:, 0].mean()
    pred_draw = pred_probs_train[:, 1].mean()
    pred_win = pred_probs_train[:, 2].mean()
    
    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Dataset: {real_win*100:.1f}%")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': final_estimator, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO CONTEXTO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_context()
