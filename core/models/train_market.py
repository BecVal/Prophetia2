import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'market_model.pkl')
PROCESSED_DIR = '../data/processed'

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
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
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

def train_market():
    # El modelo de mercado requiere dataset con cuotas
    # data_splitter intenta cargar matches_with_odds.parquet por defecto
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Feature Engineering de Mercado
    if 'open_prob_win' in df.columns and 'prob_win_implied' not in df.columns:
        # Aproximar probs de cierre si están las cuotas de cierre
        if 'odds_win' in df.columns:
            df['prob_win_implied'] = 1 / df['odds_win']
            df['prob_draw_implied'] = 1 / df['odds_draw']
            df['prob_loss_implied'] = 1 / df['odds_loss']
        else:
            df['prob_win_implied'] = df['open_prob_win']
            df['prob_draw_implied'] = df['open_prob_draw']
            df['prob_loss_implied'] = df['open_prob_loss']
            
    if 'prob_win_implied' in df.columns and 'open_prob_win' in df.columns:
        # Steam (Movimiento de cuotas)
        df['steam_win'] = df['prob_win_implied'] - df['open_prob_win']
        df['steam_draw'] = df['prob_draw_implied'] - df['open_prob_draw']
        df['steam_loss'] = df['prob_loss_implied'] - df['open_prob_loss']
        
        # Vig (Margen de la casa de apuestas)
        df['vig_open'] = df['open_prob_win'] + df['open_prob_draw'] + df['open_prob_loss'] - 1
        df['vig_close'] = df['prob_win_implied'] + df['prob_draw_implied'] + df['prob_loss_implied'] - 1
        
    feature_cols = [
        'open_prob_win', 'open_prob_draw', 'open_prob_loss',
        'steam_win', 'steam_draw', 'steam_loss',
        'vig_open', 'vig_close'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan las siguientes columnas en Market: {missing_cols}. Usando lo que hay.")
        feature_cols = [c for c in feature_cols if c in df.columns]
        
    if not feature_cols:
        logger.error("No hay variables de mercado disponibles. Abortando train_market.")
        return

    X = df[feature_cols].fillna(0).copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    logger.info("Optimizando Modelo de Mercado con Optuna (20 Trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, train_dates, cv_strategy), n_trials=20)
    
    logger.info(f"Mejores parámetros XGBoost Market: {study.best_params}")
    
    xgb_best = XGBClassifier(
        **study.best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    
    logger.info("Calculando predicciones OOF para Train (Market)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando KFold
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para evitar Leakage...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train)
        
        kf_estimator = XGBClassifier(**xgb_best.get_params())
        kf_estimator.fit(X_kf_train, y_kf_train, sample_weight=w_tr)
        
        val_indices_in_original = first_train_idx[kf_val]
        pred_probs_train[val_indices_in_original] = kf_estimator.predict_proba(X_kf_val)

    # 2. Expanding Windows estándar para el resto
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        fold_estimator = XGBClassifier(**xgb_best.get_params())
        fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo de Mercado final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    xgb_best.fit(X_train, y_train, sample_weight=final_w_tr)
    pred_probs_test = xgb_best.predict_proba(X_test)
    
    # LOGS: Verificacion
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = pred_probs_train[:, 0].mean()
    pred_draw = pred_probs_train[:, 1].mean()
    pred_win = pred_probs_train[:, 2].mean()
    
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO MERCADO ===")
    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Dataset: {real_win*100:.1f}%")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_market_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_market_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': xgb_best, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO MERCADO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_market()
