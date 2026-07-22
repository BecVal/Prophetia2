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
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.calibration import CalibratedClassifierCV

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger


OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_market.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_market')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'market_model.pkl')
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
        calibrated_eval = CalibratedClassifierCV(estimator=xgb_eval, method='isotonic', cv=3)
        if w_tr is not None:
            calibrated_eval.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            calibrated_eval.fit(X_tr, y_tr)
        
        y_prob = calibrated_eval.predict_proba(X_val)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        cv_scores.append(log_loss(y_val, y_prob, labels=[0, 1, 2]))
        
    return np.mean(cv_scores)

def train_market():
    # El modelo de mercado requiere dataset con cuotas
    # data_splitter intenta cargar matches_with_odds.parquet por defecto
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Feature Engineering de Mercado
    # 1. Calcular True Odds para el Cierre (Margin Removal)
    if 'odds_win' in df.columns and 'prob_win_implied' not in df.columns:
        implied_win_c = 1 / df['odds_win']
        implied_draw_c = 1 / df['odds_draw']
        implied_loss_c = 1 / df['odds_loss']
        
        vig_close = implied_win_c + implied_draw_c + implied_loss_c
        df['vig_close'] = vig_close - 1
        
        # Margin Removal (Basic method)
        df['prob_win_implied'] = implied_win_c / vig_close
        df['prob_draw_implied'] = implied_draw_c / vig_close
        df['prob_loss_implied'] = implied_loss_c / vig_close

    # 2. Calcular True Odds para la Apertura
    if 'open_odds_win' in df.columns and 'open_prob_win' not in df.columns:
        implied_win_o = 1 / df['open_odds_win']
        implied_draw_o = 1 / df['open_odds_draw']
        implied_loss_o = 1 / df['open_odds_loss']
        
        vig_open = implied_win_o + implied_draw_o + implied_loss_o
        df['vig_open'] = vig_open - 1
        
        df['open_prob_win'] = implied_win_o / vig_open
        df['open_prob_draw'] = implied_draw_o / vig_open
        df['open_prob_loss'] = implied_loss_o / vig_open

    # 3. Steam usando True Odds (Cierre - Apertura)
    if 'prob_win_implied' in df.columns and 'open_prob_win' in df.columns:
        df['steam_win'] = df['prob_win_implied'] - df['open_prob_win']
        df['steam_draw'] = df['prob_draw_implied'] - df['open_prob_draw']
        df['steam_loss'] = df['prob_loss_implied'] - df['open_prob_loss']
        
    feature_cols = [
        'open_prob_win', 'open_prob_draw', 'open_prob_loss',
        'prob_win_implied', 'prob_draw_implied', 'prob_loss_implied',
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
    
    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo de Mercado con Optuna ({OPTUNA_TRIALS} Trials)...")
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
                
    logger.info(f"Mejores parámetros XGBoost Market: {best_params}")
    
    xgb_best = XGBClassifier(
        **best_params,
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
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para evitar NaNs OOF...")
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
        
    logger.info("Entrenando Modelo de Mercado final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    base_final = XGBClassifier(**xgb_best.get_params())
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='isotonic', cv=3)
    if final_w_tr is not None:
        final_estimator.fit(X_train, y_train, sample_weight=final_w_tr.values if isinstance(final_w_tr, pd.Series) else final_w_tr)
    else:
        final_estimator.fit(X_train, y_train)
    pred_probs_test = final_estimator.predict_proba(X_test)
    pred_probs_test = pred_probs_test / pred_probs_test.sum(axis=1, keepdims=True)
    
    # Normalizar OOF (silencia warnings de suma != 1)
    valid_mask = ~np.isnan(pred_probs_train[:, 0])
    pred_probs_train[valid_mask] = pred_probs_train[valid_mask] / pred_probs_train[valid_mask].sum(axis=1, keepdims=True)
    
    # LOGS: Verificacion y Calibración
    valid_idx = valid_mask
    y_true_valid = y_train.iloc[valid_idx].values
    preds_valid = pred_probs_train[valid_idx]
    
    if len(preds_valid) > 0:
        logloss_val = log_loss(y_true_valid, preds_valid, labels=[0, 1, 2])
        
        brier_loss = np.mean((preds_valid[:, 0] - (y_true_valid == 0))**2)
        brier_draw = np.mean((preds_valid[:, 1] - (y_true_valid == 1))**2)
        brier_win  = np.mean((preds_valid[:, 2] - (y_true_valid == 2))**2)
        
        real_loss = np.mean(y_true_valid == 0)
        real_draw = np.mean(y_true_valid == 1)
        real_win = np.mean(y_true_valid == 2)
        
        pred_loss = np.mean(preds_valid[:, 0])
        pred_draw = np.mean(preds_valid[:, 1])
        pred_win = np.mean(preds_valid[:, 2])
        
        logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO MERCADO ===")
        logger.info(f" -> Log Loss Global (OOF): {logloss_val:.4f}")
        logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real: {real_loss*100:.1f}% | Brier Score: {brier_loss:.4f}")
        logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real: {real_draw*100:.1f}% | Brier Score: {brier_draw:.4f}")
        logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real: {real_win*100:.1f}% | Brier Score: {brier_win:.4f}")
        
        # Feature Importances
        importances = np.mean([clf.estimator.feature_importances_ for clf in final_estimator.calibrated_classifiers_], axis=0)
        feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        logger.info("=== IMPORTANCIA DE VARIABLES (TOP 5) ===")
        for _, row in feat_imp.head(5).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_mkt', 'prob_draw_mkt', 'prob_win_mkt'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_market_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_market_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': final_estimator, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO MERCADO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_market()
