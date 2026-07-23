import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, d2_tweedie_score
from sklearn.model_selection import TimeSeriesSplit, KFold

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..')))

from core.logger_config import get_logger
from core.models.data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logger = get_logger(__name__, 'train_cards_model')

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = False
OPTUNA_TRIALS = 30
# ==============================================================================

MODEL_SAVE_PATH = os.path.join(script_dir, 'save_models', 'cards_xgboost_model.pkl')
OPTUNA_PARAMS_FILE = os.path.join(script_dir, '..', '..', 'data', 'processed', 'models_best_parameters', 'optuna_params_cards.json')
PROCESSED_DIR = os.path.join(script_dir, '..', '..', 'data', 'processed')

def train_cards_model():
    logger.info("Loading dataset via data_splitter...")
    df = get_base_dataset()
    
    # 1. Definir Target: Booking Points del Equipo Local (ya que data_splitter filtra is_home==1)
    df['target_booking_points'] = df['yellow_cards'].fillna(0) + (df['red_cards'].fillna(0) * 2)
    
    # 2. Seleccionar Features
    features = [
        'fouls_committed_ema5',
        'fouls_won_ema5',
        'yellow_cards_ema5',
        'red_cards_ema5',
        'opp_fouls_committed_ema5',
        'opp_fouls_won_ema5',
        'opp_yellow_cards_ema5',
        'elo_diff',
        'h2h_points_last_5',
        'referee_avg_yellows',
        'referee_fouls_per_yellow',
        'referee_avg_fouls'
    ]
    
    # Rellenar NaNs en features
    for col in features:
        if col in df.columns:
            if col in ['elo_diff', 'h2h_points_last_5']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())

    available_features = [f for f in features if f in df.columns]
    logger.info(f"Features seleccionadas ({len(available_features)}): {available_features}")
    
    X = df[available_features]
    y = df['target_booking_points']
    
    # Train / Test split oficial (sincronizado con Stacker)
    split_idx = get_train_test_split(df)
    
    X_train = X.iloc[:split_idx].copy().reset_index(drop=True)
    y_train = y.iloc[:split_idx].copy().reset_index(drop=True)
    X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
    y_test = y.iloc[split_idx:].copy().reset_index(drop=True)
    
    os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
    
    if RUN_OPTUNA:
        logger.info(f"Running Optuna Optimization with {OPTUNA_TRIALS} trials...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'objective': 'count:poisson',
                'eval_metric': 'poisson-nloglik'
            }
            
            tscv = get_cv_strategy(n_splits=3)
            cv_scores = []
            
            for train_index, val_index in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[train_index], X_train.iloc[val_index]
                y_tr, y_va = y_train.iloc[train_index], y_train.iloc[val_index]
                
                model = XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                
                y_pred_va = model.predict(X_va)
                rmse = np.sqrt(mean_squared_error(y_va, y_pred_va))
                cv_scores.append(rmse)
                
            return np.mean(cv_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        
        logger.info(f"Mejores parámetros encontrados: {best_params}")
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        if os.path.exists(OPTUNA_PARAMS_FILE):
            logger.info(f"Cargando mejores parámetros desde {OPTUNA_PARAMS_FILE}")
            with open(OPTUNA_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
        else:
            logger.warning("No se encontró el archivo de parámetros, usando valores por defecto.")
            best_params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'objective': 'count:poisson'}
            
    # ====== OUT-OF-FOLD (OOF) PREDICTIONS ======
    logger.info("Generando predicciones Out-Of-Fold (OOF) para el Stacker...")
    y_oof_train = np.zeros(len(X_train))
    y_oof_train[:] = np.nan
    
    cv = get_cv_strategy(n_splits=5)
    splits = list(cv.split(X_train))
    
    # 1. K-Fold para el bloque inicial
    first_idx = splits[0][0]
    X_first, y_first = X_train.iloc[first_idx], y_train.iloc[first_idx]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_kf, va_kf in kf.split(X_first):
        X_tr, y_tr = X_first.iloc[tr_kf], y_first.iloc[tr_kf]
        X_va = X_first.iloc[va_kf]
        
        m = XGBRegressor(**best_params, objective='count:poisson', eval_metric='poisson-nloglik')
        m.fit(X_tr, y_tr, verbose=False)
        y_oof_train[first_idx[va_kf]] = m.predict(X_va)
        
    # 2. TimeSeriesSplit para el resto
    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        
        m = XGBRegressor(**best_params, objective='count:poisson', eval_metric='poisson-nloglik')
        m.fit(X_tr, y_tr, verbose=False)
        y_oof_train[va_idx] = m.predict(X_va)
        
    df_oof_train = pd.DataFrame({'lambda_total': y_oof_train}, index=df.iloc[:split_idx].index)
    oof_train_path = os.path.join(PROCESSED_DIR, 'oof_cards_train.parquet')
    df_oof_train.to_parquet(oof_train_path, engine='fastparquet')
    
    # ====== MODELO FINAL ======
    logger.info(f"Entrenando Modelo Final (count:poisson) on {len(X_train)} samples...")
    final_params = best_params.copy()
    final_params['objective'] = 'count:poisson'
    final_params['eval_metric'] = 'poisson-nloglik'
    
    model = XGBRegressor(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    df_oof_test = pd.DataFrame({'lambda_total': y_pred}, index=df.iloc[split_idx:].index)
    oof_test_path = os.path.join(PROCESSED_DIR, 'oof_cards_test.parquet')
    df_oof_test.to_parquet(oof_test_path, engine='fastparquet')
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    poisson_deviance_explained = d2_tweedie_score(y_test, y_pred, power=1)
    
    logger.info(f"--- MODEL METRICS (Total Match Booking Points) ---")
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"MAE  : {mae:.4f}")
    logger.info(f"Poisson Deviance Expl: {poisson_deviance_explained:.4f}")
    
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': available_features, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    
    logger.info(f"--- FEATURE IMPORTANCES ---")
    for _, row in feat_imp.iterrows():
        logger.info(f"{row['feature']:<25}: {row['importance']:.4f}")
        
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    logger.info("OOF files saved for Stacker consumption.")

if __name__ == '__main__':
    train_cards_model()
