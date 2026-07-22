import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
import json
import optuna

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
# Cambia RUN_OPTUNA a True si deseas volver a buscar los mejores hiperparámetros.
# De lo contrario (False), cargará los mejores guardados en el archivo JSON.
RUN_OPTUNA = True
OPTUNA_TRIALS = 30
# ==============================================================================

# Asegurar import de modulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_corners')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_corners.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'corners_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

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
    
    from sklearn.model_selection import KFold
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para obtener OOF completos...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train) if dates_kf_train is not None else None
        
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

def get_xgb_corners_model(**kwargs):
    params = {
        'objective': 'count:poisson',
        'n_estimators': 120,
        'learning_rate': 0.04,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'device': 'cuda'
    }
    params.update(kwargs)
    return XGBRegressor(**params)

def calc_over_probs(lambda_total, lines=[7.5, 8.5, 9.5, 10.5, 11.5]):
    """
    Calcula la probabilidad de OVER para distintas líneas utilizando la CDF de Poisson.
    Si la línea es 8.5, queremos P(X > 8) = 1 - P(X <= 8)
    """
    probs = {}
    for line in lines:
        k = int(np.floor(line))
        # P(X > k) = 1 - cdf(k)
        prob_over = 1.0 - poisson.cdf(k, lambda_total)
        probs[f'prob_over_{line}'] = prob_over
    return probs

def train_corners():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Validar que existe la variable corners
    if 'corners' not in df.columns:
        logger.error("No se encontró la columna 'corners' en el dataset. Verifica feature_engineering.py.")
        return

    # Features de Control y Ofensiva + Estado del Juego
    feature_cols = [
        'crosses_ema3', 'crosses_ema5',
        'possession_pct_ema3', 'possession_pct_ema5',
        'shots_total_ema3', 'shots_total_ema5',
        'corners_ema3', 'corners_ema5',
        'team_att_rating', 'team_def_rating', 
        'opp_att_rating', 'opp_def_rating',
        'elo_diff', 'relative_attack_strength',
        'xg_momentum_macd'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan variables en Corners: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    y_corners = df['corners'].fillna(0)
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y_corners.iloc[:split_idx], y_corners.iloc[split_idx:]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
        
    logger.info("=== ENTRENANDO MODELO DE CÓRNERS (POISSON) OOF ===")
    
    # Preparamos split para Optuna (último 20% del train)
    opt_split = int(len(X_train) * 0.8)
    X_opt_train, y_opt_train = X_train.iloc[:opt_split], y_train.iloc[:opt_split]
    X_opt_val, y_opt_val = X_train.iloc[opt_split:], y_train.iloc[opt_split:]
    w_opt_train = get_time_weights(train_dates.iloc[:opt_split]) if train_dates is not None else None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        model = get_xgb_corners_model(**params)
        model.fit(X_opt_train, y_opt_train, sample_weight=w_opt_train)
        preds = model.predict(X_opt_val)
        
        # Poisson deviance proxy (Negative Log-Likelihood)
        nloglik = np.mean(preds - y_opt_val * np.log(preds + 1e-9))
        return nloglik

    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo de Córners con Optuna ({OPTUNA_TRIALS} Trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
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
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params, f, indent=4)

    logger.info(f"Mejores parámetros XGBoost Corners: {best_params}")
    
    def configured_xgb_corners_model():
        return get_xgb_corners_model(**best_params)
    
    logger.info("Entrenando objetivo: Córners a favor...")
    pred_corners_train = get_expanding_predictions(configured_xgb_corners_model, X_train, y_train, train_dates)
    
    logger.info("Entrenando modelo final de Córners sobre todo el Train Set...")
    final_train_weights = get_time_weights(train_dates)
    
    xgb_model = configured_xgb_corners_model()
    xgb_model.fit(X_train, y_train, sample_weight=final_train_weights)
    pred_corners_test = xgb_model.predict(X_test)
    
    # --- CONVOLUCIÓN POISSON (Team A + Team B) ---
    logger.info("=== CALCULANDO PROBABILIDADES DE LÍNEAS OVER/UNDER ===")
    
    # Asignar predicciones al dataframe temporal para emparejar por partido
    df_train_tmp = df.iloc[:split_idx].copy()
    df_train_tmp['pred_corners'] = pred_corners_train
    
    df_test_tmp = df.iloc[split_idx:].copy()
    df_test_tmp['pred_corners'] = pred_corners_test
    
    # Obtener el lambda del oponente (Método vectorizado optimizado)
    df_train_tmp['opp_pred_corners'] = df_train_tmp.groupby('match_id')['pred_corners'].transform('sum') - df_train_tmp['pred_corners']
    df_test_tmp['opp_pred_corners'] = df_test_tmp.groupby('match_id')['pred_corners'].transform('sum') - df_test_tmp['pred_corners']
    
    # Total Lambda
    df_train_tmp['lambda_total'] = df_train_tmp['pred_corners'] + df_train_tmp['opp_pred_corners']
    df_test_tmp['lambda_total'] = df_test_tmp['pred_corners'] + df_test_tmp['opp_pred_corners']
    
    # True Total Corners
    df_train_tmp['opp_corners'] = df_train_tmp.groupby('match_id')['corners'].transform('sum') - df_train_tmp['corners']
    df_train_tmp['true_total_corners'] = df_train_tmp['corners'] + df_train_tmp['opp_corners']
    
    df_test_tmp['opp_corners'] = df_test_tmp.groupby('match_id')['corners'].transform('sum') - df_test_tmp['corners']
    df_test_tmp['true_total_corners'] = df_test_tmp['corners'] + df_test_tmp['opp_corners']
    
    lines = [7.5, 8.5, 9.5, 10.5, 11.5]
    
    # Calcular probabilidades para Train
    train_probs_dict = calc_over_probs(df_train_tmp['lambda_total'].values, lines)
    for col, vals in train_probs_dict.items():
        df_train_tmp[col] = vals
        
    # Calcular probabilidades para Test
    test_probs_dict = calc_over_probs(df_test_tmp['lambda_total'].values, lines)
    for col, vals in test_probs_dict.items():
        df_test_tmp[col] = vals
        
    logger.info("=== CALIBRANDO PROBABILIDADES (ISOTONIC REGRESSION) ===")
    calibrators = {}
    for line in lines:
        col = f'prob_over_{line}'
        true_over_train = (df_train_tmp['true_total_corners'] > line).astype(int).values
        
        # Entrenar calibrador isotónico para corregir sobredispersión
        ir = IsotonicRegression(out_of_bounds='clip')
        df_train_tmp[col] = ir.fit_transform(df_train_tmp[col].values, true_over_train)
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        
        calibrators[line] = ir
        
    # --- AUDITORÍA DE RESULTADOS ---
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO DE CÓRNERS ===")
    
    logger.info(f" - Media Córners por equipo (Real): Train={y_train.mean():.2f} | Test={y_test.mean():.2f}")
    logger.info(f" - Media Lambda Predicho: Train={pred_corners_train.mean():.2f} | Test={pred_corners_test.mean():.2f}")
    
    for line in lines:
        true_over_train = (df_train_tmp['true_total_corners'] > line).astype(int)
        true_over_test = (df_test_tmp['true_total_corners'] > line).astype(int)
        
        prob_train = df_train_tmp[f'prob_over_{line}']
        prob_test = df_test_tmp[f'prob_over_{line}']
        
        # Brier Score (MSE of probabilities)
        brier_train = brier_score_loss(true_over_train, prob_train)
        brier_test = brier_score_loss(true_over_test, prob_test)
        
        # Log Loss
        ll_train = log_loss(true_over_train, prob_train)
        ll_test = log_loss(true_over_test, prob_test)
        
        # Accuracy (Threshold 0.5)
        acc_train = accuracy_score(true_over_train, (prob_train > 0.5).astype(int))
        acc_test = accuracy_score(true_over_test, (prob_test > 0.5).astype(int))
        
        logger.info(f"\n--- MÉTRICAS PARA OVER {line} ---")
        logger.info(f"Distribución Real O>{line}: Train={(true_over_train.mean()*100):.1f}% | Test={(true_over_test.mean()*100):.1f}%")
        logger.info(f"Log-Loss: Train={ll_train:.4f} | Test={ll_test:.4f}")
        logger.info(f"Brier Score: Train={brier_train:.4f} | Test={brier_test:.4f}")
        logger.info(f"Accuracy: Train={acc_train:.4f} | Test={acc_test:.4f}")

    # --- GUARDADO ---
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    cols_to_save = ['pred_corners', 'opp_pred_corners', 'lambda_total'] + [f'prob_over_{L}' for L in lines]
    oof_train = df_train_tmp[cols_to_save].copy()
    oof_test = df_test_tmp[cols_to_save].copy()
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_corners_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_corners_test.parquet'), engine='fastparquet')
    
    logger.info(f"\nArchivos OOF guardados exitosamente. (Rows Train: {len(oof_train)}, Rows Test: {len(oof_test)})")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({
        'model_corners': xgb_model,
        'features': feature_cols,
        'lines': lines,
        'calibrators': calibrators
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO CÓRNERS FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_corners()
