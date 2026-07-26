import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from scipy.stats import poisson, nbinom
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import json
import optuna

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
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
    if dates is None or dates.isna().all():
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

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

def calc_over_probs(lambda_total, true_totals=None, dispersion_c=None, lines=[7.5, 8.5, 9.5, 10.5, 11.5, 12.5]):
    """
    Calcula la probabilidad de OVER para distintas líneas de córners totales.
    Utiliza dispersión residual c = Var(residuales) / Mean(residuales).
    Si c > 1.05, aplica Binomial Negativa para corregir sobredispersión residual.
    """
    probs = {}
    c = 1.0
    if dispersion_c is not None:
        c = dispersion_c
    elif true_totals is not None and len(true_totals) > 0:
        residuals = true_totals - lambda_total
        mean_lam = np.mean(lambda_total)
        if mean_lam > 0:
            c = np.var(residuals) / mean_lam
        c = max(0.8, min(c, 2.0))
        
    for line in lines:
        k = int(np.floor(line))
        if c <= 1.02:
            prob_over = 1.0 - poisson.cdf(k, np.maximum(lambda_total, 1e-4))
        else:
            p = 1.0 / c
            n = np.maximum(lambda_total, 1e-4) * p / (1.0 - p)
            n = np.clip(n, a_min=1e-4, a_max=None)
            prob_over = 1.0 - nbinom.cdf(k, n, p)
            
        probs[f'prob_over_{line}'] = prob_over
    return probs, c

def get_double_row_datasets(df_home):
    """
    Construye las matrices de características para la perspectiva Local (is_home=1)
    y la perspectiva Visitante (is_home=0) mapeando away_corners por match_id.
    """
    path_referees = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/matches_with_referees.parquet'))
    path_odds = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/matches_with_odds.parquet'))
    full_path = path_referees if os.path.exists(path_referees) else path_odds
    
    full_df = pd.read_parquet(full_path, engine='fastparquet')
    if 'match_date' in full_df.columns:
        full_df = full_df.sort_values('match_date').reset_index(drop=True)

    df_away_map = full_df[full_df['is_home'] == 0].drop_duplicates('match_id').set_index('match_id')
    df_home = df_home.copy()
    df_home['away_corners'] = df_home['match_id'].map(df_away_map['corners']).fillna(0)
    df_home['true_total_corners'] = df_home['corners'] + df_home['away_corners']
    
    feature_cols = [
        'crosses_ema3', 'crosses_ema5',
        'possession_pct_ema3', 'possession_pct_ema5',
        'shots_total_ema3', 'shots_total_ema5',
        'corners_ema3', 'corners_ema5',
        'opp_crosses_ema3', 'opp_crosses_ema5',
        'opp_possession_pct_ema3', 'opp_possession_pct_ema5',
        'opp_shots_total_ema3', 'opp_shots_total_ema5',
        'opp_corners_ema3', 'opp_corners_ema5',
        'team_att_rating', 'team_def_rating', 
        'opp_att_rating', 'opp_def_rating',
        'elo_diff', 'relative_attack_strength',
        'xg_momentum_macd', 'is_home'
    ]
    feature_cols = [c for c in feature_cols if c in df_home.columns]
    
    # Construir vista Visitante simétrica
    df_away_aligned = df_home.copy()
    df_away_aligned['is_home'] = 0
    if 'elo_diff' in df_away_aligned.columns:
        df_away_aligned['elo_diff'] = -df_away_aligned['elo_diff']
    if 'relative_attack_strength' in df_away_aligned.columns:
        df_away_aligned['relative_attack_strength'] = -df_away_aligned['relative_attack_strength']
    if 'xg_momentum_macd' in df_away_aligned.columns:
        df_away_aligned['xg_momentum_macd'] = -df_away_aligned['xg_momentum_macd']

    # Intercambiar EMAs del equipo y del rival para la perspectiva visitante
    for stat in ['crosses', 'possession_pct', 'shots_total', 'corners']:
        for span in [3, 5]:
            h_col = f'{stat}_ema{span}'
            o_col = f'opp_{stat}_ema{span}'
            if h_col in df_away_aligned.columns and o_col in df_away_aligned.columns:
                df_away_aligned[h_col], df_away_aligned[o_col] = df_home[o_col], df_home[h_col]

    for r1, r2 in [('team_att_rating', 'opp_att_rating'), ('team_def_rating', 'opp_def_rating')]:
        if r1 in df_away_aligned.columns and r2 in df_away_aligned.columns:
            df_away_aligned[r1], df_away_aligned[r2] = df_home[r2], df_home[r1]

    X_home = df_home[feature_cols].fillna(0).copy()
    X_away = df_away_aligned[feature_cols].fillna(0).copy()
    
    y_home = df_home['corners'].fillna(0).copy()
    y_away = df_home['away_corners'].fillna(0).copy()

    return df_home, X_home, X_away, y_home, y_away, feature_cols

def train_corners():
    df_base = get_base_dataset()
    split_idx = get_train_test_split(df_base)
    
    if 'corners' not in df_base.columns:
        logger.error("No se encontró la columna 'corners' en el dataset. Verifica feature_engineering.py.")
        return

    logger.info("=== PREPARANDO DATASETS CON PERSPECTIVA DOBLE (LOCAL + VISITANTE) ===")
    df_home, X_home, X_away, y_home, y_away, feature_cols = get_double_row_datasets(df_base)
    
    # Split de partidos
    X_home_train, X_home_test = X_home.iloc[:split_idx].copy(), X_home.iloc[split_idx:].copy()
    X_away_train, X_away_test = X_away.iloc[:split_idx].copy(), X_away.iloc[split_idx:].copy()
    
    y_home_train, y_home_test = y_home.iloc[:split_idx].copy(), y_home.iloc[split_idx:].copy()
    y_away_train, y_away_test = y_away.iloc[:split_idx].copy(), y_away.iloc[split_idx:].copy()

    match_ids_train = df_home['match_id'].iloc[:split_idx].reset_index(drop=True)
    
    train_dates = None
    if 'match_date' in df_home.columns:
        train_dates = pd.to_datetime(df_home['match_date'].iloc[:split_idx]).reset_index(drop=True)

    logger.info("=== ENTRENANDO MODELO DE CÓRNERS (POISSON UNIFICADO) ===")

    # Preparar dataset doble para entrenamiento
    X_double_train = pd.concat([X_home_train, X_away_train], axis=0).reset_index(drop=True)
    y_double_train = pd.concat([y_home_train, y_away_train], axis=0).reset_index(drop=True)
    
    dates_double_train = None
    if train_dates is not None:
        dates_double_train = pd.concat([train_dates, train_dates], axis=0).reset_index(drop=True)

    # Optuna con TimeSeriesSplit sobre partidos
    tscv = get_cv_strategy(n_splits=5)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        scores = []
        for tr_m_idx, va_m_idx in tscv.split(X_home_train):
            # Construir conjunto doble de entrenamiento para este fold
            X_tr_fold = pd.concat([X_home_train.iloc[tr_m_idx], X_away_train.iloc[tr_m_idx]], axis=0)
            y_tr_fold = pd.concat([y_home_train.iloc[tr_m_idx], y_away_train.iloc[tr_m_idx]], axis=0)
            
            w_tr_fold = None
            if train_dates is not None:
                d_tr = pd.concat([train_dates.iloc[tr_m_idx], train_dates.iloc[tr_m_idx]], axis=0)
                w_tr_fold = get_time_weights(d_tr)
                
            model = get_xgb_corners_model(**params)
            model.fit(X_tr_fold, y_tr_fold, sample_weight=w_tr_fold)
            
            # Predecir sobre la validación (Home + Away)
            pred_h = model.predict(X_home_train.iloc[va_m_idx])
            pred_a = model.predict(X_away_train.iloc[va_m_idx])
            lam_tot = pred_h + pred_a
            y_tot = y_home_train.iloc[va_m_idx].values + y_away_train.iloc[va_m_idx].values
            
            # Poisson NLL deviance sobre el total
            nll = np.mean(lam_tot - y_tot * np.log(lam_tot + 1e-9))
            scores.append(nll)
            
        return np.mean(scores)

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

    # === GENERAR PREDICCIONES OOF PARA EL ENTRENAMIENTO ===
    logger.info("Generando predicciones Out-Of-Fold (OOF) para el Train Set...")
    pred_home_train = np.zeros(len(X_home_train))
    pred_away_train = np.zeros(len(X_away_train))
    
    splits = list(tscv.split(X_home_train))
    first_train_idx = splits[0][0]
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} partidos) con GroupKFold(5)...")
    kf = GroupKFold(n_splits=5)
    groups_first = match_ids_train.iloc[first_train_idx]
    
    for kf_tr, kf_va in kf.split(first_train_idx, groups=groups_first):
        tr_indices = first_train_idx[kf_tr]
        va_indices = first_train_idx[kf_va]
        
        X_tr_fold = pd.concat([X_home_train.iloc[tr_indices], X_away_train.iloc[tr_indices]], axis=0)
        y_tr_fold = pd.concat([y_home_train.iloc[tr_indices], y_away_train.iloc[tr_indices]], axis=0)
        
        w_tr_fold = None
        if train_dates is not None:
            d_tr = pd.concat([train_dates.iloc[tr_indices], train_dates.iloc[tr_indices]], axis=0)
            w_tr_fold = get_time_weights(d_tr)
            
        fold_est = configured_xgb_corners_model()
        fold_est.fit(X_tr_fold, y_tr_fold, sample_weight=w_tr_fold)
        
        pred_home_train[va_indices] = fold_est.predict(X_home_train.iloc[va_indices])
        pred_away_train[va_indices] = fold_est.predict(X_away_train.iloc[va_indices])

    # Folds temporales expansivos
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr_fold = pd.concat([X_home_train.iloc[train_idx], X_away_train.iloc[train_idx]], axis=0)
        y_tr_fold = pd.concat([y_home_train.iloc[train_idx], y_away_train.iloc[train_idx]], axis=0)
        
        w_tr_fold = None
        if train_dates is not None:
            d_tr = pd.concat([train_dates.iloc[train_idx], train_dates.iloc[train_idx]], axis=0)
            w_tr_fold = get_time_weights(d_tr)
            
        fold_est = configured_xgb_corners_model()
        fold_est.fit(X_tr_fold, y_tr_fold, sample_weight=w_tr_fold)
        
        pred_home_train[val_idx] = fold_est.predict(X_home_train.iloc[val_idx])
        pred_away_train[val_idx] = fold_est.predict(X_away_train.iloc[val_idx])

    # Entrenar modelo final sobre todo el Train Set
    logger.info("Entrenando modelo final de Córners sobre todo el Train Set...")
    w_final_train = get_time_weights(dates_double_train) if dates_double_train is not None else None
    
    final_xgb = configured_xgb_corners_model()
    final_xgb.fit(X_double_train, y_double_train, sample_weight=w_final_train)
    
    pred_home_test = final_xgb.predict(X_home_test)
    pred_away_test = final_xgb.predict(X_away_test)

    # === CONVOLUCIÓN POISSON & CÁLCULO DE PROBABILIDADES DE LÍNEAS ===
    logger.info("=== CALCULANDO PROBABILIDADES DE LÍNEAS OVER/UNDER EN CÓRNERS TOTALES ===")
    
    lambda_total_train = pred_home_train + pred_away_train
    lambda_total_test = pred_home_test + pred_away_test
    
    true_total_train = (y_home_train.values + y_away_train.values)
    true_total_test = (y_home_test.values + y_away_test.values)

    lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

    train_probs_dict, c_dispersion = calc_over_probs(
        lambda_total=lambda_total_train,
        true_totals=true_total_train,
        lines=lines
    )
    
    if c_dispersion > 1.02:
        logger.info(f"-> Ajuste por dispersión residual (c={c_dispersion:.3f}). Usando Distribución Binomial Negativa.")
    else:
        logger.info(f"-> Dispersión residual óptima (c={c_dispersion:.3f}). Usando Distribución de Poisson.")

    test_probs_dict, _ = calc_over_probs(
        lambda_total=lambda_total_test,
        dispersion_c=c_dispersion,
        lines=lines
    )

    logger.info("=== CALIBRANDO PROBABILIDADES (ISOTONIC REGRESSION) ===")
    calibrators = {}
    calibrated_train_probs = {}
    calibrated_test_probs = {}
    
    for line in lines:
        col = f'prob_over_{line}'
        true_over_train = (true_total_train > line).astype(int)
        
        ir = IsotonicRegression(out_of_bounds='clip')
        calibrated_train_probs[col] = ir.fit_transform(train_probs_dict[col], true_over_train)
        calibrated_test_probs[col] = ir.predict(test_probs_dict[col])
        calibrators[line] = ir

    # AUDITORÍA DE RESULTADOS
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO DE CÓRNERS ===")
    logger.info(f" - Córners Locales (Real vs Pred): Train={y_home_train.mean():.2f} vs {pred_home_train.mean():.2f} | Test={y_home_test.mean():.2f} vs {pred_home_test.mean():.2f}")
    logger.info(f" - Córners Visitantes (Real vs Pred): Train={y_away_train.mean():.2f} vs {pred_away_train.mean():.2f} | Test={y_away_test.mean():.2f} vs {pred_away_test.mean():.2f}")
    logger.info(f" - Córners Totales Partido (Real vs Lambda): Train={true_total_train.mean():.2f} vs {lambda_total_train.mean():.2f} | Test={true_total_test.mean():.2f} vs {lambda_total_test.mean():.2f}")

    for line in lines:
        true_over_tr = (true_total_train > line).astype(int)
        true_over_ts = (true_total_test > line).astype(int)
        
        p_tr = calibrated_train_probs[f'prob_over_{line}']
        p_ts = calibrated_test_probs[f'prob_over_{line}']
        
        brier_tr = brier_score_loss(true_over_tr, p_tr)
        brier_ts = brier_score_loss(true_over_ts, p_ts)
        
        ll_tr = log_loss(true_over_tr, p_tr)
        ll_ts = log_loss(true_over_ts, p_ts)
        
        acc_tr = accuracy_score(true_over_tr, (p_tr > 0.5).astype(int))
        acc_ts = accuracy_score(true_over_ts, (p_ts > 0.5).astype(int))
        
        logger.info(f"\n--- MÉTRICAS PARA OVER {line} TOTAL ---")
        logger.info(f"Distribución Real O>{line}: Train={(true_over_tr.mean()*100):.1f}% | Test={(true_over_ts.mean()*100):.1f}%")
        logger.info(f"Log-Loss: Train={ll_tr:.4f} | Test={ll_ts:.4f}")
        logger.info(f"Brier Score: Train={brier_tr:.4f} | Test={brier_ts:.4f}")
        logger.info(f"Accuracy: Train={acc_tr:.4f} | Test={acc_ts:.4f}")

    # GUARDADO DE RESULTADOS
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    oof_train_dict = {
        'pred_corners': pred_home_train,
        'opp_pred_corners': pred_away_train,
        'lambda_total': lambda_total_train
    }
    oof_test_dict = {
        'pred_corners': pred_home_test,
        'opp_pred_corners': pred_away_test,
        'lambda_total': lambda_total_test
    }
    
    for line in lines:
        col = f'prob_over_{line}'
        oof_train_dict[col] = calibrated_train_probs[col]
        oof_test_dict[col] = calibrated_test_probs[col]
        
    oof_train_df = pd.DataFrame(oof_train_dict, index=df_home.iloc[:split_idx].index)
    oof_test_df = pd.DataFrame(oof_test_dict, index=df_home.iloc[split_idx:].index)

    oof_train_df.to_parquet(os.path.join(PROCESSED_DIR, 'oof_corners_train.parquet'), engine='fastparquet')
    oof_test_df.to_parquet(os.path.join(PROCESSED_DIR, 'oof_corners_test.parquet'), engine='fastparquet')
    logger.info(f"\nArchivos OOF guardados exitosamente. (Rows Train: {len(oof_train_df)}, Rows Test: {len(oof_test_df)})")

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    joblib.dump({
        'model_corners': final_xgb,
        'features': feature_cols,
        'lines': lines,
        'calibrators': calibrators,
        'c_dispersion': c_dispersion
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO CÓRNERS FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_corners()
