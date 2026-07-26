import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
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

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA) - EDICIÓN 11/10
# ==============================================================================
RUN_OPTUNA = True
OPTUNA_TRIALS = 80
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_context.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_context')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'context_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None or dates.isna().all():
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.total_seconds() / 86400.0
    days_diff = days_diff.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    half_life = trial.suggest_int('half_life_days', 180, 720)
    
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 120, 400),
        'subsample': trial.suggest_float('subsample', 0.55, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'enable_categorical': True
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr, half_life_days=half_life)
        
        xgb_eval = XGBClassifier(**param)
        calibrated_eval = CalibratedClassifierCV(estimator=xgb_eval, method='sigmoid', cv=3)
        if w_tr is not None:
            calibrated_eval.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            calibrated_eval.fit(X_tr, y_tr)
        
        y_prob = calibrated_eval.predict_proba(X_val)
        y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
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

    # ==============================================================================
    # FEATURE ENGINEERING DE ELITE (11/10): PRIORES MATEMÁTICOS & INTERACCIONES
    # ==============================================================================
    # 1. Logistic ELO Win Probability Prior
    if 'elo_diff' in df.columns:
        home_adv = np.where(df['is_home'] == 1, 100.0, -100.0) if 'is_home' in df.columns else 0.0
        df['elo_prob_win'] = 1.0 / (1.0 + 10.0 ** (-(df['elo_diff'] + home_adv) / 400.0))
    else:
        df['elo_prob_win'] = 0.5
    
    # 2. Scale-Invariant Log-Squad Value Ratio
    if 'team_squad_value' in df.columns and 'opp_squad_value' in df.columns:
        df['log_squad_ratio'] = np.log1p(df['team_squad_value'].clip(lower=0)) - np.log1p(df['opp_squad_value'].clip(lower=0))
    elif 'squad_value_diff' in df.columns:
        df['log_squad_ratio'] = df['squad_value_diff']
    else:
        df['log_squad_ratio'] = 0.0
    
    # 3. Curva Fisiológica de Recuperación (Rest Fatigue Non-Linear Decay)
    rest_team = df['rest_days'].clip(lower=0) if 'rest_days' in df.columns else pd.Series(4.0, index=df.index)
    rest_opp = df['opp_rest_days'].clip(lower=0) if 'opp_rest_days' in df.columns else (rest_team - df.get('rest_diff', 0)).clip(lower=0)
    df['recovery_index_team'] = 1.0 - np.exp(-rest_team / 3.0)
    df['recovery_index_opp'] = 1.0 - np.exp(-rest_opp / 3.0)
    df['recovery_diff'] = df['recovery_index_team'] - df['recovery_index_opp']

    # 4. ELO Ajustado por Volatilidad Táctica (Sharpe-type ELO Rating)
    vol = df['xg_volatility_5'].clip(lower=0.05) if 'xg_volatility_5' in df.columns else pd.Series(1.0, index=df.index)
    df['elo_sharpe_ratio'] = df.get('elo_diff', 0.0) / (vol + 1e-4)

    # 5. Sinergia ELO x Momentum de Inercia
    mom = df['xg_momentum_macd'] if 'xg_momentum_macd' in df.columns else pd.Series(0.0, index=df.index)
    df['elo_momentum_interaction'] = (df['elo_prob_win'] - 0.5) * mom

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
        'team_elo', 'opp_elo', 'elo_diff', 'elo_prob_win', 'elo_sharpe_ratio', 'elo_momentum_interaction',
        'is_home', 'rest_days', 'rest_diff', 'recovery_index_team', 'recovery_diff',
        'team_squad_value', 'opp_squad_value', 'squad_value_diff', 'log_squad_ratio',
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
        logger.info(f"Omitiendo columnas ausentes en el dataset: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    cat_cols = [c for c in feature_cols if X[c].dtype.name == 'category']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    X[num_cols] = X[num_cols].fillna(0)
    
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo B (Contexto 11/10) con Optuna ({OPTUNA_TRIALS} Trials)...")
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
                
    logger.info(f"Mejores parámetros XGBoost Contexto (11/10): {best_params}")
    
    half_life_opt = best_params.pop('half_life_days', 365)
    
    xgb_best_params = {
        **best_params,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'tree_method': 'hist',
        'enable_categorical': True
    }
    
    logger.info("Calculando predicciones OOF temporales para Train (Contexto)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. First fold TimeSeriesSplit(3)
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con TimeSeriesSplit(3)...")
    sub_tscv = TimeSeriesSplit(n_splits=3)
    for sub_tr_idx, sub_val_idx in sub_tscv.split(X_first, y_first):
        X_sub_tr, y_sub_tr = X_first.iloc[sub_tr_idx], y_first.iloc[sub_tr_idx]
        X_sub_val = X_first.iloc[sub_val_idx]
        
        dates_sub_tr = dates_first.iloc[sub_tr_idx] if dates_first is not None else None
        w_tr = get_time_weights(dates_sub_tr, half_life_days=half_life_opt)
        
        base_sub = XGBClassifier(**xgb_best_params)
        sub_estimator = CalibratedClassifierCV(estimator=base_sub, method='sigmoid', cv=3)
        if w_tr is not None:
            sub_estimator.fit(X_sub_tr, y_sub_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            sub_estimator.fit(X_sub_tr, y_sub_tr)
        
        val_indices_in_original = first_train_idx[sub_val_idx]
        pred_probs_train[val_indices_in_original] = sub_estimator.predict_proba(X_sub_val)

    # Rellenar micro-bloque inicial
    unfilled_mask = np.isnan(pred_probs_train[first_train_idx]).any(axis=1)
    if np.any(unfilled_mask):
        init_n = max(50, len(first_train_idx) // 4)
        X_init, y_init = X_first.iloc[:init_n], y_first.iloc[:init_n]
        dates_init = dates_first.iloc[:init_n] if dates_first is not None else None
        w_init = get_time_weights(dates_init, half_life_days=half_life_opt)
        
        base_init = XGBClassifier(**xgb_best_params)
        init_estimator = CalibratedClassifierCV(estimator=base_init, method='sigmoid', cv=3)
        if w_init is not None:
            init_estimator.fit(X_init, y_init, sample_weight=w_init.values if isinstance(w_init, pd.Series) else w_init)
        else:
            init_estimator.fit(X_init, y_init)
        
        unfilled_global_idx = first_train_idx[unfilled_mask]
        pred_probs_train[unfilled_global_idx] = init_estimator.predict_proba(X_train.iloc[unfilled_global_idx])

    # 2. Expanding Windows
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr, half_life_days=half_life_opt)
        
        base_fold = XGBClassifier(**xgb_best_params)
        fold_estimator = CalibratedClassifierCV(estimator=base_fold, method='sigmoid', cv=3)
        if w_tr is not None:
            fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            fold_estimator.fit(X_tr, y_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo B final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates, half_life_days=half_life_opt)
    base_final = XGBClassifier(**xgb_best_params)
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='sigmoid', cv=3)
    if final_w_tr is not None:
        final_estimator.fit(X_train, y_train, sample_weight=final_w_tr.values if isinstance(final_w_tr, pd.Series) else final_w_tr)
    else:
        final_estimator.fit(X_train, y_train)
    pred_probs_test = final_estimator.predict_proba(X_test)

    # Normalización estricta de probabilidades
    pred_probs_train = np.clip(pred_probs_train, 1e-7, 1.0 - 1e-7)
    pred_probs_train = pred_probs_train / pred_probs_train.sum(axis=1, keepdims=True)
    
    pred_probs_test = np.clip(pred_probs_test, 1e-7, 1.0 - 1e-7)
    pred_probs_test = pred_probs_test / pred_probs_test.sum(axis=1, keepdims=True)
    
    # ==============================================================================
    # MÉTRICAS Y AUDITORÍA ESTADÍSTICA DEL MODELO B (EDICIÓN 11/10)
    # ==============================================================================
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO B (EDICIÓN 11/10) ===")
    
    oof_acc = accuracy_score(y_train, np.argmax(pred_probs_train, axis=1))
    oof_loss = log_loss(y_train, pred_probs_train)
    
    y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_train_oh = np.zeros_like(pred_probs_train)
    y_train_oh[np.arange(len(y_train_arr)), y_train_arr] = 1
    
    brier_loss = brier_score_loss(y_train_oh[:, 0], pred_probs_train[:, 0])
    brier_draw = brier_score_loss(y_train_oh[:, 1], pred_probs_train[:, 1])
    brier_win = brier_score_loss(y_train_oh[:, 2], pred_probs_train[:, 2])
    brier_global = np.mean([brier_loss, brier_draw, brier_win])
    
    logger.info(f" - Half-Life Optimo de Decaimiento Temporal: {half_life_opt} días")
    logger.info(f" - Accuracy OOF: {oof_acc:.4f}")
    logger.info(f" - Log-Loss OOF: {oof_loss:.4f}")
    logger.info(f" - Brier Score Global OOF: {brier_global:.4f} (Loss: {brier_loss:.4f} | Draw: {brier_draw:.4f} | Win: {brier_win:.4f})")
    
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = pred_probs_train[:, 0].mean()
    pred_draw = pred_probs_train[:, 1].mean()
    pred_win = pred_probs_train[:, 2].mean()
    
    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Dataset: {real_win*100:.1f}%")
    
    # Feature Importances de los modelos calibrados
    importances = np.zeros(len(feature_cols))
    for cal_clf in final_estimator.calibrated_classifiers_:
        importances += cal_clf.estimator.feature_importances_
    importances /= len(final_estimator.calibrated_classifiers_)
    
    top_indices = np.argsort(importances)[::-1][:10]
    logger.info("=== TOP 10 CARACTERÍSTICAS MÁS INFLUYENTES ===")
    for idx in top_indices:
        logger.info(f"  * {feature_cols[idx]}: {importances[idx]:.4f}")

    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_ctx', 'prob_draw_ctx', 'prob_win_ctx'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_context_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    best_params['half_life_days'] = half_life_opt
    joblib.dump({'model': final_estimator, 'features': feature_cols, 'best_params': best_params}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO CONTEXTO (11/10) FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_context()
