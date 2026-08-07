import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
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

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = True
OPTUNA_TRIALS = 100
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_draws.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_draws')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'draws_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def compute_poisson_draw_prob(lambda_home, lambda_away, max_goals=8):
    lh = np.maximum(0.2, lambda_home.to_numpy() if isinstance(lambda_home, pd.Series) else lambda_home)
    la = np.maximum(0.2, lambda_away.to_numpy() if isinstance(lambda_away, pd.Series) else lambda_away)
    draw_prob = np.zeros(len(lh))
    for k in range(max_goals + 1):
        draw_prob += poisson.pmf(k, lh) * poisson.pmf(k, la)
    return draw_prob

def compute_competition_draw_rate(df):
    if 'competition' in df.columns and 'outcome' in df.columns:
        df_copy = df[['competition', 'outcome']].copy()
        df_copy['is_draw_tmp'] = (df_copy['outcome'] == 0).astype(float)
        comp_rates = df_copy.groupby('competition')['is_draw_tmp'].transform('mean')
        return comp_rates.fillna(0.26)
    return pd.Series(0.26, index=df.index)

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def fit_calibrated_model(base_estimator, X_tr, y_tr, w_tr=None):
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method='isotonic', cv=3)
    try:
        if w_tr is not None:
            calibrated.fit(X_tr, y_tr, sample_weight=w_tr)
        else:
            calibrated.fit(X_tr, y_tr)
    except (TypeError, ValueError):
        calibrated.fit(X_tr, y_tr)
    return calibrated

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0)
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr)
        
        xgb_eval = XGBClassifier(**param_xgb)
        xgb_eval.fit(X_tr, y_tr, sample_weight=w_tr)
        
        y_prob = xgb_eval.predict_proba(X_val)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

class HybridDrawsEnsemble:
    def __init__(self, xgb_params):
        self.xgb_base = XGBClassifier(**xgb_params, objective='binary:logistic', random_state=42, device='cuda')
        self.lgb_base = LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            verbose=-1,
            max_depth=3,
            n_estimators=200,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7
        )
        self.xgb_cal = None
        self.lgb_cal = None

    def fit(self, X_tr, y_tr, w_tr=None):
        self.xgb_cal = fit_calibrated_model(self.xgb_base, X_tr, y_tr, w_tr)
        self.lgb_cal = fit_calibrated_model(self.lgb_base, X_tr, y_tr, w_tr)
        return self

    def predict_proba(self, X_val):
        p_xgb = self.xgb_cal.predict_proba(X_val)
        p_lgb = self.lgb_cal.predict_proba(X_val)
        return 0.5 * p_xgb + 0.5 * p_lgb

def train_draws():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Feature Engineering de Nivel 10 para Cazar Empates
    lh = df['xg_created_ema5'].fillna(1.2) if 'xg_created_ema5' in df.columns else pd.Series(1.2, index=df.index)
    la = df['opp_xg_created_ema5'].fillna(1.2) if 'opp_xg_created_ema5' in df.columns else pd.Series(1.2, index=df.index)
    df['poisson_draw_prob'] = compute_poisson_draw_prob(lh, la)
    df['competition_draw_rate'] = compute_competition_draw_rate(df)

    if 'odds_draw' in df.columns:
        df['implied_prob_draw'] = np.where(df['odds_draw'] > 0, 1.0 / df['odds_draw'], 0.26)
    if 'open_odds_draw' in df.columns:
        df['open_implied_prob_draw'] = np.where(df['open_odds_draw'] > 0, 1.0 / df['open_odds_draw'], 0.26)
    if 'odds_win' in df.columns and 'odds_loss' in df.columns:
        p_win = np.where(df['odds_win'] > 0, 1.0 / df['odds_win'], 0.37)
        p_loss = np.where(df['odds_loss'] > 0, 1.0 / df['odds_loss'], 0.37)
        df['odds_match_balance'] = np.abs(p_win - p_loss)

    if 'elo_diff' in df.columns:
        df['abs_elo_diff'] = df['elo_diff'].abs()

    if 'xg_created_ema5' in df.columns and 'opp_xg_created_ema5' in df.columns:
        df['expected_match_xg'] = df['xg_created_ema5'].fillna(1.3) + df['opp_xg_created_ema5'].fillna(1.3)

    # Features optimizadas (33 características avanzadas)
    feature_cols = [
        'poisson_draw_prob', 'competition_draw_rate',
        'rest_days', 'rest_diff',
        'team_squad_value', 'opp_squad_value', 'squad_value_diff',
        'h2h_games_played', 'h2h_draw_rate_hist', 'h2h_win_rate_hist',
        'win_streak_3', 'loss_streak_3', 'xg_momentum_macd',
        'opp_win_streak_3', 'opp_loss_streak_3', 'opp_xg_momentum_macd',
        'fatigue_index', 'fatigue_diff', 'xg_volatility_5', 'opp_xg_volatility_5', 'volatility_diff',
        'team_elo', 'opp_elo', 'abs_elo_diff',
        'odds_draw', 'implied_prob_draw', 'open_implied_prob_draw', 'odds_match_balance',
        'expected_match_xg', 'team_att_rating', 'team_def_rating', 'opp_att_rating', 'opp_def_rating'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan las siguientes columnas opcionales en Draws: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info(f"Usando {len(feature_cols)} características de nivel 10 para el modelo Caza-Empates.")

    X = df[feature_cols].fillna(0).copy()
    
    # TARGET BINARIO: 1 si es Empate, 0 si no.
    y_multi = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y = (y_multi == 1).astype(int)
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
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
    
    logger.info("Calculando predicciones OOF con Ensamble Híbrido (XGBoost + LightGBM)...")
    pred_probs_train = np.zeros((len(X_train), 2))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Fold inicial
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para OOF completo...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train)
        
        hybrid_fold = HybridDrawsEnsemble(best_params)
        hybrid_fold.fit(X_kf_train, y_kf_train, w_tr)
        
        val_indices_in_original = first_train_idx[kf_val]
        pred_probs_train[val_indices_in_original] = hybrid_fold.predict_proba(X_kf_val)

    # 2. Expanding Windows estándar para el resto
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        hybrid_fold = HybridDrawsEnsemble(best_params)
        hybrid_fold.fit(X_tr, y_tr, w_tr)
        pred_probs_train[val_idx] = hybrid_fold.predict_proba(X_val)
        
    logger.info("Entrenando Modelo Ensamble Híbrido final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    final_model = HybridDrawsEnsemble(best_params)
    final_model.fit(X_train, y_train, final_w_tr)
    pred_probs_test = final_model.predict_proba(X_test)
    
    prob_draw_train = pred_probs_train[:, 1]
    prob_draw_test = pred_probs_test[:, 1]
    
    # METRICAS DE EVALUACIÓN
    oof_ll = log_loss(y_train, prob_draw_train)
    oof_brier = brier_score_loss(y_train, prob_draw_train)
    oof_auc = roc_auc_score(y_train, prob_draw_train)
    
    test_ll = log_loss(y_test, prob_draw_test)
    test_brier = brier_score_loss(y_test, prob_draw_test)
    test_auc = roc_auc_score(y_test, prob_draw_test)
    
    real_draw = y_train.mean()
    pred_draw = prob_draw_train.mean()
    real_draw_test = y_test.mean()
    pred_draw_test = prob_draw_test.mean()
    
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO DRAWS 10/10 ===")
    logger.info(f" - Empate (Draw) TRAIN | Predicha: {pred_draw*100:.2f}% | Real: {real_draw*100:.2f}%")
    logger.info(f" - Empate (Draw) TEST  | Predicha: {pred_draw_test*100:.2f}% | Real: {real_draw_test*100:.2f}%")
    logger.info(f" - METRICAS OOF  -> LogLoss: {oof_ll:.4f} | BrierScore: {oof_brier:.4f} | ROC-AUC: {oof_auc:.4f}")
    logger.info(f" - METRICAS TEST -> LogLoss: {test_ll:.4f} | BrierScore: {test_brier:.4f} | ROC-AUC: {test_auc:.4f}")
    
    # AUDITORÍA DE IMPORTANCIA DE CARACTERÍSTICAS (XGBoost base del Ensamble)
    try:
        raw_xgb = final_model.xgb_base
        raw_xgb.fit(X_train, y_train, sample_weight=final_w_tr)
        importances = pd.Series(raw_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
        logger.info(" Top 10 Características Clave para Cazar Empates:")
        for col_name, score in importances.head(10).items():
            logger.info(f"   * {col_name:25s}: {score*100:.2f}%")
    except Exception as e:
        logger.warning(f"No se pudo imprimir importancia de características: {e}")

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
    logger.info(f"=== MODELO DRAWS 10/10 FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_draws()
