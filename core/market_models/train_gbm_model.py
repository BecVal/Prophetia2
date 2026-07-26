import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger


# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = True
OPTUNA_TRIALS = 30
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_gbm_model.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_gbm')
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'gbm_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def safe_logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def shin_margin_removal(odds_w, odds_d, odds_l):
    """
    Desmargina cuotas 1X2 aplicando el Método de Shin (Shin 1993).
    Estima la proporción de apostadores informados (insiders - z) y calcula
    las probabilidades reales verdaderas libres del Favorite-Longshot Bias.
    """
    ow = np.array(odds_w, dtype=float)
    od = np.array(odds_d, dtype=float)
    ol = np.array(odds_l, dtype=float)
    
    ow = np.where(ow > 1.001, ow, 1.001)
    od = np.where(od > 1.001, od, 1.001)
    ol = np.where(ol > 1.001, ol, 1.001)
    
    q_w = 1.0 / ow
    q_d = 1.0 / od
    q_l = 1.0 / ol
    
    q = np.column_stack([q_w, q_d, q_l])
    beta = q.sum(axis=1, keepdims=True)
    
    # Estimación inicial de z (proporción insider)
    z = (beta - 1.0) / np.clip(beta, 1.001, None)
    z = np.clip(z, 1e-5, 0.4)
    
    # Newton-Raphson vectorizado para resolver z de Shin: sum(p_i(z)) = 1
    for _ in range(8):
        term = np.sqrt(z**2 + 4 * (1.0 - z) * (q**2) / beta)
        p = (term - z) / (2 * (1.0 - z))
        f = p.sum(axis=1, keepdims=True) - 1.0
        
        dp_dz = ((z - 2 * (q**2) / beta) / term - 1.0) / (2 * (1.0 - z)) + (term - z) / (2 * (1.0 - z)**2)
        df_dz = dp_dz.sum(axis=1, keepdims=True)
        df_dz = np.where(np.abs(df_dz) < 1e-9, -1e-9, df_dz)
        
        z = z - f / df_dz
        z = np.clip(z, 1e-6, 0.5)
        
    term = np.sqrt(z**2 + 4 * (1.0 - z) * (q**2) / beta)
    p_shin = (term - z) / (2 * (1.0 - z))
    p_shin = p_shin / p_shin.sum(axis=1, keepdims=True)
    
    p_simple = q / beta
    valid = (beta.ravel() > 0.99) & (beta.ravel() < 1.6)
    p_final = np.where(valid[:, None], p_shin, p_simple)
    z_final = np.where(valid, z.ravel(), 0.0)
    
    return p_final[:, 0], p_final[:, 1], p_final[:, 2], z_final

def compute_gbm_features(df):
    df = df.copy()
    
    req_cols = ['open_odds_win', 'open_odds_draw', 'open_odds_loss', 'odds_win', 'odds_draw', 'odds_loss', 'team']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logger.error(f"Faltan columnas para calcular GBM: {missing}")
        return None
        
    # 1. Remoción de Margen y Estimación de Insiders por Método de Shin (Apertura y Cierre)
    logger.info("Aplicando Método de Shin (1993) para remoción de margen y extracción de dinero informado (insider z)...")
    p_open_w, p_open_d, p_open_l, z_open = shin_margin_removal(df['open_odds_win'], df['open_odds_draw'], df['open_odds_loss'])
    p_w, p_d, p_l, z_close = shin_margin_removal(df['odds_win'], df['odds_draw'], df['odds_loss'])
    
    df['shin_base_prob_win'] = p_w
    df['shin_base_prob_draw'] = p_d
    df['shin_base_prob_loss'] = p_l
    df['shin_insider_open'] = z_open
    df['shin_insider_close'] = z_close
    df['shin_insider_drift'] = z_close - z_open
    
    # 2. Logits y Deriva Estocástica (mu) sobre Probabilidades Shin
    logit_open_w = safe_logit(p_open_w)
    logit_open_d = safe_logit(p_open_d)
    logit_open_l = safe_logit(p_open_l)
    
    logit_w = safe_logit(p_w)
    logit_d = safe_logit(p_d)
    logit_l = safe_logit(p_l)
    
    df['gbm_mu_win'] = logit_w - logit_open_w
    df['gbm_mu_draw'] = logit_d - logit_open_d
    df['gbm_mu_loss'] = logit_l - logit_open_l
    
    # 3. Volatilidad (sigma) y Deriva con Ventanas Móviles y EWMA por Equipo
    logger.info("Calculando volatilidad (sigma), EWMA y desviaciones temporales (sin Data Leakage)...")
    
    df['orig_order'] = np.arange(len(df))
    df = df.sort_values(by=['team', 'match_date']).reset_index(drop=True)
    
    for outcome in ['win', 'draw', 'loss']:
        mu_col = f'gbm_mu_{outcome}'
        sigma_col = f'gbm_sigma_{outcome}'
        mean_drift_col = f'gbm_mean_{outcome}'
        ewma_mu_col = f'gbm_ewma_mu_{outcome}'
        ewma_std_col = f'gbm_ewma_std_{outcome}'
        z_col = f'gbm_z_{outcome}'
        
        grouped_mu = df.groupby('team')[mu_col]
        
        df[sigma_col] = grouped_mu.transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())
        df[mean_drift_col] = grouped_mu.transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
        
        df[ewma_mu_col] = grouped_mu.transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
        df[ewma_std_col] = grouped_mu.transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).std())
        
        # Imputación de NaNs usando estadísticas en expansión históricas
        exp_std = grouped_mu.transform(lambda x: x.shift(1).expanding(min_periods=1).std()).fillna(0.1)
        exp_mean = grouped_mu.transform(lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(0.0)
        
        df[sigma_col] = df[sigma_col].fillna(exp_std)
        df[mean_drift_col] = df[mean_drift_col].fillna(exp_mean)
        df[ewma_mu_col] = df[ewma_mu_col].fillna(exp_mean)
        df[ewma_std_col] = df[ewma_std_col].fillna(exp_std)
        
        raw_z = (df[mu_col] - df[mean_drift_col]) / df[sigma_col].replace(0, 1e-5)
        df[z_col] = raw_z.clip(-4.0, 4.0).fillna(0.0)
        
    df = df.sort_values(by='orig_order').drop(columns=['orig_order']).reset_index(drop=True)
    return df

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cpu',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr)
        
        xgb = XGBClassifier(**param)
        if w_tr is not None:
            xgb.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            xgb.fit(X_tr, y_tr)
            
        y_prob = xgb.predict_proba(X_val)
        cv_scores.append(log_loss(y_val, y_prob, labels=[0, 1, 2]))
        
    return np.mean(cv_scores)

def train_gbm_model():
    df = get_base_dataset()
    df_gbm = compute_gbm_features(df)
    
    if df_gbm is None:
        logger.error("Abortando entrenamiento GBM por falta de datos.")
        return
        
    split_idx = get_train_test_split(df_gbm)
    
    feature_cols = [
        'shin_base_prob_win', 'shin_base_prob_draw', 'shin_base_prob_loss',
        'shin_insider_open', 'shin_insider_close', 'shin_insider_drift',
        'gbm_mu_win', 'gbm_mu_draw', 'gbm_mu_loss',
        'gbm_sigma_win', 'gbm_sigma_draw', 'gbm_sigma_loss',
        'gbm_ewma_mu_win', 'gbm_ewma_mu_draw', 'gbm_ewma_mu_loss',
        'gbm_ewma_std_win', 'gbm_ewma_std_draw', 'gbm_ewma_std_loss',
        'gbm_z_win', 'gbm_z_draw', 'gbm_z_loss'
    ]
    
    X = df_gbm[feature_cols].fillna(0).copy()
    y = df_gbm['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df_gbm.columns:
        train_dates = pd.to_datetime(df_gbm['match_date'].iloc[:split_idx])
        
    cv_strategy = get_cv_strategy(n_splits=5)
    
    if RUN_OPTUNA:
        logger.info(f"Optimizando Modelo Cuantitativo GBM con Método de Shin y Optuna ({OPTUNA_TRIALS} Trials)...")
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
            logger.warning(f"Archivo {OPTUNA_PARAMS_FILE} no encontrado. Ejecutando Optuna como fallback.")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X_train, y_train, train_dates, cv_strategy), n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params, f, indent=4)
                
    logger.info(f"Mejores parámetros XGBoost GBM (Shin): {best_params}")
    
    xgb_best_params = {
        **best_params,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    logger.info("Calculando predicciones OOF temporales para Train (GBM + Shin)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    for i, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        base_fold = XGBClassifier(**xgb_best_params)
        fold_estimator = CalibratedClassifierCV(estimator=base_fold, method='sigmoid', cv=3)
        if w_tr is not None:
            fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            fold_estimator.fit(X_tr, y_tr)
            
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo GBM final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    base_final = XGBClassifier(**xgb_best_params)
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='sigmoid', cv=3)
    
    if final_w_tr is not None:
        final_estimator.fit(X_train, y_train, sample_weight=final_w_tr.values if isinstance(final_w_tr, pd.Series) else final_w_tr)
    else:
        final_estimator.fit(X_train, y_train)
        
    pred_probs_test = final_estimator.predict_proba(X_test)
    pred_probs_test = pred_probs_test / pred_probs_test.sum(axis=1, keepdims=True)
    
    # Normalizar OOF
    valid_mask = ~np.isnan(pred_probs_train[:, 0])
    pred_probs_train[valid_mask] = pred_probs_train[valid_mask] / pred_probs_train[valid_mask].sum(axis=1, keepdims=True)
    
    # Auditoría (Logs de Brier y Log Loss)
    valid_idx = valid_mask
    y_true_valid = y_train.iloc[valid_idx].values
    preds_valid = pred_probs_train[valid_idx]
    
    if len(preds_valid) > 0:
        logloss_val = log_loss(y_true_valid, preds_valid, labels=[0, 1, 2])
        
        brier_loss = np.mean((preds_valid[:, 0] - (y_true_valid == 0))**2)
        brier_draw = np.mean((preds_valid[:, 1] - (y_true_valid == 1))**2)
        brier_win  = np.mean((preds_valid[:, 2] - (y_true_valid == 2))**2)
        brier_total = np.mean(np.sum((preds_valid - np.eye(3)[y_true_valid])**2, axis=1))
        
        real_loss = np.mean(y_true_valid == 0)
        real_draw = np.mean(y_true_valid == 1)
        real_win = np.mean(y_true_valid == 2)
        
        pred_loss = np.mean(preds_valid[:, 0])
        pred_draw = np.mean(preds_valid[:, 1])
        pred_win = np.mean(preds_valid[:, 2])
        
        logger.info("=== AUDITORÍA Y RESULTADOS DEL MODELO GBM QUANT (MÉTODO DE SHIN) ===")
        logger.info(f" -> Log Loss Global (OOF): {logloss_val:.4f}")
        logger.info(f" -> Brier Score Total (OOF): {brier_total:.4f}")
        logger.info(f" - Derrota (Loss) | Pred: {pred_loss*100:.1f}% | Real: {real_loss*100:.1f}% | Brier: {brier_loss:.4f}")
        logger.info(f" - Empate (Draw)  | Pred: {pred_draw*100:.1f}% | Real: {real_draw*100:.1f}% | Brier: {brier_draw:.4f}")
        logger.info(f" - Victoria (Win) | Pred: {pred_win*100:.1f}% | Real: {real_win*100:.1f}% | Brier: {brier_win:.4f}")
        
        importances = np.mean([clf.estimator.feature_importances_ for clf in final_estimator.calibrated_classifiers_], axis=0)
        feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        logger.info("=== TOP 5 FEATURES GBM MÁS IMPORTANTES (SHIN) ===")
        for _, row in feat_imp.head(5).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
            
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_gbm', 'prob_draw_gbm', 'prob_win_gbm'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_gbm', 'prob_draw_gbm', 'prob_win_gbm'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_gbm_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_gbm_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': final_estimator, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== ENTRENAMIENTO GBM FINALIZADO === Modelo guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_gbm_model()
