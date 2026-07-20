import os
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

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

def compute_gbm_features(df):
    df = df.copy()
    
    # Verificar columnas necesarias
    req_cols = ['open_odds_win', 'open_odds_draw', 'open_odds_loss', 'odds_win', 'odds_draw', 'odds_loss', 'team']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logger.error(f"Faltan columnas para calcular GBM: {missing}")
        return None
        
    # Margin Removal Apertura
    inv_open_w = 1 / df['open_odds_win']
    inv_open_d = 1 / df['open_odds_draw']
    inv_open_l = 1 / df['open_odds_loss']
    vig_open = inv_open_w + inv_open_d + inv_open_l
    p_open_w = inv_open_w / vig_open
    p_open_d = inv_open_d / vig_open
    p_open_l = inv_open_l / vig_open
    
    # Margin Removal Cierre
    inv_w = 1 / df['odds_win']
    inv_d = 1 / df['odds_draw']
    inv_l = 1 / df['odds_loss']
    vig = inv_w + inv_d + inv_l
    p_w = inv_w / vig
    p_d = inv_d / vig
    p_l = inv_l / vig
    
    # Logits
    logit_open_w = safe_logit(p_open_w)
    logit_open_d = safe_logit(p_open_d)
    logit_open_l = safe_logit(p_open_l)
    
    logit_w = safe_logit(p_w)
    logit_d = safe_logit(p_d)
    logit_l = safe_logit(p_l)
    
    # Drift Estocástico (mu)
    df['gbm_mu_win'] = logit_w - logit_open_w
    df['gbm_mu_draw'] = logit_d - logit_open_d
    df['gbm_mu_loss'] = logit_l - logit_open_l
    
    # Volatilidad (sigma) - Ventana móvil de 10 partidos por equipo
    logger.info("Calculando volatilidad (sigma) y desviaciones con ventana móvil de 10 partidos...")
    df = df.sort_values(by=['team', 'match_date'])
    
    for outcome in ['win', 'draw', 'loss']:
        mu_col = f'gbm_mu_{outcome}'
        sigma_col = f'gbm_sigma_{outcome}'
        mean_drift_col = f'gbm_mean_{outcome}'
        z_col = f'gbm_z_{outcome}'
        
        # Volatilidad (Desviación estándar de los últimos 10 drifts del equipo)
        df[sigma_col] = df.groupby('team')[mu_col].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())
        # Deriva media histórica
        df[mean_drift_col] = df.groupby('team')[mu_col].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
        
        # Llenar NaNs iniciales (cuando no hay suficientes partidos) con la media/std global de la liga
        global_sigma = df[mu_col].std()
        df[sigma_col] = df[sigma_col].fillna(global_sigma)
        
        global_mean = df[mu_col].mean()
        df[mean_drift_col] = df[mean_drift_col].fillna(global_mean)
        
        # Z-Score de Sobre-reacción (Desviación de la Trayectoria Teórica)
        # Z = (Drift_Actual - Media_Drift) / Sigma
        df[z_col] = (df[mu_col] - df[mean_drift_col]) / df[sigma_col].replace(0, 1e-5)
    
    # Restablecer el orden exacto original por su índice
    df = df.sort_index()
    
    return df

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5)
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

def train_gbm_model():
    df = get_base_dataset()
    df_gbm = compute_gbm_features(df)
    
    if df_gbm is None:
        logger.error("Abortando entrenamiento GBM por falta de datos.")
        return
        
    split_idx = get_train_test_split(df_gbm)
    
    feature_cols = [
        'gbm_mu_win', 'gbm_mu_draw', 'gbm_mu_loss',
        'gbm_sigma_win', 'gbm_sigma_draw', 'gbm_sigma_loss',
        'gbm_z_win', 'gbm_z_draw', 'gbm_z_loss'
    ]
    
    # Se añaden cuotas implícitas base para que el modelo sepa el nivel de probabilidad donde ocurre el GBM
    if 'odds_win' in df_gbm.columns:
        inv_w = 1 / df_gbm['odds_win']
        inv_d = 1 / df_gbm['odds_draw']
        inv_l = 1 / df_gbm['odds_loss']
        vig = inv_w + inv_d + inv_l
        df_gbm['gbm_base_prob_win'] = inv_w / vig
        df_gbm['gbm_base_prob_draw'] = inv_d / vig
        df_gbm['gbm_base_prob_loss'] = inv_l / vig
        feature_cols.extend(['gbm_base_prob_win', 'gbm_base_prob_draw', 'gbm_base_prob_loss'])
    
    X = df_gbm[feature_cols].fillna(0).copy()
    y = df_gbm['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df_gbm.columns:
        train_dates = pd.to_datetime(df_gbm['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    logger.info("Optimizando Modelo Cuantitativo GBM con Optuna (20 Trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, train_dates, cv_strategy), n_trials=20)
    
    logger.info(f"Mejores parámetros XGBoost GBM: {study.best_params}")
    
    xgb_best = XGBClassifier(
        **study.best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    
    logger.info("Calculando predicciones OOF para Train (GBM)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. KFold para el primer fold para evitar NaNs en OOF
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
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

    # 2. Expanding Windows estándar
    for i, (train_idx, val_idx) in enumerate(splits):
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
        
    logger.info("Entrenando Modelo GBM final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    base_final = XGBClassifier(**xgb_best.get_params())
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='isotonic', cv=3)
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
        
        real_loss = np.mean(y_true_valid == 0)
        real_draw = np.mean(y_true_valid == 1)
        real_win = np.mean(y_true_valid == 2)
        
        pred_loss = np.mean(preds_valid[:, 0])
        pred_draw = np.mean(preds_valid[:, 1])
        pred_win = np.mean(preds_valid[:, 2])
        
        logger.info("=== AUDITORÍA Y RESULTADOS DEL MODELO GBM QUANT ===")
        logger.info(f" -> Log Loss Global (OOF): {logloss_val:.4f}")
        logger.info(f" - Derrota (Loss) | Pred: {pred_loss*100:.1f}% | Real: {real_loss*100:.1f}% | Brier: {brier_loss:.4f}")
        logger.info(f" - Empate (Draw)  | Pred: {pred_draw*100:.1f}% | Real: {real_draw*100:.1f}% | Brier: {brier_draw:.4f}")
        logger.info(f" - Victoria (Win) | Pred: {pred_win*100:.1f}% | Real: {real_win*100:.1f}% | Brier: {brier_win:.4f}")
        
        importances = np.mean([clf.estimator.feature_importances_ for clf in final_estimator.calibrated_classifiers_], axis=0)
        feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        logger.info("=== TOP 5 FEATURES GBM MAS IMPORTANTES ===")
        for _, row in feat_imp.head(5).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

    # Guardar Resultados OOF
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
