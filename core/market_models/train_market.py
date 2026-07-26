import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# Asegurar import de data_splitter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = True
OPTUNA_TRIALS = 20
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_market.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_market')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'market_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def compute_shin_probs(odds_win, odds_draw, odds_loss):
    """
    Calcula probabilidades des-marginadas usando el método de Shin (1992, 1993).
    Ajusta el Favorite-Longshot Bias modelando la proporción 'z' de insider trading / ruido.
    Vectorizado y numéricamente estable.
    """
    ow = np.maximum(np.asarray(odds_win, dtype=float), 1.001)
    od = np.maximum(np.asarray(odds_draw, dtype=float), 1.001)
    ol = np.maximum(np.asarray(odds_loss, dtype=float), 1.001)
    
    pi = np.column_stack([1.0 / ow, 1.0 / od, 1.0 / ol])  # (N, 3)
    S = pi.sum(axis=1, keepdims=True)                       # Overround total (N, 1)
    
    z = np.clip((S - 1.0) / np.maximum(S, 1.0), 0.0, 0.45)
    
    for _ in range(6):
        num = np.sqrt(z**2 + 4.0 * (1.0 - z) * (pi**2 / S)) - z
        den = 2.0 * np.maximum(1.0 - z, 1e-6)
        p = num / den
        
        diff = p.sum(axis=1, keepdims=True) - 1.0
        if np.max(np.abs(diff)) < 1e-5:
            break
        dp_dz = -0.5 / den + (2.0 * z - 4.0 * (pi**2 / S)) / (4.0 * den * np.maximum(np.sqrt(z**2 + 4.0 * (1.0 - z) * (pi**2 / S)), 1e-6)) + num / (2.0 * den**2)
        f_prime = dp_dz.sum(axis=1, keepdims=True)
        f_prime = np.where(np.abs(f_prime) < 1e-6, 1.0, f_prime)
        z = np.clip(z - diff / f_prime, 0.0, 0.499)
        
    num = np.sqrt(z**2 + 4.0 * (1.0 - z) * (pi**2 / S)) - z
    den = 2.0 * np.maximum(1.0 - z, 1e-6)
    p = num / den
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)
    return p[:, 0], p[:, 1], p[:, 2], z.squeeze()

def compute_power_probs(odds_win, odds_draw, odds_loss):
    """
    Calcula probabilidades des-marginadas usando el Power Method (Štrumbelj 2014).
    Encuentra exponente k tal que sum( (1/odds_i)^k ) = 1.
    Excelente para neutralizar sesgos en probabilidades extremas.
    """
    ow = np.maximum(np.asarray(odds_win, dtype=float), 1.001)
    od = np.maximum(np.asarray(odds_draw, dtype=float), 1.001)
    ol = np.maximum(np.asarray(odds_loss, dtype=float), 1.001)
    
    pi = np.column_stack([1.0 / ow, 1.0 / od, 1.0 / ol])
    S = pi.sum(axis=1, keepdims=True)
    k = 1.0 / np.maximum(S, 1.0)
    
    for _ in range(6):
        pik = np.power(pi, k)
        sum_pik = pik.sum(axis=1, keepdims=True)
        diff = sum_pik - 1.0
        if np.max(np.abs(diff)) < 1e-5:
            break
        f_prime = (pik * np.log(np.maximum(pi, 1e-9))).sum(axis=1, keepdims=True)
        f_prime = np.where(np.abs(f_prime) < 1e-6, 1.0, f_prime)
        k = k - diff / f_prime
        
    pik = np.power(pi, k)
    p = pik / np.maximum(pik.sum(axis=1, keepdims=True), 1e-9)
    return p[:, 0], p[:, 1], p[:, 2]

def get_time_weights(dates, half_life_days=365):
    if dates is None or len(dates) == 0:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def objective(trial, X_train, y_train, dates_train, cv_strategy):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 60, 250),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    
    cv_scores = []
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        dates_tr = dates_train.iloc[train_idx] if dates_train is not None else None
        w_tr = get_time_weights(dates_tr)
        
        xgb_eval = XGBClassifier(**param)
        calibrated_eval = CalibratedClassifierCV(estimator=xgb_eval, method='sigmoid', cv=3)
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
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # ==============================================================================
    # FEATURE ENGINEERING AVANZADO DE MERCADO (11/10)
    # ==============================================================================
    # 1. Remoción de Margen Proporcional, Shin y Power para Cierre
    if 'odds_win' in df.columns and 'prob_win_implied' not in df.columns:
        cw_win = df['odds_win']
        cw_draw = df['odds_draw']
        cw_loss = df['odds_loss']
        
        sum_implied_c = (1 / cw_win) + (1 / cw_draw) + (1 / cw_loss)
        df['vig_close'] = sum_implied_c - 1.0
        
        df['prob_win_implied'] = (1 / cw_win) / sum_implied_c
        df['prob_draw_implied'] = (1 / cw_draw) / sum_implied_c
        df['prob_loss_implied'] = (1 / cw_loss) / sum_implied_c
        
        shin_w, shin_d, shin_l, shin_z = compute_shin_probs(cw_win, cw_draw, cw_loss)
        df['shin_prob_win'] = shin_w
        df['shin_prob_draw'] = shin_d
        df['shin_prob_loss'] = shin_l
        df['shin_z_param'] = shin_z
        
        pow_w, pow_d, pow_l = compute_power_probs(cw_win, cw_draw, cw_loss)
        df['power_prob_win'] = pow_w
        df['power_prob_draw'] = pow_d
        df['power_prob_loss'] = pow_l

    # 2. Imputación inteligente de Apertura (fallback a cierre si falta apertura)
    if 'open_odds_win' in df.columns:
        ow_win = df['open_odds_win'].fillna(df['odds_win'])
        ow_draw = df['open_odds_draw'].fillna(df['odds_draw'])
        ow_loss = df['open_odds_loss'].fillna(df['odds_loss'])
    else:
        ow_win, ow_draw, ow_loss = df['odds_win'], df['odds_draw'], df['odds_loss']
        
    sum_implied_o = (1 / ow_win) + (1 / ow_draw) + (1 / ow_loss)
    df['vig_open'] = sum_implied_o - 1.0
    
    df['open_prob_win'] = (1 / ow_win) / sum_implied_o
    df['open_prob_draw'] = (1 / ow_draw) / sum_implied_o
    df['open_prob_loss'] = (1 / ow_loss) / sum_implied_o

    # 3. Features de Steam y Movimiento de Líneas
    df['steam_win'] = df['prob_win_implied'] - df['open_prob_win']
    df['steam_draw'] = df['prob_draw_implied'] - df['open_prob_draw']
    df['steam_loss'] = df['prob_loss_implied'] - df['open_prob_loss']
    
    # Log Steam Ratio
    df['log_steam_win'] = np.log(np.maximum(df['prob_win_implied'], 1e-4) / np.maximum(df['open_prob_win'], 1e-4))
    df['log_steam_draw'] = np.log(np.maximum(df['prob_draw_implied'], 1e-4) / np.maximum(df['open_prob_draw'], 1e-4))
    df['log_steam_loss'] = np.log(np.maximum(df['prob_loss_implied'], 1e-4) / np.maximum(df['open_prob_loss'], 1e-4))
    
    # Steam Relativo (%)
    df['rel_steam_win'] = df['steam_win'] / np.maximum(df['open_prob_win'], 1e-4)
    df['rel_steam_draw'] = df['steam_draw'] / np.maximum(df['open_prob_draw'], 1e-4)
    df['rel_steam_loss'] = df['steam_loss'] / np.maximum(df['open_prob_loss'], 1e-4)

    # 4. Entropía de Mercado y Microestructura
    p_close = df[['prob_win_implied', 'prob_draw_implied', 'prob_loss_implied']].values
    p_open = df[['open_prob_win', 'open_prob_draw', 'open_prob_loss']].values
    
    df['market_entropy'] = -np.sum(p_close * np.log(np.maximum(p_close, 1e-9)), axis=1)
    df['open_market_entropy'] = -np.sum(p_open * np.log(np.maximum(p_open, 1e-9)), axis=1)
    df['entropy_change'] = df['market_entropy'] - df['open_market_entropy']
    
    # Brecha de Favoritismo y Liquidez
    sorted_probs = np.sort(p_close, axis=1)
    df['fav_prob_gap'] = sorted_probs[:, 2] - sorted_probs[:, 1]
    df['overround_drop'] = df['vig_open'] - df['vig_close']
        
    feature_cols = [
        'open_prob_win', 'open_prob_draw', 'open_prob_loss',
        'prob_win_implied', 'prob_draw_implied', 'prob_loss_implied',
        'shin_prob_win', 'shin_prob_draw', 'shin_prob_loss', 'shin_z_param',
        'power_prob_win', 'power_prob_draw', 'power_prob_loss',
        'steam_win', 'steam_draw', 'steam_loss',
        'log_steam_win', 'log_steam_draw', 'log_steam_loss',
        'rel_steam_win', 'rel_steam_draw', 'rel_steam_loss',
        'vig_open', 'vig_close', 'overround_drop',
        'market_entropy', 'entropy_change', 'fav_prob_gap'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan las siguientes columnas en Market: {missing_cols}. Usando las disponibles.")
        feature_cols = [c for c in feature_cols if c in df.columns]
        
    if not feature_cols:
        logger.error("No hay variables de mercado disponibles. Abortando train_market.")
        return

    # Imputación con media/mediana en lugar de fillna(0)
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
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
        tree_method='hist',
        device='cuda'
    )
    
    logger.info("Calculando predicciones OOF para Train (Market) sin Data Leakage...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando Sub-splits Temporales (Sin KFold Shuffle)
    first_train_idx = splits[0][0]
    n_first = len(first_train_idx)
    logger.info(f"  -> Procesando Primer Fold Inicial ({n_first} muestras) con Sub-splits Temporales Puros...")
    
    n_sub = 4
    sub_size = n_first // n_sub
    for sub in range(1, n_sub):
        sub_tr_idx = first_train_idx[:sub * sub_size]
        sub_val_idx = first_train_idx[sub * sub_size : (sub + 1) * sub_size] if sub < n_sub - 1 else first_train_idx[sub * sub_size:]
        
        X_sub_tr, y_sub_tr = X_train.iloc[sub_tr_idx], y_train.iloc[sub_tr_idx]
        X_sub_val = X_train.iloc[sub_val_idx]
        
        dates_sub_tr = train_dates.iloc[sub_tr_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_sub_tr)
        
        base_sub = XGBClassifier(**xgb_best.get_params())
        sub_estimator = CalibratedClassifierCV(estimator=base_sub, method='sigmoid', cv=3)
        if w_tr is not None:
            sub_estimator.fit(X_sub_tr, y_sub_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            sub_estimator.fit(X_sub_tr, y_sub_tr)
            
        pred_probs_train[sub_val_idx] = sub_estimator.predict_proba(X_sub_val)
        
    half_sub = max(100, sub_size // 2)
    X_h_tr, y_h_tr = X_train.iloc[:half_sub], y_train.iloc[:half_sub]
    w_h_tr = get_time_weights(train_dates.iloc[:half_sub]) if train_dates is not None else None
    base_h = XGBClassifier(**xgb_best.get_params())
    h_estimator = CalibratedClassifierCV(estimator=base_h, method='sigmoid', cv=3)
    if w_h_tr is not None:
        h_estimator.fit(X_h_tr, y_h_tr, sample_weight=w_h_tr.values if isinstance(w_h_tr, pd.Series) else w_h_tr)
    else:
        h_estimator.fit(X_h_tr, y_h_tr)
    pred_probs_train[first_train_idx[:sub_size]] = h_estimator.predict_proba(X_train.iloc[first_train_idx[:sub_size]])

    # 2. Expanding Windows estándar para el resto
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        base_fold = XGBClassifier(**xgb_best.get_params())
        fold_estimator = CalibratedClassifierCV(estimator=base_fold, method='sigmoid', cv=3)
        if w_tr is not None:
            fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        else:
            fold_estimator.fit(X_tr, y_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo de Mercado final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    base_final = XGBClassifier(**xgb_best.get_params())
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
        
        logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO MERCADO (OPTIMIZADO 11/10) ===")
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
