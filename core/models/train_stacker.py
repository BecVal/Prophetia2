import os

import json

RUN_OPTUNA = False
OPTUNA_TRIALS = 20
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger


OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_stacker.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
logger = get_logger(__name__, 'train_stacker')

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH_FUND = os.path.join(MODEL_SAVE_DIR, 'stacker_fundamental_model.pkl')
MODEL_SAVE_PATH_FINAL = os.path.join(MODEL_SAVE_DIR, 'stacker_final_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None or dates.isna().all():
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

def load_oof(model_name, split_idx):
    train_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_train.parquet')
    test_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_test.parquet')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_parquet(train_path, engine='fastparquet'), pd.read_parquet(test_path, engine='fastparquet')
    else:
        logger.warning(f"OOF para {model_name} no encontrado. Se omitirá.")
        return None, None
        
def compute_meta_features(X_base, df_orig, prefix=""):
    # X_base contiene las probabilidades de los modelos
    meta = pd.DataFrame(index=X_base.index)
    
    # 1. Varianza de los modelos
    cols_loss = [c for c in X_base.columns if 'loss' in c.lower()]
    cols_draw = [c for c in X_base.columns if 'draw' in c.lower()]
    cols_win = [c for c in X_base.columns if 'win' in c.lower()]
    
    if cols_loss: meta[f'{prefix}std_loss'] = X_base[cols_loss].std(axis=1).fillna(0)
    if cols_draw: meta[f'{prefix}std_draw'] = X_base[cols_draw].std(axis=1).fillna(0)
    if cols_win: meta[f'{prefix}std_win'] = X_base[cols_win].std(axis=1).fillna(0)
    
    # 2. Entropía Media del consenso (Para saber qué tan seguro está el consenso general)
    mean_probs = pd.DataFrame({
        'loss': X_base[cols_loss].mean(axis=1) if cols_loss else 0,
        'draw': X_base[cols_draw].mean(axis=1) if cols_draw else 0,
        'win': X_base[cols_win].mean(axis=1) if cols_win else 1,
    })
    
    # Normalizar para entropía (por si no suman 1 perfectamente)
    sums = mean_probs.sum(axis=1)
    mean_probs = mean_probs.div(np.where(sums > 0, sums, 1), axis=0)
    
    def calc_entropy(row):
        return entropy(row + 1e-9)
        
    meta[f'{prefix}entropy'] = mean_probs.apply(calc_entropy, axis=1)
    
    # 3. Cuotas Implícitas (Contexto de Mercado)
    if 'open_odds_win' in df_orig.columns:
        meta['implied_open_loss'] = (1 / df_orig['open_odds_loss'].clip(lower=1.01)).fillna(0)
        meta['implied_open_draw'] = (1 / df_orig['open_odds_draw'].clip(lower=1.01)).fillna(0)
        meta['implied_open_win'] = (1 / df_orig['open_odds_win'].clip(lower=1.01)).fillna(0)
    
    # 4. ID de Competición (Para reducir la ceguera)
    if 'competition' in df_orig.columns:
        meta['competition_id'] = pd.factorize(df_orig['competition'])[0]
    else:
        meta['competition_id'] = 0
        
    return pd.concat([X_base, meta], axis=1)

def train_stacker():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    df_train_orig = df.iloc[:split_idx].copy()
    df_test_orig = df.iloc[split_idx:].copy()
    
    train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx]) if 'match_date' in df.columns else None
    
    logger.info("Cargando predicciones OOF de los modelos base...")
    
    oof_data = {}
    for mod in ['quant', 'poisson', 'context', 'nn', 'draws', 'market', 'gbm']:
        tr, ts = load_oof(mod, split_idx)
        if tr is not None:
            oof_data[mod] = (tr, ts)
            
    # ====== ETAPA 1: FUNDAMENTAL STACKER ======
    fundamental_train_list = []
    fundamental_test_list = []
    
    if 'quant' in oof_data:
        logger.info("Modelo 'quant' encontrado. Se usará como modelo base fundamental.")
        base_models = ['quant', 'context', 'nn', 'draws']
    else:
        logger.info("Modelo 'quant' no encontrado. Se usará 'poisson' como plan B.")
        base_models = ['poisson', 'context', 'nn', 'draws']
        
    for mod in base_models:
        if mod in oof_data:
            fundamental_train_list.append(oof_data[mod][0])
            fundamental_test_list.append(oof_data[mod][1])
            
    if not fundamental_train_list:
        raise ValueError("No se encontraron modelos fundamentales.")
        
    X_train_fund = pd.concat(fundamental_train_list, axis=1).fillna(0)
    X_test_fund = pd.concat(fundamental_test_list, axis=1).fillna(0)
    
    logger.info("Entrenando Stacker Fundamental (Nivel 1)...")
    
    # Eliminamos Isotonic Calibration. Usamos regresión logística simple con L2 moderada (para evitar curvas dentadas)
    fund_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    
    w_train = get_time_weights(train_dates) if train_dates is not None else None
    
    if w_train is not None:
        fund_pipeline.fit(X_train_fund, y_train, lr__sample_weight=w_train)
    else:
        fund_pipeline.fit(X_train_fund, y_train)
        
    # Obtener predicciones fundamentales para pasar al Nivel 2
    fund_prob_train = fund_pipeline.predict_proba(X_train_fund)
    fund_prob_test = fund_pipeline.predict_proba(X_test_fund)
    
    df_fund_train = pd.DataFrame(fund_prob_train, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_train_fund.index)
    df_fund_test = pd.DataFrame(fund_prob_test, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_test_fund.index)
    
    joblib.dump({'model': fund_pipeline, 'features': X_train_fund.columns.tolist()}, MODEL_SAVE_PATH_FUND)
    
    # ====== ETAPA 2: FINAL (MARKET) STACKER con Optuna + Meta-Features ======
    logger.info("Entrenando Stacker Final (Nivel 2) combinando Fundamentales + Mercado + GBM...")
    
    market_models = []
    if 'market' in oof_data:
        market_models.append(oof_data['market'])
    if 'gbm' in oof_data:
        market_models.append(oof_data['gbm'])
        
    if market_models:
        mkt_train = pd.concat([m[0] for m in market_models], axis=1)
        mkt_test = pd.concat([m[1] for m in market_models], axis=1)
        X_train_meta = pd.concat([df_fund_train, mkt_train], axis=1).fillna(0)
        X_test_meta = pd.concat([df_fund_test, mkt_test], axis=1).fillna(0)
    else:
        logger.warning("No se encontró OOF de Market. El Stacker Final será igual al Fundamental, pero mejorado con árboles.")
        X_train_meta = df_fund_train.copy()
        X_test_meta = df_fund_test.copy()
        
    # Inyectar Meta-Features (Ceguera curada)
    X_train_meta = compute_meta_features(X_train_meta, df_train_orig, prefix="meta_")
    X_test_meta = compute_meta_features(X_test_meta, df_test_orig, prefix="meta_")
    
    # Determinar columnas categóricas (la competición) para HistGradientBoosting
    cat_features = [i for i, col in enumerate(X_train_meta.columns) if col == 'competition_id']
    if not cat_features:
        cat_features = None
    
    # Optuna para el Nivel 2
    logger.info("Iniciando optimización con Optuna para el Nivel 2 (HistGradientBoosting)...")
    
    # Creamos un pequeño split temporal de validación para Optuna dentro del Train set (20% más reciente)
    opt_split = int(len(X_train_meta) * 0.8)
    X_opt_train, y_opt_train = X_train_meta.iloc[:opt_split], y_train.iloc[:opt_split]
    X_opt_val, y_opt_val = X_train_meta.iloc[opt_split:], y_train.iloc[opt_split:]
    
    w_opt_train = w_train[:opt_split] if w_train is not None else None
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_iter': trial.suggest_int('max_iter', 50, 300),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
            'categorical_features': cat_features,
            'random_state': 42
        }
        
        model = HistGradientBoostingClassifier(**params)
        
        # HistGradientBoosting no acepta sample_weight en fit en algunas versiones via pipeline,
        # pero sí directamente. Usaremos fit directo.
        try:
            model.fit(X_opt_train, y_opt_train, sample_weight=w_opt_train)
        except TypeError:
            # Por si la versión de sklearn no soporta sample_weight aquí
            model.fit(X_opt_train, y_opt_train)
            
        preds = model.predict_proba(X_opt_val)
        return log_loss(y_opt_val, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if RUN_OPTUNA:
        logger.info(f"Iniciando optimización con Optuna ({OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800)
        best_params_optuna = study.best_params
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params_optuna, f, indent=4)
        best_params = best_params_optuna.copy()
    else:
        logger.info("Cargando mejores parámetros de Optuna guardados...")
        if os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
        else:
            logger.warning(f"Archivo de parámetros {OPTUNA_PARAMS_FILE} no encontrado. Ejecutando Optuna como fallback.")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=1800)
            best_params_optuna = study.best_params
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params_optuna, f, indent=4)
            best_params = best_params_optuna.copy()
            
    best_params['categorical_features'] = cat_features
    best_params['random_state'] = 42
    logger.info(f"Mejores parámetros encontrados para Nivel 2: {best_params}")
    
    # Entrenar el modelo final de árboles con todos los datos de Train
    base_final_model = HistGradientBoostingClassifier(**best_params)
    tscv_final = get_cv_strategy(n_splits=5)
    final_model = CalibratedClassifierCV(estimator=base_final_model, method='isotonic', cv=tscv_final)
    try:
        if w_train is not None:
            final_model.fit(X_train_meta, y_train, sample_weight=w_train)
        else:
            final_model.fit(X_train_meta, y_train)
    except TypeError:
        final_model.fit(X_train_meta, y_train)
        
    # Evaluación OOF para evitar Data Leakage (In-Sample) y Look-Ahead Bias
    from sklearn.model_selection import TimeSeriesSplit, KFold
    y_prob_train = np.zeros((len(X_train_meta), 3))
    y_prob_train[:] = np.nan
    
    cv = TimeSeriesSplit(n_splits=5)
    splits = list(cv.split(X_train_meta))
    
    # 1. Resolver el primer bloque usando K-Fold para tener OOF
    first_idx = splits[0][0]
    X_first, y_first = X_train_meta.iloc[first_idx], y_train.iloc[first_idx]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_kf, va_kf in kf.split(X_first):
        X_tr, y_tr = X_first.iloc[tr_kf], y_first.iloc[tr_kf]
        X_va = X_first.iloc[va_kf]
        
        if w_train is not None:
            if isinstance(w_train, pd.Series) or isinstance(w_train, np.ndarray):
                w_tr = w_train.iloc[first_idx[tr_kf]] if isinstance(w_train, pd.Series) else w_train[first_idx[tr_kf]]
            else:
                w_tr = None
        else:
            w_tr = None
            
        m_base = HistGradientBoostingClassifier(**best_params)
        tscv_m = get_cv_strategy(n_splits=3)
        m = CalibratedClassifierCV(estimator=m_base, method='isotonic', cv=tscv_m)
        try:
            m.fit(X_tr, y_tr, sample_weight=w_tr) if w_tr is not None else m.fit(X_tr, y_tr)
        except TypeError:
            m.fit(X_tr, y_tr)
            
        y_prob_train[first_idx[va_kf]] = m.predict_proba(X_va)
        
    # 2. Resolver los demás bloques respetando la flecha del tiempo
    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_train_meta.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va = X_train_meta.iloc[va_idx]
        
        if w_train is not None:
            if isinstance(w_train, pd.Series) or isinstance(w_train, np.ndarray):
                w_tr = w_train.iloc[tr_idx] if isinstance(w_train, pd.Series) else w_train[tr_idx]
            else:
                w_tr = None
        else:
            w_tr = None
            
        m_base = HistGradientBoostingClassifier(**best_params)
        tscv_m = get_cv_strategy(n_splits=3)
        m = CalibratedClassifierCV(estimator=m_base, method='isotonic', cv=tscv_m)
        try:
            m.fit(X_tr, y_tr, sample_weight=w_tr) if w_tr is not None else m.fit(X_tr, y_tr)
        except TypeError:
            m.fit(X_tr, y_tr)
            
        y_prob_train[va_idx] = m.predict_proba(X_va)
        
    y_prob_train = y_prob_train / y_prob_train.sum(axis=1, keepdims=True)
    
    y_prob_test = final_model.predict_proba(X_test_meta)
    y_prob_test = y_prob_test / y_prob_test.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob_test)
    
    logger.info("=== RESULTADOS META-MODELO FINAL (HGB + META-FEATURES) ===")
    logger.info(f"Accuracy Global: {acc:.4f}")
    logger.info(f"Log-Loss: {loss:.4f}")
    
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
    y_test_oh = np.zeros_like(y_prob_test)
    y_test_oh[np.arange(len(y_test_arr)), y_test_arr] = 1
    
    brier_loss = brier_score_loss(y_test_oh[:, 0], y_prob_test[:, 0])
    brier_draw = brier_score_loss(y_test_oh[:, 1], y_prob_test[:, 1])
    brier_win = brier_score_loss(y_test_oh[:, 2], y_prob_test[:, 2])
    brier_global = np.mean([brier_loss, brier_draw, brier_win])
    
    logger.info(f"Brier Score Global (Promedio de Clases): {brier_global:.4f}")
    logger.info(f"Brier Score por Clase -> Loss: {brier_loss:.4f} | Draw: {brier_draw:.4f} | Win: {brier_win:.4f}")
    
    # LOGS: Verificacion de calibracion (Auditoría)
    logger.info("=== AUDITORÍA ESTADÍSTICA DEL ENSAMBLE FINAL ===")
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = y_prob_train[:, 0].mean()
    pred_draw = y_prob_train[:, 1].mean()
    pred_win = y_prob_train[:, 2].mean()
    
    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Train Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Train Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Train Dataset: {real_win*100:.1f}%")
    
    # Guardar resultados y datasets para el modelo CLV y el Simulador
    # Usar df_orig asegura que no tengamos variables meta en el simulador
    df_train_save = pd.DataFrame({
        'match_date': df_train_orig['match_date'].values if 'match_date' in df.columns else np.array([None]*len(y_train)),
        'competition': df_train_orig['competition'].values if 'competition' in df.columns else np.array([None]*len(y_train)),
        'team': df_train_orig['team'].values,
        'opponent': df_train_orig['opponent'].values,
        'is_home': df_train_orig['is_home'].values,
        'prob_loss': y_prob_train[:, 0],
        'prob_draw': y_prob_train[:, 1],
        'prob_win': y_prob_train[:, 2],
        'outcome': y_train.values,
        'odds_win': df_train_orig['open_odds_win'].values if 'open_odds_win' in df.columns else df_train_orig['odds_win'].values,
        'odds_draw': df_train_orig['open_odds_draw'].values if 'open_odds_draw' in df.columns else df_train_orig['odds_draw'].values,
        'odds_loss': df_train_orig['open_odds_loss'].values if 'open_odds_loss' in df.columns else df_train_orig['odds_loss'].values,
        'closing_odds_win': df_train_orig['odds_win'].values,
        'closing_odds_draw': df_train_orig['odds_draw'].values,
        'closing_odds_loss': df_train_orig['odds_loss'].values
    })
    
    has_odds = 'odds_win' in df.columns
    if has_odds:
        df_test_save = pd.DataFrame({
            'match_date': df_test_orig['match_date'].values if 'match_date' in df.columns else np.array([None]*len(y_test)),
            'competition': df_test_orig['competition'].values if 'competition' in df.columns else np.array([None]*len(y_test)),
            'team': df_test_orig['team'].values,
            'opponent': df_test_orig['opponent'].values,
            'is_home': df_test_orig['is_home'].values,
            'prob_loss': y_prob_test[:, 0],
            'prob_draw': y_prob_test[:, 1],
            'prob_win': y_prob_test[:, 2],
            'outcome': y_test.values,
            'odds_win': df_test_orig['open_odds_win'].values if 'open_odds_win' in df.columns else df_test_orig['odds_win'].values,
            'odds_draw': df_test_orig['open_odds_draw'].values if 'open_odds_draw' in df.columns else df_test_orig['odds_draw'].values,
            'odds_loss': df_test_orig['open_odds_loss'].values if 'open_odds_loss' in df.columns else df_test_orig['odds_loss'].values,
            'closing_odds_win': df_test_orig['odds_win'].values,
            'closing_odds_draw': df_test_orig['odds_draw'].values,
            'closing_odds_loss': df_test_orig['odds_loss'].values
        })
        
        df_test_save.to_parquet(os.path.join(PROCESSED_DIR, 'test_predictions.parquet'), engine='fastparquet')
        logger.info("Guardado test_predictions.parquet")
    
    df_train_save.to_parquet(os.path.join(PROCESSED_DIR, 'train_predictions.parquet'), engine='fastparquet')
    
    # Pasar variables fair si existen
    if 'fair_loss' in df.columns:
        for col in ['fair_loss', 'fair_draw', 'fair_win']:
            X_train_meta[col] = df_train_orig[col].values
            X_test_meta[col] = df_test_orig[col].values
            
    if 'open_fair_loss' in df.columns:
        for col in ['open_fair_loss', 'open_fair_draw', 'open_fair_win']:
            X_train_meta[col] = df_train_orig[col].values
            X_test_meta[col] = df_test_orig[col].values

    X_train_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_train.parquet'), engine='fastparquet')
    X_test_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_test.parquet'), engine='fastparquet')
    
    joblib.dump({'model': final_model, 'features': X_train_meta.columns.tolist()}, MODEL_SAVE_PATH_FINAL)
    logger.info(f"Modelo Stacker Final (HGB) guardado en {MODEL_SAVE_PATH_FINAL}")

if __name__ == "__main__":
    train_stacker()
