import os
import json
import sys
import pandas as pd
import numpy as np
import joblib
import optuna
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger



# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
# Cambia RUN_OPTUNA a True si deseas volver a buscar los mejores hiperparámetros.
# De lo contrario (False), cargará los mejores guardados en el archivo JSON.
RUN_OPTUNA = False
OPTUNA_TRIALS = 40
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_stacker.json'))  # <-- NUEVA RUTA: models_best_parameters
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

def load_oof(model_name, split_idx, train_len=None, test_len=None):
    train_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_train.parquet')
    test_path = os.path.join(PROCESSED_DIR, f'oof_{model_name}_test.parquet')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        tr = pd.read_parquet(train_path, engine='fastparquet')
        ts = pd.read_parquet(test_path, engine='fastparquet')
        
        if 'lambda_total' in tr.columns:
            tr = tr.rename(columns={'lambda_total': f'lambda_total_{model_name}'})
            ts = ts.rename(columns={'lambda_total': f'lambda_total_{model_name}'})
            
        # Resetear índices para alineación posicional limpia
        tr = tr.reset_index(drop=True).fillna(0.0)
        ts = ts.reset_index(drop=True).fillna(0.0)
        
        # Si las dimensiones difieren por filtrados previos, reindexar a la longitud esperada
        if train_len is not None and len(tr) != train_len:
            tr = tr.reindex(range(train_len)).fillna(0.0)
        if test_len is not None and len(ts) != test_len:
            ts = ts.reindex(range(test_len)).fillna(0.0)
            
        return tr, ts
    else:
        logger.warning(f"OOF para {model_name} no encontrado. Se omitirá.")
        return None, None
        
def compute_meta_features(X_base, df_orig, prefix=""):
    # X_base contiene las probabilidades de los modelos con RangeIndex
    X_base = X_base.reset_index(drop=True).fillna(0.0)
    meta = pd.DataFrame(index=X_base.index)
    
    # 1. Varianza de los modelos
    cols_loss = [c for c in X_base.columns if 'loss' in c.lower()]
    cols_draw = [c for c in X_base.columns if 'draw' in c.lower()]
    cols_win = [c for c in X_base.columns if 'win' in c.lower()]
    
    if cols_loss: meta[f'{prefix}std_loss'] = X_base[cols_loss].std(axis=1).fillna(0.0)
    if cols_draw: meta[f'{prefix}std_draw'] = X_base[cols_draw].std(axis=1).fillna(0.0)
    if cols_win: meta[f'{prefix}std_win'] = X_base[cols_win].std(axis=1).fillna(0.0)
    
    # 2. Entropía Media del consenso (Vectorizada para máxima eficiencia)
    mean_probs = pd.DataFrame({
        'loss': X_base[cols_loss].mean(axis=1) if cols_loss else 0.0,
        'draw': X_base[cols_draw].mean(axis=1) if cols_draw else 0.0,
        'win': X_base[cols_win].mean(axis=1) if cols_win else 1.0,
    }, index=X_base.index)
    
    sums = mean_probs.sum(axis=1).values[:, None]
    p = mean_probs.values / np.where(sums > 0, sums, 1.0)
    p = np.clip(p, 1e-9, 1.0)
    meta[f'{prefix}entropy'] = -np.sum(p * np.log(p), axis=1)
    
    # 3. Cuotas Implícitas (Contexto de Mercado extraído mediante .values)
    if 'open_odds_win' in df_orig.columns:
        o_loss = df_orig['open_odds_loss'].fillna(df_orig.get('odds_loss', 0.0)).clip(lower=1.01).values
        o_draw = df_orig['open_odds_draw'].fillna(df_orig.get('odds_draw', 0.0)).clip(lower=1.01).values
        o_win = df_orig['open_odds_win'].fillna(df_orig.get('odds_win', 0.0)).clip(lower=1.01).values
        meta['implied_open_loss'] = np.nan_to_num(1.0 / o_loss, nan=0.0)
        meta['implied_open_draw'] = np.nan_to_num(1.0 / o_draw, nan=0.0)
        meta['implied_open_win'] = np.nan_to_num(1.0 / o_win, nan=0.0)
    elif 'odds_win' in df_orig.columns:
        o_loss = df_orig['odds_loss'].clip(lower=1.01).values
        o_draw = df_orig['odds_draw'].clip(lower=1.01).values
        o_win = df_orig['odds_win'].clip(lower=1.01).values
        meta['implied_open_loss'] = np.nan_to_num(1.0 / o_loss, nan=0.0)
        meta['implied_open_draw'] = np.nan_to_num(1.0 / o_draw, nan=0.0)
        meta['implied_open_win'] = np.nan_to_num(1.0 / o_win, nan=0.0)
    else:
        meta['implied_open_loss'] = 0.0
        meta['implied_open_draw'] = 0.0
        meta['implied_open_win'] = 0.0
    
    meta = meta.fillna(0.0)
    res = pd.concat([X_base, meta], axis=1).fillna(0.0)
    return res

def _causal_consensus_prior(df_probs):
    cols_l = [c for c in df_probs.columns if 'loss' in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower()]
    cols_d = [c for c in df_probs.columns if 'draw' in c.lower() and 'is_draw' not in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower()]
    cols_w = [c for c in df_probs.columns if 'win' in c.lower() and 'implied' not in c.lower() and 'std' not in c.lower()]
    p_l = df_probs[cols_l].mean(axis=1).values if cols_l else np.full(len(df_probs), 0.30)
    p_d = df_probs[cols_d].mean(axis=1).values if cols_d else np.full(len(df_probs), 0.25)
    p_w = df_probs[cols_w].mean(axis=1).values if cols_w else np.full(len(df_probs), 0.45)
    probs = np.column_stack([p_l, p_d, p_w])
    probs = np.clip(probs, 1e-6, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)

def train_stacker():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    
    train_len = len(y_train)
    test_len = len(y_test)
    
    df_train_orig = df.iloc[:split_idx].reset_index(drop=True)
    df_test_orig = df.iloc[split_idx:].reset_index(drop=True)
    
    train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx]).reset_index(drop=True) if 'match_date' in df.columns else None
    
    logger.info("Cargando predicciones OOF de los modelos 1X2 base y de mercado...")
    
    oof_data = {}
    for mod in ['quant', 'poisson', 'context', 'nn', 'draws', 'market', 'gbm']:
        tr, ts = load_oof(mod, split_idx, train_len=train_len, test_len=test_len)
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
        
    X_train_fund = pd.concat(fundamental_train_list, axis=1).reset_index(drop=True).fillna(0.0)
    X_test_fund = pd.concat(fundamental_test_list, axis=1).reset_index(drop=True).fillna(0.0)
    
    logger.info("Entrenando Stacker Fundamental (Nivel 1) con predicciones Out-Of-Fold (sin Data Leakage)...")
    
    w_train = get_time_weights(train_dates) if train_dates is not None else None
    
    # Generar predicciones Out-Of-Fold (OOF) temporales estrictas para el set de Train
    fund_prob_train = np.zeros((len(X_train_fund), 3))
    tscv_l1 = TimeSeriesSplit(n_splits=5)
    l1_splits = list(tscv_l1.split(X_train_fund))
    
    # Primer bloque temporal
    first_idx_l1 = l1_splits[0][0]
    sub_tscv = TimeSeriesSplit(n_splits=3)
    for sub_tr, sub_va in sub_tscv.split(X_train_fund.iloc[first_idx_l1]):
        m_sub = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ])
        sub_w = w_train.iloc[first_idx_l1[sub_tr]] if w_train is not None else None
        m_sub.fit(X_train_fund.iloc[first_idx_l1[sub_tr]], y_train.iloc[first_idx_l1[sub_tr]], lr__sample_weight=sub_w) if sub_w is not None else m_sub.fit(X_train_fund.iloc[first_idx_l1[sub_tr]], y_train.iloc[first_idx_l1[sub_tr]])
        fund_prob_train[first_idx_l1[sub_va]] = m_sub.predict_proba(X_train_fund.iloc[first_idx_l1[sub_va]])
        
    # Filas iniciales previas al primer sub-split: consenso causal de los modelos base (sin ajuste in-sample)
    unfilled_l1 = (fund_prob_train.sum(axis=1) == 0)
    if np.any(unfilled_l1):
        fund_prob_train[unfilled_l1] = _causal_consensus_prior(X_train_fund.iloc[unfilled_l1])
        
    # Bloques temporales 1 a 5
    for tr_idx, va_idx in l1_splits:
        m_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ])
        fold_w = w_train.iloc[tr_idx] if w_train is not None else None
        m_fold.fit(X_train_fund.iloc[tr_idx], y_train.iloc[tr_idx], lr__sample_weight=fold_w) if fold_w is not None else m_fold.fit(X_train_fund.iloc[tr_idx], y_train.iloc[tr_idx])
        fund_prob_train[va_idx] = m_fold.predict_proba(X_train_fund.iloc[va_idx])

    # Normalizar por seguridad
    fund_prob_train = fund_prob_train / fund_prob_train.sum(axis=1, keepdims=True)
    
    # Entrenar pipeline de Nivel 1 final sobre todo Train para predecir Test
    fund_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    if w_train is not None:
        fund_pipeline.fit(X_train_fund, y_train, lr__sample_weight=w_train)
    else:
        fund_pipeline.fit(X_train_fund, y_train)
        
    fund_prob_test = fund_pipeline.predict_proba(X_test_fund)
    fund_prob_test = fund_prob_test / fund_prob_test.sum(axis=1, keepdims=True)
    
    df_fund_train = pd.DataFrame(fund_prob_train, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_train_fund.index)
    df_fund_test = pd.DataFrame(fund_prob_test, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_test_fund.index)
    
    joblib.dump({'model': fund_pipeline, 'features': X_train_fund.columns.tolist()}, MODEL_SAVE_PATH_FUND)
    
    # ====== ETAPA 2: FINAL (MARKET) STACKER con Regresión Logística L2 + Meta-Features 1X2 ======
    logger.info("Entrenando Stacker Final (Nivel 2) combinando Fundamentales + Mercado + GBM...")
    
    market_models = []
    if 'market' in oof_data:
        market_models.append(oof_data['market'])
    if 'gbm' in oof_data:
        market_models.append(oof_data['gbm'])
        
    if market_models:
        mkt_train = pd.concat([m[0] for m in market_models], axis=1).reset_index(drop=True)
        mkt_test = pd.concat([m[1] for m in market_models], axis=1).reset_index(drop=True)
        X_train_meta = pd.concat([df_fund_train, mkt_train], axis=1).reset_index(drop=True).fillna(0.0)
        X_test_meta = pd.concat([df_fund_test, mkt_test], axis=1).reset_index(drop=True).fillna(0.0)
    else:
        logger.warning("No se encontró OOF de Market. El Stacker Final usará probabilidades fundamentales.")
        X_train_meta = df_fund_train.copy().reset_index(drop=True)
        X_test_meta = df_fund_test.copy().reset_index(drop=True)
        
    # Inyectar Meta-Features 1X2 (dispersión inter-modelo, entropía y cuotas de apertura implícitas)
    X_train_meta = compute_meta_features(X_train_meta, df_train_orig, prefix="meta_").fillna(0.0)
    X_test_meta = compute_meta_features(X_test_meta, df_test_orig, prefix="meta_").fillna(0.0)
    
    # Registrar la lista exacta de características usadas para ajustar final_model
    fit_features = X_train_meta.columns.tolist()

    # Entrenar el modelo final de Nivel 2 (Regresión Logística L2 regularizada)
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))
    ])
    if w_train is not None:
        final_model.fit(X_train_meta, y_train, lr__sample_weight=w_train)
    else:
        final_model.fit(X_train_meta, y_train)
        
    # Evaluación OOF estricta temporal (sin KFold shuffle ni ajuste in-sample en burn-in)
    y_prob_train = np.zeros((len(X_train_meta), 3))
    
    cv = TimeSeriesSplit(n_splits=5)
    splits = list(cv.split(X_train_meta))
    
    # 1. Primer bloque temporal usando sub-splits temporales
    first_idx = splits[0][0]
    sub_cv = TimeSeriesSplit(n_splits=3)
    for sub_tr, sub_va in sub_cv.split(X_train_meta.iloc[first_idx]):
        X_tr, y_tr = X_train_meta.iloc[first_idx[sub_tr]], y_train.iloc[first_idx[sub_tr]]
        X_va = X_train_meta.iloc[first_idx[sub_va]]
        
        w_tr = w_train.iloc[first_idx[sub_tr]] if w_train is not None else None
            
        m = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))])
        m.fit(X_tr, y_tr, lr__sample_weight=w_tr) if w_tr is not None else m.fit(X_tr, y_tr)
        y_prob_train[first_idx[sub_va]] = m.predict_proba(X_va)

    unfilled = (y_prob_train.sum(axis=1) == 0)
    if np.any(unfilled):
        y_prob_train[unfilled] = _causal_consensus_prior(X_train_meta.iloc[unfilled])
        
    # 2. Demás bloques respetando la flecha del tiempo
    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_train_meta.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va = X_train_meta.iloc[va_idx]
        w_tr = w_train.iloc[tr_idx] if w_train is not None else None
            
        m = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))])
        m.fit(X_tr, y_tr, lr__sample_weight=w_tr) if w_tr is not None else m.fit(X_tr, y_tr)
        y_prob_train[va_idx] = m.predict_proba(X_va)

        
    y_prob_train = y_prob_train / y_prob_train.sum(axis=1, keepdims=True)
    
    y_prob_test = final_model.predict_proba(X_test_meta)
    y_prob_test = y_prob_test / y_prob_test.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob_test)
    
    logger.info("=== RESULTADOS META-MODELO FINAL (LOGISTIC REGRESSION L2 + META-FEATURES) ===")
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
        'odds_win': df_train_orig['open_odds_win'].fillna(df_train_orig['odds_win']).values if 'open_odds_win' in df.columns else df_train_orig['odds_win'].values,
        'odds_draw': df_train_orig['open_odds_draw'].fillna(df_train_orig['odds_draw']).values if 'open_odds_draw' in df.columns else df_train_orig['odds_draw'].values,
        'odds_loss': df_train_orig['open_odds_loss'].fillna(df_train_orig['odds_loss']).values if 'open_odds_loss' in df.columns else df_train_orig['odds_loss'].values,
        'closing_odds_win': df_train_orig['closing_odds_win'].values if 'closing_odds_win' in df.columns else df_train_orig['odds_win'].values,
        'closing_odds_draw': df_train_orig['closing_odds_draw'].values if 'closing_odds_draw' in df.columns else df_train_orig['odds_draw'].values,
        'closing_odds_loss': df_train_orig['closing_odds_loss'].values if 'closing_odds_loss' in df.columns else df_train_orig['odds_loss'].values
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
            'odds_win': df_test_orig['open_odds_win'].fillna(df_test_orig['odds_win']).values if 'open_odds_win' in df.columns else df_test_orig['odds_win'].values,
            'odds_draw': df_test_orig['open_odds_draw'].fillna(df_test_orig['odds_draw']).values if 'open_odds_draw' in df.columns else df_test_orig['odds_draw'].values,
            'odds_loss': df_test_orig['open_odds_loss'].fillna(df_test_orig['odds_loss']).values if 'open_odds_loss' in df.columns else df_test_orig['odds_loss'].values,
            'closing_odds_win': df_test_orig['closing_odds_win'].values if 'closing_odds_win' in df.columns else df_test_orig['odds_win'].values,
            'closing_odds_draw': df_test_orig['closing_odds_draw'].values if 'closing_odds_draw' in df.columns else df_test_orig['odds_draw'].values,
            'closing_odds_loss': df_test_orig['closing_odds_loss'].values if 'closing_odds_loss' in df.columns else df_test_orig['odds_loss'].values
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
    
    joblib.dump({'model': final_model, 'features': fit_features}, MODEL_SAVE_PATH_FINAL)
    logger.info(f"Modelo Stacker Final (XGBoost GPU) guardado en {MODEL_SAVE_PATH_FINAL}")

if __name__ == "__main__":
    train_stacker()
