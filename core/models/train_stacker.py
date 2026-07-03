import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.frozen import FrozenEstimator

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

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

def train_stacker():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx]) if 'match_date' in df.columns else None
    
    logger.info("Cargando predicciones OOF de los modelos base...")
    
    oof_data = {}
    for mod in ['poisson', 'context', 'nn', 'draws', 'market', 'gbm']:
        tr, ts = load_oof(mod, split_idx)
        if tr is not None:
            oof_data[mod] = (tr, ts)
            
    # ====== ETAPA 1: FUNDAMENTAL STACKER ======
    # Modelos fundamentales: poisson, context, nn, draws
    fundamental_train_list = []
    fundamental_test_list = []
    
    for mod in ['poisson', 'context', 'nn', 'draws']:
        if mod in oof_data:
            fundamental_train_list.append(oof_data[mod][0])
            fundamental_test_list.append(oof_data[mod][1])
            
    if not fundamental_train_list:
        raise ValueError("No se encontraron modelos fundamentales. Debes entrenar al menos poisson o context primero.")
        
    X_train_fund = pd.concat(fundamental_train_list, axis=1).fillna(0)
    X_test_fund = pd.concat(fundamental_test_list, axis=1).fillna(0)
    
    logger.info("Entrenando Stacker Fundamental (Nivel 1)...")
    
    fund_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # C=1.0 para L2 moderada
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])
    
    # Calibración para Nivel 1
    calib_idx = int(len(X_train_fund) * 0.75)
    X_tr_f_sub, X_calib_f = X_train_fund.iloc[:calib_idx], X_train_fund.iloc[calib_idx:]
    y_tr_sub, y_calib = y_train.iloc[:calib_idx], y_train.iloc[calib_idx:]
    
    w_tr_sub = get_time_weights(train_dates.iloc[:calib_idx]) if train_dates is not None else None
    w_calib = get_time_weights(train_dates.iloc[calib_idx:]) if train_dates is not None else None
    
    if w_tr_sub is not None:
        fund_pipeline.fit(X_tr_f_sub, y_tr_sub, lr__sample_weight=w_tr_sub)
    else:
        fund_pipeline.fit(X_tr_f_sub, y_tr_sub)
        
    calibrated_fund = CalibratedClassifierCV(estimator=FrozenEstimator(fund_pipeline), method='isotonic')
    
    if w_calib is not None:
        calibrated_fund.fit(X_calib_f, y_calib, sample_weight=w_calib)
    else:
        calibrated_fund.fit(X_calib_f, y_calib)
        
    # Obtener predicciones fundamentales para pasar al Nivel 2
    fund_prob_train = calibrated_fund.predict_proba(X_train_fund)
    fund_prob_test = calibrated_fund.predict_proba(X_test_fund)
    
    df_fund_train = pd.DataFrame(fund_prob_train, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_train_fund.index)
    df_fund_test = pd.DataFrame(fund_prob_test, columns=['fund_prob_loss', 'fund_prob_draw', 'fund_prob_win'], index=X_test_fund.index)
    
    joblib.dump({'model': calibrated_fund, 'features': X_train_fund.columns.tolist()}, MODEL_SAVE_PATH_FUND)
    
    # ====== ETAPA 2: FINAL (MARKET) STACKER ======
    # Combina Predicciones Fundamentales con el Modelo de Mercado
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
        logger.warning("No se encontró OOF de Market. El Stacker Final usará open_probs si existen, o será igual al Fundamental.")
        X_train_meta = df_fund_train.copy()
        X_test_meta = df_fund_test.copy()
        if 'open_prob_win' in df.columns:
            open_probs_train = df[['open_prob_loss', 'open_prob_draw', 'open_prob_win']].iloc[:split_idx]
            open_probs_test = df[['open_prob_loss', 'open_prob_draw', 'open_prob_win']].iloc[split_idx:]
            X_train_meta = pd.concat([X_train_meta, open_probs_train], axis=1).fillna(0)
            X_test_meta = pd.concat([X_test_meta, open_probs_test], axis=1).fillna(0)
    
    # Para evitar que Mercado domine (Lazy Stacker), usamos regularización más fuerte (C=0.1)
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42, C=0.1))
    ])
    
    X_tr_m_sub, X_calib_m = X_train_meta.iloc[:calib_idx], X_train_meta.iloc[calib_idx:]
    
    if w_tr_sub is not None:
        final_pipeline.fit(X_tr_m_sub, y_tr_sub, lr__sample_weight=w_tr_sub)
    else:
        final_pipeline.fit(X_tr_m_sub, y_tr_sub)
        
    calibrated_final = CalibratedClassifierCV(estimator=FrozenEstimator(final_pipeline), method='isotonic')
    
    if w_calib is not None:
        calibrated_final.fit(X_calib_m, y_calib, sample_weight=w_calib)
    else:
        calibrated_final.fit(X_calib_m, y_calib)
        
    final_model = calibrated_final
    
    # Evaluación
    y_prob_train = final_model.predict_proba(X_train_meta)
    y_prob_train = y_prob_train / y_prob_train.sum(axis=1, keepdims=True)
    
    y_prob_test = final_model.predict_proba(X_test_meta)
    y_prob_test = y_prob_test / y_prob_test.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob_test)
    
    logger.info("=== RESULTADOS META-MODELO FINAL (DOBLE STACKING) ===")
    logger.info(f"Accuracy Global: {acc:.4f}")
    logger.info(f"Log-Loss: {loss:.4f}")
    
    # Análisis de pesos de Nivel 2
    try:
        lr_coefs = final_pipeline.named_steps['lr'].coef_
        logger.info(f"Coeficientes Nivel 2 (Regresión Logística L2): {lr_coefs}")
    except:
        pass
        
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
    df_train = pd.DataFrame({
        'match_date': df['match_date'].iloc[:split_idx].values if 'match_date' in df.columns else np.array([None]*len(y_train)),
        'prob_loss': y_prob_train[:, 0],
        'prob_draw': y_prob_train[:, 1],
        'prob_win': y_prob_train[:, 2],
    })
    
    has_odds = 'odds_win' in df.columns
    if has_odds:
        df_test = pd.DataFrame({
            'match_date': df['match_date'].iloc[split_idx:].values if 'match_date' in df.columns else np.array([None]*len(y_test)),
            'competition': df['competition'].iloc[split_idx:].values if 'competition' in df.columns else np.array([None]*len(y_test)),
            'team': df['team'].iloc[split_idx:].values,
            'opponent': df['opponent'].iloc[split_idx:].values,
            'is_home': df['is_home'].iloc[split_idx:].values,
            'prob_loss': y_prob_test[:, 0],
            'prob_draw': y_prob_test[:, 1],
            'prob_win': y_prob_test[:, 2],
            'outcome': y_test.values,
            'odds_win': df['open_odds_win'].iloc[split_idx:].values if 'open_odds_win' in df.columns else df['odds_win'].iloc[split_idx:].values,
            'odds_draw': df['open_odds_draw'].iloc[split_idx:].values if 'open_odds_draw' in df.columns else df['odds_draw'].iloc[split_idx:].values,
            'odds_loss': df['open_odds_loss'].iloc[split_idx:].values if 'open_odds_loss' in df.columns else df['odds_loss'].iloc[split_idx:].values,
            'closing_odds_win': df['odds_win'].iloc[split_idx:].values,
            'closing_odds_draw': df['odds_draw'].iloc[split_idx:].values,
            'closing_odds_loss': df['odds_loss'].iloc[split_idx:].values
        })
        
        df_test.to_parquet(os.path.join(PROCESSED_DIR, 'test_predictions.parquet'), engine='fastparquet')
        logger.info("Guardado test_predictions.parquet")
    
    df_train.to_parquet(os.path.join(PROCESSED_DIR, 'train_predictions.parquet'), engine='fastparquet')
    
    # Pasar variables fair si existen
    if 'fair_loss' in df.columns:
        for col in ['fair_loss', 'fair_draw', 'fair_win']:
            X_train_meta[col] = df[col].iloc[:split_idx].values
            X_test_meta[col] = df[col].iloc[split_idx:].values
            
    if 'open_fair_loss' in df.columns:
        for col in ['open_fair_loss', 'open_fair_draw', 'open_fair_win']:
            X_train_meta[col] = df[col].iloc[:split_idx].values
            X_test_meta[col] = df[col].iloc[split_idx:].values

    X_train_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_train.parquet'), engine='fastparquet')
    X_test_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_test.parquet'), engine='fastparquet')
    
    joblib.dump({'model': final_model, 'features': X_train_meta.columns.tolist()}, MODEL_SAVE_PATH_FINAL)
    logger.info(f"Modelo Stacker Final guardado en {MODEL_SAVE_PATH_FINAL}")

if __name__ == "__main__":
    train_stacker()
