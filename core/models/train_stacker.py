import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'stacker_model.pkl')
PROCESSED_DIR = '../data/processed'

def train_stacker():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info("Cargando predicciones OOF de los modelos base...")
    poisson_train = pd.read_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_train.parquet'), engine='fastparquet')
    poisson_test = pd.read_parquet(os.path.join(PROCESSED_DIR, 'oof_poisson_test.parquet'), engine='fastparquet')
    
    context_train = pd.read_parquet(os.path.join(PROCESSED_DIR, 'oof_context_train.parquet'), engine='fastparquet')
    context_test = pd.read_parquet(os.path.join(PROCESSED_DIR, 'oof_context_test.parquet'), engine='fastparquet')
    
    # Construir X_train y X_test para el Stacker
    X_train_meta = pd.concat([poisson_train, context_train], axis=1)
    X_test_meta = pd.concat([poisson_test, context_test], axis=1)
    
    # Añadir cuotas de apertura si existen
    if 'open_prob_win' in df.columns:
        logger.info("Añadiendo open_probs al meta-modelo...")
        open_probs = df[['open_prob_loss', 'open_prob_draw', 'open_prob_win']].copy()
        
        X_train_meta = pd.concat([X_train_meta.reset_index(drop=True), open_probs.iloc[:split_idx].reset_index(drop=True)], axis=1)
        X_test_meta = pd.concat([X_test_meta.reset_index(drop=True), open_probs.iloc[split_idx:].reset_index(drop=True)], axis=1)
        
    # Llenar NaNs de open_probs o cualquier otra columna para LogisticRegression
    X_train_meta = X_train_meta.fillna(0)
    X_test_meta = X_test_meta.fillna(0)
        
    logger.info("Entrenando Meta-Modelo (Logistic Regression para evitar overfitting)...")
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Para calibrar sin leakage, separamos un conjunto de calibración temporal del Train
    calib_idx = int(len(X_train_meta) * 0.75)
    X_tr_sub, X_calib = X_train_meta.iloc[:calib_idx], X_train_meta.iloc[calib_idx:]
    y_tr_sub, y_calib = y_train.iloc[:calib_idx], y_train.iloc[calib_idx:]
    
    meta_model.fit(X_tr_sub, y_tr_sub)
    
    logger.info("Aplicando Calibración Isotónica al Meta-Modelo...")
    calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(meta_model), method='isotonic')
    calibrated_clf.fit(X_calib, y_calib)
    
    final_model = calibrated_clf
    
    # Evaluación
    y_prob_train = final_model.predict_proba(X_train_meta)
    y_prob_train = y_prob_train / y_prob_train.sum(axis=1, keepdims=True)
    
    y_prob_test = final_model.predict_proba(X_test_meta)
    y_prob_test = y_prob_test / y_prob_test.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(y_prob_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob_test)
    
    logger.info("=== RESULTADOS META-MODELO ===")
    logger.info(f"Accuracy Global: {acc:.4f}")
    logger.info(f"Log-Loss: {loss:.4f}")
    
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
    
    # Save the original base features so CLV model can calculate Fair Odds properly if needed
    # Actually, train_clv_model.py expects fair_loss, fair_draw, fair_win from X_train and X_test, which are inside X_train.parquet
    # Let's just save the meta features as X_train.parquet, but make sure to include fair_* columns!
    # Wait, the fair_* columns were created by removing Vig from open_odds. Let's add them back to X_train_meta.
    if 'fair_loss' in df.columns:
        X_train_meta['fair_loss'] = df['fair_loss'].iloc[:split_idx].values
        X_train_meta['fair_draw'] = df['fair_draw'].iloc[:split_idx].values
        X_train_meta['fair_win'] = df['fair_win'].iloc[:split_idx].values
        
        X_test_meta['fair_loss'] = df['fair_loss'].iloc[split_idx:].values
        X_test_meta['fair_draw'] = df['fair_draw'].iloc[split_idx:].values
        X_test_meta['fair_win'] = df['fair_win'].iloc[split_idx:].values

    if 'open_fair_loss' in df.columns:
        X_train_meta['open_fair_loss'] = df['open_fair_loss'].iloc[:split_idx].values
        X_train_meta['open_fair_draw'] = df['open_fair_draw'].iloc[:split_idx].values
        X_train_meta['open_fair_win'] = df['open_fair_win'].iloc[:split_idx].values
        
        X_test_meta['open_fair_loss'] = df['open_fair_loss'].iloc[split_idx:].values
        X_test_meta['open_fair_draw'] = df['open_fair_draw'].iloc[split_idx:].values
        X_test_meta['open_fair_win'] = df['open_fair_win'].iloc[split_idx:].values

    X_train_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_train.parquet'), engine='fastparquet')
    X_test_meta.to_parquet(os.path.join(PROCESSED_DIR, 'X_test.parquet'), engine='fastparquet')
    
    joblib.dump({'model': final_model, 'features': X_train_meta.columns.tolist()}, MODEL_SAVE_PATH)
    logger.info(f"Modelo Stacker guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_stacker()
