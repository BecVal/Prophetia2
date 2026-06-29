import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = '../data/processed/matches_with_odds.parquet'
PREDICTIONS_PATH = '../data/processed/test_predictions.parquet'
X_TRAIN_PATH = '../data/processed/X_train.parquet'
X_TEST_PATH = '../data/processed/X_test.parquet'
TRAIN_PREDS_PATH = '../data/processed/train_predictions.parquet'

def train_clv_model():
    if not all(os.path.exists(p) for p in [DATASET_PATH, PREDICTIONS_PATH, X_TRAIN_PATH, X_TEST_PATH, TRAIN_PREDS_PATH]):
        logger.error("Faltan archivos requeridos. Asegúrate de haber ejecutado train.py primero.")
        return

    logger.info("Cargando datasets base y cuotas...")
    df = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    if 'match_date' in df.columns:
        df = df.sort_values('match_date').reset_index(drop=True)
    df = df[df['is_home'] == 1].reset_index(drop=True)

    # Cargar X_train y X_test (con las features generadas como poisson y medias)
    X_train = pd.read_parquet(X_TRAIN_PATH, engine='fastparquet')
    X_test = pd.read_parquet(X_TEST_PATH, engine='fastparquet')
    
    # Cargar probabilidades
    df_train_preds = pd.read_parquet(TRAIN_PREDS_PATH, engine='fastparquet')
    df_test_preds = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    
    # Calcular target drifts
    df['drift_loss'] = (df['odds_loss'] / df['open_odds_loss']) - 1
    df['drift_draw'] = (df['odds_draw'] / df['open_odds_draw']) - 1
    df['drift_win'] = (df['odds_win'] / df['open_odds_win']) - 1
    
    y_drift_loss = df['drift_loss'].fillna(0)
    y_drift_draw = df['drift_draw'].fillna(0)
    y_drift_win = df['drift_win'].fillna(0)
    
    split_idx = int(len(df) * 0.8)
    
    y_drift_loss_train = y_drift_loss.iloc[:split_idx]
    y_drift_draw_train = y_drift_draw.iloc[:split_idx]
    y_drift_win_train = y_drift_win.iloc[:split_idx]
    
    # Inyectar probabilidades base a las features
    X_train['prob_loss'] = df_train_preds['prob_loss'].values
    X_train['prob_draw'] = df_train_preds['prob_draw'].values
    X_train['prob_win'] = df_train_preds['prob_win'].values
    
    X_test['prob_loss'] = df_test_preds['prob_loss'].values
    X_test['prob_draw'] = df_test_preds['prob_draw'].values
    X_test['prob_win'] = df_test_preds['prob_win'].values
    
    logger.info("Entrenando Meta-Modelos XGBoost (Odds Drift)...")
    xgb_drift = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, device='cuda')
    
    xgb_drift.fit(X_train, y_drift_win_train)
    pred_drift_win = xgb_drift.predict(X_test)
    
    xgb_drift.fit(X_train, y_drift_draw_train)
    pred_drift_draw = xgb_drift.predict(X_test)
    
    xgb_drift.fit(X_train, y_drift_loss_train)
    pred_drift_loss = xgb_drift.predict(X_test)
    
    logger.info("Actualizando test_predictions.parquet con el meta-modelo...")
    
    df_test_preds['pred_drift_loss'] = pred_drift_loss
    df_test_preds['pred_drift_draw'] = pred_drift_draw
    df_test_preds['pred_drift_win'] = pred_drift_win
    
    df_test_preds.to_parquet(PREDICTIONS_PATH, engine='fastparquet')
    logger.info(f"Meta-predicciones añadidas exitosamente a {PREDICTIONS_PATH}")
    logger.info("Ejecuta 'python core/simulate_bankroll.py' a continuación para la evaluación financiera independiente.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_clv_model()
