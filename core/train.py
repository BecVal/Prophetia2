import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = '../data/processed/matches_dataset.parquet'
MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'prophetia_rf_model.pkl')

def train_model():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset no encontrado en {DATASET_PATH}")
        return
        
    logger.info("Cargando dataset procesado...")
    df = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    
    logger.info(f"Dimensiones del dataset: {df.shape}")
    
    # Definir las variables independientes (Features - X)
    # NOTA: Para este MVP (36 filas) usamos las estadísticas del propio partido 
    # para clasificar el resultado. En producción usaremos promedios móviles históricos.
    feature_cols = [
        'is_home', 'xg_created_rolling', 'xg_conceded_rolling', 'shots_total_rolling', 'shots_on_target_rolling',
        'passes_total_rolling', 'passes_completed_rolling', 'pass_accuracy_rolling', 'possession_pct_rolling',
        'crosses_rolling', 'corners_rolling', 'through_balls_rolling', 'key_passes_rolling', 'dribbles_completed_rolling',
        'pressures_rolling', 'interceptions_rolling', 'clearances_rolling', 'blocks_rolling', 'ball_recoveries_rolling',
        'actions_under_pressure_rolling', 'fouls_committed_rolling', 'fouls_won_rolling', 
        'yellow_cards_rolling', 'red_cards_rolling', 'aerials_won_rolling'
    ]
    
    # Verificar que todas las columnas existan
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas en el dataset: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]
        
    X = df[feature_cols]
    y_raw = df['outcome']  # 1: Win, 0: Draw, -1: Loss
    
    # Mapear clases para XGBoost (Requiere iniciar en 0) -> 0: Derrota, 1: Empate, 2: Victoria
    y = y_raw.map({-1: 0, 0: 1, 1: 2})
    
    # Manejar valores nulos si los hay
    X = X.fillna(0)
    
    logger.info(f"Entrenando modelo con {len(feature_cols)} variables tácticas...")
    
    # Split: 80% train, 20% test
    # (Al ser tan pocos datos, el test será mínimo, pero mantenemos la estructura correcta)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir y Entrenar XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    
    logger.info("Ajustando modelo XGBoost (Fitting)...")
    xgb_model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    
    logger.info(f"=== RESULTADOS DE EVALUACIÓN ===")
    logger.info(f"Accuracy Global: {acc:.2f}")
    logger.info(f"Log-Loss (Cuanto más bajo, mejor): {loss:.4f}")
    
    logger.info("--- Muestra de Probabilidades (Test Set) ---")
    logger.info("Derrota% | Empate% | Victoria%  -> Realidad (0=D, 1=E, 2=V)")
    for i in range(min(5, len(y_test))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test.iloc[i]
        logger.info(f"{p_loss*100:6.1f}% | {p_draw*100:6.1f}% | {p_win*100:6.1f}%     -> Clase {real}")
    
    # Feature Importance (Importancia de Características)
    logger.info("=== IMPORTANCIA DE CARACTERÍSTICAS (Top 10) ===")
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_cols))):
        logger.info(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # Guardar el modelo
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    MODEL_SAVE_PATH_XGB = os.path.join(MODEL_SAVE_DIR, 'prophetia_xgb_model.pkl')
    joblib.dump(xgb_model, MODEL_SAVE_PATH_XGB)
    logger.info(f"Modelo guardado exitosamente en: {MODEL_SAVE_PATH_XGB}")
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_model()
