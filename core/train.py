import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
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
        'yellow_cards_rolling', 'red_cards_rolling', 'aerials_won_rolling',
        'rest_days', 'relative_attack_strength'
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
    
    # Split Temporal: 80% train, 20% test
    # Al ser series de tiempo, respetamos el orden cronológico
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Definir TimeSeriesSplit para validación cruzada
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Definir la grilla de hiperparámetros
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_base = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    
    logger.info("Iniciando optimización de hiperparámetros (RandomizedSearchCV)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=15,
        scoring='neg_log_loss',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    logger.info(f"Mejores parámetros: {random_search.best_params_}")
    
    # Feature Selection (Selección de características)
    logger.info("Aplicando Feature Selection...")
    selector = SelectFromModel(best_xgb, prefit=True, max_features=15)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    selected_features = np.array(feature_cols)[selected_mask]
    logger.info(f"Variables seleccionadas ({len(selected_features)}): {list(selected_features)}")
    
    # Re-entrenar modelo optimizado con características seleccionadas
    logger.info("Calculando pesos de clases (Sample Weights)...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    logger.info("Configurando Ensamblaje de Modelos (VotingClassifier)...")
    xgb_best = XGBClassifier(**random_search.best_params_, objective='multi:softprob', num_class=3, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_best), ('lr', lr_model)],
        voting='soft'
    )
    
    logger.info("Entrenando modelo final ensamblado (VotingClassifier)...")
    try:
        voting_clf.fit(X_train_sel, y_train, sample_weight=sample_weights)
    except TypeError:
        # Fallback
        voting_clf.fit(X_train_sel, y_train)
        
    final_model = voting_clf
    
    # Evaluar
    y_pred = final_model.predict(X_test_sel)
    y_prob = final_model.predict_proba(X_test_sel)
    
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
    
    # Feature Importance Final (Extraído del mejor XGBoost base ya que el calibrador/ensamble oculta esto)
    logger.info("=== IMPORTANCIA DE CARACTERÍSTICAS (Base XGBoost) ===")
    importances = best_xgb.feature_importances_
    # Ojo: importances es del modelo pre-selección (27 features), tenemos que mapearlo
    # Solo mostramos las seleccionadas
    sel_importances = importances[selected_mask]
    indices = np.argsort(sel_importances)[::-1]
    
    for i in range(len(selected_features)):
        logger.info(f"{i+1}. {selected_features[indices[i]]}: {sel_importances[indices[i]]:.4f}")
        
    # Guardar el modelo
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    MODEL_SAVE_PATH_XGB = os.path.join(MODEL_SAVE_DIR, 'prophetia_xgb_model.pkl')
    # Guardamos un diccionario con el modelo y el selector, más fácil para inferencia
    joblib.dump({'model': final_model, 'selector': selector, 'features': selected_features}, MODEL_SAVE_PATH_XGB)
    logger.info(f"Modelo guardado exitosamente en: {MODEL_SAVE_PATH_XGB}")
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_model()
