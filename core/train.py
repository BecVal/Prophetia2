import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = '../data/processed/matches_dataset.parquet'
MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'prophetia_rf_model.pkl')

FEATURE_DESCRIPTIONS = {
    'is_home': "Ventaja de localía (1 si juega en su estadio, 0 si es visitante).",
    'rest_days': "Días de descanso desde el último partido jugado.",
    'relative_attack_strength': "Fuerza relativa de ataque (xG a favor vs xG en contra del rival).",
    'team_elo': "Calidad histórica del equipo según el sistema de puntuación ELO.",
    'opp_elo': "Calidad histórica del equipo rival (ELO).",
    'elo_diff': "Diferencia de ELO (Ventaja matemática sobre el rival antes del partido).",
}

def get_feature_description(feat_name):
    if feat_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feat_name]
    
    base_name = feat_name.replace('_ema3', '').replace('_ema5', '')
    span = "Tendencia reciente (últimos 3 partidos)" if "_ema3" in feat_name else "Tendencia a mediano plazo (últimos 5 partidos)"
    
    base_desc = {
        'xg_created': "Goles Esperados (xG) generados",
        'xg_conceded': "Goles Esperados (xG) concedidos",
        'shots_total': "Tiros totales realizados",
        'shots_on_target': "Tiros a puerta",
        'passes_total': "Volumen total de pases intentados",
        'passes_completed': "Pases completados con éxito",
        'pass_accuracy': "Precisión de pases (% de acierto)",
        'possession_pct': "Dominio de posesión del balón",
        'crosses': "Centros al área",
        'corners': "Tiros de esquina provocados",
        'through_balls': "Pases filtrados (rompe-líneas)",
        'key_passes': "Pases clave que terminan en tiro",
        'dribbles_completed': "Regates completados",
        'pressures': "Presión ejercida sobre el rival",
        'interceptions': "Intercepciones tácticas (cortes de pase)",
        'clearances': "Despejes defensivos",
        'blocks': "Tiros o pases bloqueados",
        'ball_recoveries': "Recuperaciones de balón",
        'actions_under_pressure': "Acciones exitosas bajo presión rival",
        'fouls_committed': "Faltas tácticas o agresivas cometidas",
        'fouls_won': "Faltas recibidas",
        'yellow_cards': "Tarjetas amarillas acumuladas",
        'red_cards': "Expulsiones",
        'aerials_won': "Duelos aéreos ganados"
    }
    
    if base_name in base_desc:
        return f"{base_desc[base_name]} - {span}"
    
    return "Métrica táctica avanzada."


def train_model():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset no encontrado en {DATASET_PATH}")
        return

    logger.info("Cargando dataset procesado...")
    df = pd.read_parquet(DATASET_PATH, engine='fastparquet')

    logger.info(f"Dimensiones del dataset: {df.shape}")

    # Definir las variables independientes (Features - X)
    base_stats = [
        'xg_created', 'xg_conceded', 'shots_total', 'shots_on_target',
        'passes_total', 'passes_completed', 'pass_accuracy', 'possession_pct',
        'crosses', 'corners', 'through_balls', 'key_passes',
        'dribbles_completed', 'pressures', 'interceptions', 'clearances',
        'blocks', 'ball_recoveries', 'actions_under_pressure',
        'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
        'aerials_won']
        
    feature_cols = ['is_home', 'rest_days', 'relative_attack_strength', 'team_elo', 'opp_elo', 'elo_diff']
    for stat in base_stats:
        feature_cols.append(f"{stat}_ema3")
        feature_cols.append(f"{stat}_ema5")

    # Verificar que todas las columnas existan
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(
            f"Faltan las siguientes columnas en el dataset: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y_raw = df['outcome']  # 1: Win, 0: Draw, -1: Loss

    # Mapear clases para XGBoost (Requiere iniciar en 0) -> 0: Derrota, 1:
    # Empate, 2: Victoria
    y = y_raw.replace({-1: 0, 0: 1, 1: 2})

    # Manejar valores nulos si los hay
    X = X.fillna(0)

    msg = f"Entrenando modelo con {len(feature_cols)} variables tácticas..."
    logger.info(msg)

    # Split Temporal: 80% train, 20% test
    # Al ser series de tiempo, respetamos el orden cronológico
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    teams_test = df['team'].iloc[split_idx:]
    opponents_test = df['opponent'].iloc[split_idx:]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Definir validación cruzada (StratifiedKFold para dataset pequeño)
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Definir la grilla de hiperparámetros (Simplificada para evitar overfitting)
    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [3, 5]
    }

    xgb_base = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )

    logger.info(
        "Iniciando optimización de hiperparámetros (RandomizedSearchCV)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=15,
        scoring='neg_log_loss',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    logger.info(f"Mejores parámetros: {random_search.best_params_}")

    # Feature Selection (Selección de características)
    logger.info("Aplicando Feature Selection...")
    selector = SelectFromModel(best_xgb, prefit=False, max_features=15)
    selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    selected_mask = selector.get_support()
    selected_features = np.array(feature_cols)[selected_mask]
    msg = (f"Variables seleccionadas ({len(selected_features)}): "
           f"{list(selected_features)}")
    logger.info(msg)

    # Re-entrenar modelo optimizado con características seleccionadas
    logger.info("Calculando pesos de clases (Sample Weights)...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    logger.info("Configurando Ensamblaje de Modelos (VotingClassifier)...")
    xgb_best = XGBClassifier(
        **random_search.best_params_,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced')

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

    logger.info("Aplicando Calibración Probabilística (Platt Scaling)...")
    # Usaremos el cv_strategy definido arriba para asegurar que los folds tengan todas las clases.
    calibrated_clf = CalibratedClassifierCV(estimator=voting_clf, method='sigmoid', cv=cv_strategy)
    calibrated_clf.fit(X_train_sel, y_train)

    final_model = calibrated_clf

    # Evaluar
    y_pred = final_model.predict(X_test_sel)
    y_prob = final_model.predict_proba(X_test_sel)

    # Normalizar probabilidades para evitar advertencias de log_loss
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    logger.info("=== RESULTADOS DE EVALUACIÓN ===")
    logger.info(f"Accuracy Global: {acc:.2f}")
    logger.info(f"Log-Loss (Cuanto más bajo, mejor): {loss:.4f}")

    logger.info("--- Muestra de Predicciones (Set de Prueba) ---")
    logger.info("Interpretación de Clases: 0 = Derrota Local (o Victoria Visitante), 1 = Empate, 2 = Victoria Local")
    for i in range(min(5, len(y_test))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test.iloc[i]
        
        team_name = teams_test.iloc[i]
        opp_name = opponents_test.iloc[i]
        is_home_flag = X_test['is_home'].iloc[i]
        
        if is_home_flag == 1:
            local = team_name
            visitante = opp_name
        else:
            local = opp_name
            visitante = team_name

        real_str = f"Victoria {team_name}" if real == 2 else "Empate" if real == 1 else f"Victoria {opp_name}"
        
        logger.info(f"Partido {i+1}: {local} (Local) vs {visitante} (Visitante)")
        logger.info(f"  -> Prob. Victoria {team_name}: {p_win*100:5.1f}%")
        logger.info(f"  -> Prob. Empate:         {p_draw*100:5.1f}%")
        logger.info(f"  -> Prob. Victoria {opp_name}: {p_loss*100:5.1f}%")
        logger.info(f"  => Resultado Real: {real_str} (Clase {real})\n")

    # Feature Importance Final
    logger.info("=== IMPORTANCIA DE CARACTERÍSTICAS (Base XGBoost) ===")
    logger.info("¿Qué variables consideró más importantes el modelo para tomar sus decisiones?")
    
    importances = best_xgb.feature_importances_
    sel_importances = importances[selected_mask]
    indices = np.argsort(sel_importances)[::-1]

    for i in range(len(selected_features)):
        feat_name = selected_features[indices[i]]
        feat_imp = sel_importances[indices[i]]
        desc = get_feature_description(feat_name)
        logger.info(f"{i + 1}. {feat_name} (Importancia: {feat_imp:.4f})")
        logger.info(f"   ↳ {desc}")

    logger.info("=== RESULTADOS TÉCNICOS ===")
    logger.info("Derrota% | Empate% | Victoria%  -> Realidad (0=D, 1=E, 2=V)")
    for i in range(min(5, len(y_test))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test.iloc[i]
        msg = (f"{p_loss*100:6.1f}% | {p_draw*100:6.1f}% | "
               f"{p_win*100:6.1f}%     -> Clase {real}")
        logger.info(msg)

    # Guardar el modelo
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    MODEL_SAVE_PATH_XGB = os.path.join(
        MODEL_SAVE_DIR, 'prophetia_xgb_model.pkl')
    # Guardamos un diccionario con el modelo y el selector, más fácil para
    # inferencia
    joblib.dump({'model': final_model, 'selector': selector,
                'features': selected_features}, MODEL_SAVE_PATH_XGB)
    logger.info(f"Modelo guardado exitosamente en: {MODEL_SAVE_PATH_XGB}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_model()
