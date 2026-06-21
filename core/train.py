import os
import pandas as pd
import numpy as np
import logging
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reducir verbosidad de Optuna para no ensuciar la consola
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_PATH = '../data/processed/matches_dataset.parquet'
MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH_XGB = os.path.join(MODEL_SAVE_DIR, 'prophetia_xgb_model.pkl')

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

def objective(trial, X_train, y_train, cv_strategy, sample_weights):
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
    }
    
    cv_scores = []
    
    for train_idx, val_idx in cv_strategy.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        xgb_eval = XGBClassifier(**param)
        
        # SelectFromModel in the CV loop for fair evaluation (no data leak)
        selector = SelectFromModel(XGBClassifier(random_state=42, device='cuda'), max_features=15)
        
        pipeline = Pipeline([
            ('feature_selection', selector),
            ('classifier', xgb_eval)
        ])
        
        # Entrenamos el pipeline
        pipeline.fit(X_tr, y_tr)
        
        y_prob = pipeline.predict_proba(X_val)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

def train_model():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset no encontrado en {DATASET_PATH}")
        return

    logger.info("Cargando dataset procesado...")
    df = pd.read_parquet(DATASET_PATH, engine='fastparquet')

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

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})

    logger.info(f"Entrenando modelo con {len(feature_cols)} variables tácticas...")

    # Split Temporal: 80% train, 20% test (Chronological Split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    teams_test = df['team'].iloc[split_idx:]
    opponents_test = df['opponent'].iloc[split_idx:]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    logger.info("Iniciando optimización de hiperparámetros (Optuna) minimizando Log-Loss...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, cv_strategy, sample_weights), n_trials=30)
    
    logger.info(f"Mejores parámetros XGBoost: {study.best_params}")

    xgb_best = XGBClassifier(
        **study.best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    
    xgb_selector = SelectFromModel(XGBClassifier(random_state=42, device='cuda'), max_features=15)
    
    pipeline_xgb = Pipeline([
        ('feature_selection', xgb_selector),
        ('clf', xgb_best)
    ])
    
    # Regresión Logística
    lr_model = LogisticRegression(max_iter=1500, random_state=42, class_weight='balanced')
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', lr_model)
    ])

    voting_clf = VotingClassifier(
        estimators=[('xgb_pipe', pipeline_xgb), ('lr_pipe', pipeline_lr)],
        voting='soft'
    )

    logger.info("Entrenando modelo final ensamblado (VotingClassifier con Pipelines)...")
    voting_clf.fit(X_train, y_train)

    logger.info("Aplicando Calibración Probabilística (Platt Scaling)...")
    calibrated_clf = CalibratedClassifierCV(estimator=voting_clf, method='sigmoid', cv=cv_strategy)
    calibrated_clf.fit(X_train, y_train)

    final_model = calibrated_clf

    # Evaluar
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    logger.info("=== RESULTADOS DE EVALUACIÓN ===")
    logger.info(f"Accuracy Global: {acc:.4f}")
    logger.info(f"Log-Loss (Optimizando rentabilidad): {loss:.4f}")

    logger.info("--- Muestra de Predicciones de Valor (Set de Prueba) ---")
    for i in range(min(5, len(y_test))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test.iloc[i]
        team_name = teams_test.iloc[i]
        opp_name = opponents_test.iloc[i]
        is_home_flag = X_test['is_home'].iloc[i]
        
        local = team_name if is_home_flag == 1 else opp_name
        visitante = opp_name if is_home_flag == 1 else team_name
        real_str = f"Victoria {team_name}" if real == 2 else "Empate" if real == 1 else f"Victoria {opp_name}"
        
        logger.info(f"Partido {i+1}: {local} vs {visitante}")
        logger.info(f"  -> Prob. {team_name}: {p_win*100:5.1f}% | Empate: {p_draw*100:5.1f}% | {opp_name}: {p_loss*100:5.1f}%")
        logger.info(f"  => Realidad: {real_str} (Clase {real})\n")

    # Guardar modelo
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    joblib.dump({'model': final_model, 'features': feature_cols}, MODEL_SAVE_PATH_XGB)
    logger.info(f"Modelo anti-leakage guardado en: {MODEL_SAVE_PATH_XGB}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_model()
