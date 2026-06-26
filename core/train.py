import os
import pandas as pd
import numpy as np
import logging
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reducir verbosidad de Optuna para no ensuciar la consola
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Intenta cargar el dataset con cuotas, si no, usa el base.
DATASET_PATH = '../data/processed/matches_with_odds.parquet'
FALLBACK_DATASET = '../data/processed/matches_dataset.parquet'
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

def objective(trial, X_train, y_train, cv_strategy):
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
        
        pipeline = Pipeline([
            ('classifier', xgb_eval)
        ])
        
        # Entrenamos el pipeline sin pesos (Fix Isotonic Calibration)
        pipeline.fit(X_tr, y_tr)
        
        y_prob = pipeline.predict_proba(X_val)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        cv_scores.append(log_loss(y_val, y_prob))
        
    return np.mean(cv_scores)

def train_model():
    path_to_load = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    if not os.path.exists(path_to_load):
        logger.error(f"Dataset no encontrado en {path_to_load}")
        return

    logger.info(f"Cargando dataset procesado: {path_to_load}...")
    df = pd.read_parquet(path_to_load, engine='fastparquet')

    if 'match_date' in df.columns:
        logger.info("Ordenando el dataset cronológicamente para evitar Data Leakage...")
        df = df.sort_values('match_date').reset_index(drop=True)
    else:
        logger.warning("No se encontró columna 'match_date'. Posible Data Leakage.")

    logger.info("Filtrando eventos solo desde la perspectiva local para evitar Double-Row Betting...")
    df = df[df['is_home'] == 1].reset_index(drop=True)

    base_stats = [
        'xg_created', 'xg_conceded', 'shots_total', 'shots_on_target',
        'passes_total', 'passes_completed', 'pass_accuracy', 'possession_pct',
        'crosses', 'corners', 'through_balls', 'key_passes',
        'dribbles_completed', 'pressures', 'interceptions', 'clearances',
        'blocks', 'ball_recoveries', 'actions_under_pressure',
        'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
        'aerials_won']
        
    feature_cols = ['is_home', 'rest_days', 'relative_attack_strength', 
                    'team_att_rating', 'team_def_rating', 'opp_att_rating', 'opp_def_rating',
                    'team_elo', 'opp_elo', 'elo_diff',
                    'h2h_games_played', 'h2h_points_last_5', 'h2h_win_rate_hist', 'h2h_draw_rate_hist', 'is_european_hangover',
                    'open_prob_win', 'open_prob_draw', 'open_prob_loss']
    for stat in base_stats:
        feature_cols.append(f"{stat}_ema3")
        feature_cols.append(f"{stat}_ema5")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    y_scored = df['goals_scored'].fillna(0)
    y_conceded = df['goals_conceded'].fillna(0)

    logger.info(f"Entrenando modelo con {len(feature_cols)} variables base...")

    # Split Temporal: 80% train, 20% test (Chronological Split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_scored_train, y_conceded_train = y_scored.iloc[:split_idx], y_conceded.iloc[:split_idx]
    y_scored_test, y_conceded_test = y_scored.iloc[split_idx:], y_conceded.iloc[split_idx:]
    teams_test = df['team'].iloc[split_idx:]
    opponents_test = df['opponent'].iloc[split_idx:]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ==========================
    # FASE 1: MODELOS POISSON
    # ==========================
    logger.info("Fase 1: Entrenando Modelos Poisson para Goles Esperados...")
    xgb_poisson = XGBRegressor(objective='count:poisson', n_estimators=100, learning_rate=0.05, random_state=42, device='cuda')
    
    # Función personalizada para predecir OOS (Out-Of-Sample) con TimeSeriesSplit
    def get_expanding_predictions(estimator, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        preds = np.zeros(len(X))
        preds[:] = np.nan
        # Generar predicciones estrictamente para el futuro
        for train_idx, val_idx in tscv.split(X):
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds[val_idx] = estimator.predict(X.iloc[val_idx])
        
        # Para la primera partición base (que no tiene predicción OOS),
        # rellenamos entrenando y prediciendo sobre ella misma (subóptimo, pero evita perder los datos)
        first_train_idx = next(tscv.split(X))[0]
        estimator.fit(X.iloc[first_train_idx], y.iloc[first_train_idx])
        preds[first_train_idx] = estimator.predict(X.iloc[first_train_idx])
        return preds

    # Predecir Out-Of-Sample en Train usando Expanding Window para eliminar Leakage temporal
    logger.info("Calculando predicciones Poisson Out-Of-Sample (Expanding Window)...")
    pred_scored_train = get_expanding_predictions(xgb_poisson, X_train, y_scored_train)
    pred_conceded_train = get_expanding_predictions(xgb_poisson, X_train, y_conceded_train)
    
    # Entrenar en todo Train y predecir en Test
    xgb_poisson.fit(X_train, y_scored_train)
    pred_scored_test = xgb_poisson.predict(X_test)
    
    xgb_poisson.fit(X_train, y_conceded_train)
    pred_conceded_test = xgb_poisson.predict(X_test)

    # Calcular Probabilidad Bivariada de Empate (Dixon-Coles)
    def calc_dixon_coles_draw(lam_scored, lam_conceded, rho=-0.15):
        prob = 0
        for i in range(6):
            p_scored = poisson.pmf(i, lam_scored)
            p_conceded = poisson.pmf(i, lam_conceded)
            base_prob = p_scored * p_conceded
            
            # Ajuste de correlación (tau) para Dixon-Coles
            tau = 1.0
            if i == 0: # 0-0
                tau = 1 - (lam_scored * lam_conceded * rho)
            elif i == 1: # 1-1
                tau = 1 - rho
                
            tau = max(0, tau)
            prob += base_prob * tau
        return prob

    draw_prob_train = np.vectorize(calc_dixon_coles_draw)(pred_scored_train, pred_conceded_train)
    draw_prob_test = np.vectorize(calc_dixon_coles_draw)(pred_scored_test, pred_conceded_test)

    # Inyectar variables Poisson al Dataset de los clasificadores finales
    X_train['predicted_xg_scored'] = pred_scored_train
    X_train['predicted_xg_conceded'] = pred_conceded_train
    X_train['poisson_draw_prob'] = draw_prob_train

    X_test['predicted_xg_scored'] = pred_scored_test
    X_test['predicted_xg_conceded'] = pred_conceded_test
    X_test['poisson_draw_prob'] = draw_prob_test
    
    feature_cols.extend(['predicted_xg_scored', 'predicted_xg_conceded', 'poisson_draw_prob'])

    # CAMBIO CRÍTICO: Usar TimeSeriesSplit en vez de StratifiedKFold
    cv_strategy = TimeSeriesSplit(n_splits=5)

    logger.info("Iniciando optimización de hiperparámetros (Optuna) minimizando Log-Loss...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, cv_strategy), n_trials=30)
    
    logger.info(f"Mejores parámetros XGBoost: {study.best_params}")

    xgb_best = XGBClassifier(
        **study.best_params,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        device='cuda'
    )
    
    pipeline_xgb = Pipeline([
        ('clf', xgb_best)
    ])
    
    logger.info("Entrenando modelo final (XGBoost puro en lugar de Stacking)...")
    
    # Para la calibración Isotonic sin Leakage ni errores de partición de sklearn,
    # dividimos cronológicamente el Train Set (75% Train, 25% Calibración)
    calib_idx = int(len(X_train) * 0.75)
    X_train_sub, X_calib = X_train.iloc[:calib_idx], X_train.iloc[calib_idx:]
    y_train_sub, y_calib = y_train.iloc[:calib_idx], y_train.iloc[calib_idx:]
    
    pipeline_xgb.fit(X_train_sub, y_train_sub)

    logger.info("Aplicando Calibración Probabilística cronológica (Isotonic)...")
    calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(pipeline_xgb), method='isotonic')
    calibrated_clf.fit(X_calib, y_calib)

    final_model = calibrated_clf
    
    # Encontrar umbral óptimo en el set de Calibración para evitar Leakage (Threshold Moving)
    y_prob_calib = final_model.predict_proba(X_calib)
    y_prob_calib = y_prob_calib / y_prob_calib.sum(axis=1, keepdims=True)
    # Asumimos una frecuencia natural de empate del ~26%
    draw_threshold = np.percentile(y_prob_calib[:, 1], 74)

    # Evaluar
    y_prob = final_model.predict_proba(X_test)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(y_prob, axis=1)
    # Forzamos predicción de empate si supera el umbral
    y_pred[y_prob[:, 1] >= draw_threshold] = 1

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    
    total_draws = sum(y_test == 1)
    predicted_as_draw = sum(y_pred == 1)
    correct_draws = sum((y_test == 1) & (y_pred == 1))
    draw_recall = (correct_draws / total_draws) * 100 if total_draws > 0 else 0
    draw_precision = (correct_draws / predicted_as_draw) * 100 if predicted_as_draw > 0 else 0

    logger.info("=== RESULTADOS ESTADÍSTICOS ===")
    logger.info(f"Accuracy Global: {acc:.4f}")
    logger.info(f"Log-Loss (Optimizando rentabilidad): {loss:.4f}")
    logger.info(f"Umbral de Empate Ajustado: {draw_threshold:.4f}")
    logger.info(f"Empates Correctos (Recall): {correct_draws} de {total_draws} ({draw_recall:.1f}%)")
    logger.info(f"Precisión en Empates: {correct_draws} correctos de {predicted_as_draw} predichos ({draw_precision:.1f}%)")
    
    # -------------------------------------
    # EVALUACIÓN FINANCIERA (Kelly Criterion)
    # -------------------------------------
    has_odds = 'odds_win' in df.columns
    if has_odds:
        logger.info("=== EVALUACIÓN FINANCIERA (Bankroll Simulation) ===")
        # Extraemos cuotas del Test Set
        odds_win = df['odds_win'].iloc[split_idx:].values
        odds_draw = df['odds_draw'].iloc[split_idx:].values
        odds_loss = df['odds_loss'].iloc[split_idx:].values
        
        bankroll = 1000.0  # Bankroll inicial
        bankroll_history = [bankroll]
        total_staked = 0.0
        bets_placed = 0
        bets_won = 0
        
        # Kelly Fraction (0.25 is standard conservative fractional Kelly)
        kelly_fraction = 0.25 
        max_stake_pct = 0.05  # Cap de apuesta por partido (5% del bankroll)
        
        for i in range(len(y_test)):
            p_loss, p_draw, p_win = y_prob[i]
            real_outcome = y_test.iloc[i] # 0: loss, 1: draw, 2: win
            
            # Cuotas para las 3 opciones (0: loss, 1: draw, 2: win)
            odds = [odds_loss[i], odds_draw[i], odds_win[i]]
            probs = [p_loss, p_draw, p_win]
            
            # Si alguna cuota es NaN, saltamos este partido
            if np.isnan(odds).any():
                continue
                
            # Calculamos Expected Value (EV) para cada opción
            # EV = (Probabilidad * Cuota) - 1
            evs = [ (probs[j] * odds[j]) - 1 for j in range(3) ]
            
            # Seleccionamos la opción con mayor EV
            best_choice = np.argmax(evs)
            best_ev = evs[best_choice]
            
            # Solo apostamos si EV > 5% (Value Bet Real)
            if best_ev > 0.05:
                # Formula de Kelly: f* = (p*odds - 1) / (odds - 1) = EV / (odds - 1)
                b = odds[best_choice] - 1
                kelly_pct = best_ev / b
                
                # Aplicamos Kelly fraccional y el cap máximo
                stake_pct = min(kelly_pct * kelly_fraction, max_stake_pct)
                
                # Mínimo del 0.5%
                if stake_pct < 0.005:
                    continue
                    
                stake = bankroll * stake_pct
                total_staked += stake
                bets_placed += 1
                
                if real_outcome == best_choice:
                    profit = stake * (odds[best_choice] - 1)
                    bankroll += profit
                    bets_won += 1
                else:
                    bankroll -= stake
                    
                bankroll_history.append(bankroll)
                
        yield_pct = ((bankroll - 1000.0) / total_staked) * 100 if total_staked > 0 else 0
        roi_pct = ((bankroll - 1000.0) / 1000.0) * 100
        
        logger.info(f"Capital Inicial: $1000.00 | Capital Final: ${bankroll:.2f}")
        logger.info(f"Apuestas Realizadas: {bets_placed} | Apuestas Ganadas: {bets_won} ({(bets_won/bets_placed)*100:.1f}% WinRate)")
        logger.info(f"Volumen Apostado (Turnover): ${total_staked:.2f}")
        logger.info(f"Yield (Beneficio Neto / Turnover): {yield_pct:.2f}%")
        logger.info(f"ROI del Capital Inicial: {roi_pct:.2f}%")
        
        if yield_pct > 0:
            logger.info("EL MODELO ES RENTABLE. (Tiene Edge real contra el mercado).")
        else:
            logger.warning("EL MODELO PIERDE DINERO. (Las cuotas del mercado son más eficientes que el modelo).")
    else:
        logger.warning("=== EVALUACIÓN FINANCIERA ===")
        logger.warning("No se encontraron cuotas en el dataset. No se puede calcular ROI/Yield.")

    logger.info("--- Muestra de Predicciones de Valor (Set de Prueba) ---")
    for i in range(min(5, len(y_test))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test.iloc[i]
        team_name = teams_test.iloc[i]
        opp_name = opponents_test.iloc[i]
        is_home_flag = X_test['is_home'].iloc[i]
        
        local = team_name if is_home_flag == 1 else opp_name
        visitante = opp_name if is_home_flag == 1 else team_name
        
        pred_xg = X_test['predicted_xg_scored'].iloc[i]
        pred_xg_opp = X_test['predicted_xg_conceded'].iloc[i]
        
        xg_local = pred_xg if is_home_flag == 1 else pred_xg_opp
        xg_visitante = pred_xg_opp if is_home_flag == 1 else pred_xg
        
        real_str = f"Victoria {team_name}" if real == 2 else "Empate" if real == 1 else f"Victoria {opp_name}"
        
        logger.info(f"Partido {i+1}: {local} ({xg_local:.2f} xG) vs {visitante} ({xg_visitante:.2f} xG)")
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
