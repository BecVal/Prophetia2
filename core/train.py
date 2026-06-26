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

# Configuración Quant
WHITELIST_LEAGUES = ['I1', 'E1', 'D2', 'SP2']
FILTER_BY_WHITELIST = True  # True: ignora partidos fuera de Whitelist en la simulación financiera

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
                    'team_squad_value', 'opp_squad_value', 'squad_value_diff',
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
    # EVALUACIÓN FINANCIERA (Bankroll Simulation)
    # -------------------------------------
    has_odds = 'odds_win' in df.columns
    if has_odds:
        logger.info("=== EVALUACIÓN FINANCIERA (Bankroll Simulation) ===")
        # Extraemos cuotas y variables del Test Set
        odds_win = df['odds_win'].iloc[split_idx:].values
        odds_draw = df['odds_draw'].iloc[split_idx:].values
        odds_loss = df['odds_loss'].iloc[split_idx:].values
        dates = df['match_date'].iloc[split_idx:].values
        competitions = df['competition'].iloc[split_idx:].values if 'competition' in df.columns else np.array([None]*len(y_test))
        
        liquid_bankroll = 1000.0  # Bankroll líquido disponible
        bankroll_history = [liquid_bankroll]
        daily_multipliers = []  # Para Bootstrapping Monte Carlo
        total_staked = 0.0
        bets_placed = 0
        bets_won = 0
        
        kelly_fraction = 0.25 
        max_stake_pct = 0.05  # Cap de apuesta por partido (5% del bankroll líquido)
        
        # Agrupar índices por fecha (timestamp) para evitar Sequential Loop Bug
        unique_dates = np.unique(dates)
        league_stats = {}

        # Preparación de CLV (Closing Line Value)
        # TODO: Para el futuro, calcular 'clv = open_odds - close_odds' y usarlo como feature.
        
        for current_date in sorted(unique_dates):
            start_of_day_bankroll = liquid_bankroll
            day_indices = np.where(dates == current_date)[0]
            daily_bets = []
            
            for i in day_indices:
                comp = competitions[i]
                if FILTER_BY_WHITELIST and comp not in WHITELIST_LEAGUES:
                    continue
                    
                p_loss, p_draw, p_win = y_prob[i]
                real_outcome = y_test.iloc[i] 
                odds = [odds_loss[i], odds_draw[i], odds_win[i]]
                probs = [p_loss, p_draw, p_win]
                
                if np.isnan(odds).any():
                    continue
                    
                evs = [ (probs[j] * odds[j]) - 1 for j in range(3) ]
                
                # Check for Dutching / Doble Oportunidad (Local y Empate)
                ev_local = evs[2]
                ev_draw = evs[1]
                
                bet_type = 'single'
                best_choice = np.argmax(evs)
                best_ev = evs[best_choice]
                secondary_choice = None
                
                # Dutching logic if both Local and Draw have EV > 0.05
                if ev_local > 0.05 and ev_draw > 0.05:
                    bet_type = 'dutching'
                    implied_prob_1 = 1 / odds[2]
                    implied_prob_X = 1 / odds[1]
                    total_implied = implied_prob_1 + implied_prob_X
                    combined_odds = 1 / total_implied
                    best_ev = ((probs[2] + probs[1]) * combined_odds) - 1
                    best_choice = 2
                    secondary_choice = 1
                
                if best_ev > 0.05:
                    daily_bets.append({
                        'index': i,
                        'best_choice': best_choice,
                        'secondary_choice': secondary_choice,
                        'best_ev': best_ev,
                        'odds': odds,
                        'probs': probs,
                        'bet_type': bet_type,
                        'real_outcome': real_outcome,
                        'comp': comp
                    })
                    
            # Ordenar las apuestas del día por EV descendente para priorizar la asignación de capital
            daily_bets.sort(key=lambda x: x['best_ev'], reverse=True)
            
            day_profit = 0.0
            day_staked = 0.0
            
            for bet in daily_bets:
                best_choice = bet['best_choice']
                secondary_choice = bet['secondary_choice']
                odds = bet['odds']
                bet_type = bet['bet_type']
                comp = bet['comp']
                real_outcome = bet['real_outcome']
                best_ev = bet['best_ev']
                
                if bet_type == 'single':
                    b = odds[best_choice] - 1
                    kelly_pct = best_ev / b
                    
                    # Conservador en cuotas bajas para mitigar trampa isotónica
                    if odds[best_choice] < 1.30:
                        kelly_pct = min(kelly_pct, 0.01) # Cap 1% Kelly
                        
                    stake_pct = min(kelly_pct * kelly_fraction, max_stake_pct)
                    if stake_pct < 0.005: continue
                        
                    stake = liquid_bankroll * stake_pct
                    if liquid_bankroll - stake < 0:
                        stake = liquid_bankroll
                        if stake <= 0: break
                            
                    liquid_bankroll -= stake
                    day_staked += stake
                    total_staked += stake
                    bets_placed += 1
                    
                    if comp not in league_stats:
                        league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0}
                    league_stats[comp]['bets'] += 1
                    league_stats[comp]['staked'] += stake
                    
                    if real_outcome == best_choice:
                        profit = stake * odds[best_choice]
                        day_profit += profit
                        bets_won += 1
                        league_stats[comp]['profit'] += (profit - stake)
                        league_stats[comp]['won'] += 1
                    else:
                        league_stats[comp]['profit'] -= stake
                        
                elif bet_type == 'dutching':
                    implied_prob_1 = 1 / odds[best_choice]
                    implied_prob_X = 1 / odds[secondary_choice]
                    total_implied = implied_prob_1 + implied_prob_X
                    combined_odds = 1 / total_implied
                    
                    b = combined_odds - 1
                    kelly_pct = best_ev / b if b > 0 else 0
                    if combined_odds < 1.30:
                        kelly_pct = min(kelly_pct, 0.01)
                        
                    stake_pct = min(kelly_pct * kelly_fraction, max_stake_pct)
                    if stake_pct < 0.005: continue
                        
                    total_dutch_stake = liquid_bankroll * stake_pct
                    if liquid_bankroll - total_dutch_stake < 0:
                        total_dutch_stake = liquid_bankroll
                        if total_dutch_stake <= 0: break
                            
                    stake_1 = total_dutch_stake * (implied_prob_1 / total_implied)
                    stake_X = total_dutch_stake * (implied_prob_X / total_implied)
                    
                    liquid_bankroll -= total_dutch_stake
                    day_staked += total_dutch_stake
                    total_staked += total_dutch_stake
                    bets_placed += 1
                    
                    if comp not in league_stats:
                        league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0}
                    league_stats[comp]['bets'] += 1
                    league_stats[comp]['staked'] += total_dutch_stake
                    
                    if real_outcome == best_choice:
                        profit = stake_1 * odds[best_choice]
                        day_profit += profit
                        bets_won += 1
                        league_stats[comp]['profit'] += (profit - total_dutch_stake)
                        league_stats[comp]['won'] += 1
                    elif real_outcome == secondary_choice:
                        profit = stake_X * odds[secondary_choice]
                        day_profit += profit
                        bets_won += 1
                        league_stats[comp]['profit'] += (profit - total_dutch_stake)
                        league_stats[comp]['won'] += 1
                    else:
                        league_stats[comp]['profit'] -= total_dutch_stake
            
            # Reintegrar ganancias al capital disponible
            liquid_bankroll += day_profit
            bankroll_history.append(liquid_bankroll)
            
            if day_staked > 0:
                daily_multiplier = liquid_bankroll / start_of_day_bankroll
                daily_multipliers.append(daily_multiplier)

        yield_pct = ((liquid_bankroll - 1000.0) / total_staked) * 100 if total_staked > 0 else 0
        roi_pct = ((liquid_bankroll - 1000.0) / 1000.0) * 100
        
        logger.info(f"Capital Inicial: $1000.00 | Capital Final Líquido: ${liquid_bankroll:.2f}")
        logger.info(f"Apuestas Realizadas: {bets_placed} | Apuestas Ganadas: {bets_won} ({(bets_won/bets_placed)*100:.1f}% WinRate)")
        logger.info(f"Volumen Apostado (Turnover): ${total_staked:.2f}")
        logger.info(f"Yield (Beneficio Neto / Turnover): {yield_pct:.2f}%")
        logger.info(f"ROI del Capital Inicial: {roi_pct:.2f}%")
        
        # --- DESGLOSE POR LIGAS ---
        logger.info("=== RENDIMIENTO POR LIGA ===")
        for comp, stats in sorted(league_stats.items(), key=lambda x: x[1]['profit'], reverse=True):
            if stats['bets'] > 0:
                l_winrate = (stats['won'] / stats['bets']) * 100
                l_yield = (stats['profit'] / stats['staked']) * 100
                logger.info(f"Liga {comp}: {stats['bets']} apuestas | WinRate: {l_winrate:.1f}% | Yield: {l_yield:.2f}% | Profit: ${stats['profit']:.2f}")
        
        
        if yield_pct > 0:
            logger.info("EL MODELO ES RENTABLE. (Tiene Edge real contra el mercado).")
        else:
            logger.warning("EL MODELO PIERDE DINERO. (Las cuotas del mercado son más eficientes que el modelo).")
            
        # --- MONTE CARLO BOOTSTRAPPING ---
        if len(daily_multipliers) > 10:
            logger.info("=== PRUEBA DE RESISTENCIA (MONTE CARLO) ===")
            logger.info(f"Ejecutando 10,000 simulaciones de bootstrapping sobre {len(daily_multipliers)} bloques diarios...")
            import random
            
            n_sims = 10000
            ruin_count = 0
            max_drawdowns = []
            final_capitals = []
            
            for _ in range(n_sims):
                sim_bankroll = 1000.0
                peak = 1000.0
                max_dd = 0.0
                is_ruined = False
                
                # Remuestreo con reemplazo (Block Bootstrapping)
                sampled_multipliers = random.choices(daily_multipliers, k=len(daily_multipliers))
                
                for mult in sampled_multipliers:
                    sim_bankroll *= mult
                    
                    if sim_bankroll > peak:
                        peak = sim_bankroll
                    
                    dd = (peak - sim_bankroll) / peak
                    if dd > max_dd:
                        max_dd = dd
                        
                    if sim_bankroll <= 10.0:  # Umbral de ruina técnica
                        is_ruined = True
                        break
                        
                if is_ruined:
                    ruin_count += 1
                    max_drawdowns.append(1.0)
                    final_capitals.append(0.0)
                else:
                    max_drawdowns.append(max_dd)
                    final_capitals.append(sim_bankroll)
            
            por = (ruin_count / n_sims) * 100
            avg_mdd = np.mean(max_drawdowns) * 100
            p95_mdd = np.percentile(max_drawdowns, 95) * 100
            median_cap = np.median(final_capitals)
            
            logger.info(f"Probabilidad de Ruina (PoR): {por:.2f}%")
            logger.info(f"Maximum Drawdown Promedio (MDD): {avg_mdd:.2f}%")
            logger.info(f"MDD Tail Risk (Percentil 95): {p95_mdd:.2f}%")
            logger.info(f"Capital Final Mediano Esperado: ${median_cap:.2f}")
            
            if por > 1.0:
                logger.warning("ALERTA QUANT: La Probabilidad de Ruina supera el 1%. Considera reducir el `kelly_fraction` o el `max_stake_pct`.")
            else:
                logger.info("VALIDACIÓN QUANT: Estrategia de bankroll robusta frente a la varianza extrema.")

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
