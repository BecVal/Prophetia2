import os
import json
import numpy as np
import pandas as pd
import random
import optuna
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'simulate_bankroll')


# Configuración Quant
FILTER_BY_WHITELIST = True  # True: ignora partidos fuera de Whitelist en la simulación financiera
WHITELIST_LEAGUES = ['F1', 'F2', 'SP1', 'G1', 'B1', 'P1', 'SP2', 'SC0', 'D1', 'D2', 'E1']
ENABLE_VALUE_TRAP_FILTER = True # True: Activa el filtro contra Winner's Curse (descarte por asimetría de información)

# --- CONFIGURACIÓN DE OPTIMIZACIÓN ---
# Opciones: 'NONE', 'ALL', 'WHITELIST', o una liga específica como 'I1'
OPTIMIZATION_MODE = 'WHITELIST'
OPTUNA_TRIALS = 1000
OPTIMIZED_PARAMS_FILE = '../data/processed/optimal_bankroll_params.json'

# Diccionarios de riesgo por liga iniciales/por defecto
KELLY_FRACTIONS = {'D2': 0.03, 'I1': 0.01, 'SP1': 0.03, 'F2': 0.02, 'G1': 0.01, 'D1': 0.02, 'T1': 0.03, 'F1': 0.02, 'E1': 0.02, 'N1': 0.01, 'SP2': 0.01, 'P1': 0.01, 'DEFAULT': 0.015}
EV_THRESHOLDS = {'D2': 0.015, 'I1': 0.02, 'SP1': 0.01, 'F2': 0.015, 'G1': 0.02, 'D1': 0.015, 'T1': 0.01, 'F1': 0.015, 'E1': 0.015, 'N1': 0.02, 'SP2': 0.02, 'P1': 0.02, 'DEFAULT': 0.015}

MAX_STAKE_PCT = 0.6  # Cap reducido al 3% para mayor seguridad con el nuevo filtro

# Parámetros Institucionales/Fricción Añadidos
TAX_RETENTION_RATE = 0.0075  # Retención del 0.75% sobre ganancias netas (Polymarket)
MARKET_BLEND_ALPHA = 0.85  # Peso del modelo vs mercado (85% modelo, 15% mercado)
EXPECTED_CLV_DROP = 0.015  # Penalización por slippage esperado del CLV (-1.5%)
MAX_BET_LIQUIDITY = {      # Límites de liquidez absolutos en USD o Unidad Base
    'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
    'D2': 2000.0, 'F2': 2000.0,
    'T1': 2000.0,
    'DEFAULT': 2000.0
}

def load_optimized_params():
    global KELLY_FRACTIONS, EV_THRESHOLDS
    if os.path.exists(OPTIMIZED_PARAMS_FILE):
        try:
            with open(OPTIMIZED_PARAMS_FILE, 'r') as f:
                data = json.load(f)
                if 'KELLY_FRACTIONS' in data:
                    KELLY_FRACTIONS.update(data['KELLY_FRACTIONS'])
                if 'EV_THRESHOLDS' in data:
                    EV_THRESHOLDS.update(data['EV_THRESHOLDS'])
            logger.info("Parámetros optimizados cargados desde archivo local.")
        except Exception as e:
            logger.error(f"Error al cargar parámetros optimizados: {e}")
    else:
        logger.info("No se encontró archivo de parámetros optimizados. Se usarán los predeterminados.")

def save_optimized_params():
    data = {
        'KELLY_FRACTIONS': KELLY_FRACTIONS,
        'EV_THRESHOLDS': EV_THRESHOLDS
    }
    try:
        os.makedirs(os.path.dirname(OPTIMIZED_PARAMS_FILE), exist_ok=True)
        with open(OPTIMIZED_PARAMS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Parámetros óptimos guardados exitosamente en {OPTIMIZED_PARAMS_FILE}")
    except Exception as e:
        logger.error(f"Error al guardar parámetros optimizados: {e}")

def evaluate_league_params(df_league, ev_thresh, kelly_fraction):
    """ Función rápida para simular el bankroll de una sola liga de forma independiente """
    liquid_bankroll = 1000.0
    historical_peak = liquid_bankroll
    historical_mdd = 0.0
    
    dates = df_league['match_date'].values
    unique_dates = np.unique(dates)
    
    odds_win = df_league['odds_win'].values
    odds_draw = df_league['odds_draw'].values
    odds_loss = df_league['odds_loss'].values
    y_test = df_league['outcome'].values
    y_prob = df_league[['prob_loss', 'prob_draw', 'prob_win']].values
    
    has_pred_clv = all(c in df_league.columns for c in ['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win'])
    if has_pred_clv:
        pred_clv_vals = df_league[['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win']].values
    
    for current_date in sorted(unique_dates):
        day_indices = np.where(dates == current_date)[0]
        daily_bets = []
        
        for idx_local in day_indices:
            p_loss, p_draw, p_win = y_prob[idx_local]
            odds = [odds_loss[idx_local], odds_draw[idx_local], odds_win[idx_local]]
            if np.isnan(odds).any(): continue
            
            market_implied = [1 / odds[j] for j in range(3)]
            margin = sum(market_implied)
            market_probs = [p / margin for p in market_implied]
            
            blended_probs = []
            for j in range(3):
                divergence = abs([p_loss, p_draw, p_win][j] - market_probs[j])
                dynamic_alpha = MARKET_BLEND_ALPHA
                if divergence > 0.20: dynamic_alpha = 0.30
                elif divergence > 0.15: dynamic_alpha = 0.50
                elif divergence > 0.10: dynamic_alpha = 0.70
                blended_probs.append((dynamic_alpha * [p_loss, p_draw, p_win][j]) + ((1 - dynamic_alpha) * market_probs[j]))
            
            net_odds = [1 + (odds[j] - 1) * (1 - TAX_RETENTION_RATE) for j in range(3)]
            evs = [ (blended_probs[j] * net_odds[j]) - 1 - EXPECTED_CLV_DROP for j in range(3) ]
            
            if ENABLE_VALUE_TRAP_FILTER:
                for j in range(3):
                    divergence = abs([p_loss, p_draw, p_win][j] - market_probs[j])
                    # Umbrales relajados: 15% para súper favoritos, 25% para general
                    if [p_loss, p_draw, p_win][j] > 0.65 and divergence > 0.15:
                        evs[j] = -1.0
                    elif divergence > 0.25:
                        evs[j] = -1.0
                    # NUEVO: Asimetría de información en Favoritos Claros (Value Trap)
                    # Si el mercado cree que hay >50% prob, y nuestro EV es marginal (<10%), es una trampa.
                    elif market_probs[j] > 0.50 and 0 < evs[j] < 0.10:
                        evs[j] = -1.0
                        
            best_choice = np.argmax(evs)
            best_ev = evs[best_choice]
            
            MIN_EXPECTED_CLV = 0.005
            if has_pred_clv:
                if pred_clv_vals[idx_local][best_choice] < MIN_EXPECTED_CLV:
                    best_ev = -1.0
                    
            bet_type = 'single'
            secondary_choice = None
            ev_local = evs[2]
            ev_draw = evs[1]
            ev_away = evs[0]
            
            if ev_local > ev_thresh and ev_draw > ev_thresh:
                if (not has_pred_clv) or (pred_clv_vals[idx_local][2] >= MIN_EXPECTED_CLV and pred_clv_vals[idx_local][1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    implied_prob_1 = 1 / odds[2]
                    implied_prob_X = 1 / odds[1]
                    total_implied = implied_prob_1 + implied_prob_X
                    combined_odds = 1 / total_implied
                    blended_prob_1X = blended_probs[2] + blended_probs[1]
                    net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_1X * net_combined_odds) - 1 - EXPECTED_CLV_DROP
                    best_choice = 2
                    secondary_choice = 1
            elif ev_away > ev_thresh and ev_draw > ev_thresh:
                if (not has_pred_clv) or (pred_clv_vals[idx_local][0] >= MIN_EXPECTED_CLV and pred_clv_vals[idx_local][1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    implied_prob_2 = 1 / odds[0]
                    implied_prob_X = 1 / odds[1]
                    total_implied = implied_prob_2 + implied_prob_X
                    combined_odds = 1 / total_implied
                    blended_prob_X2 = blended_probs[0] + blended_probs[1]
                    net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_X2 * net_combined_odds) - 1 - EXPECTED_CLV_DROP
                    best_choice = 0
                    secondary_choice = 1
            
            if best_ev > ev_thresh:
                daily_bets.append({
                    'best_choice': best_choice,
                    'secondary_choice': secondary_choice,
                    'best_ev': best_ev,
                    'odds': odds,
                    'bet_type': bet_type,
                    'real_outcome': y_test[idx_local]
                })
        
        daily_bets.sort(key=lambda x: x['best_ev'], reverse=True)
        day_profit = 0.0
        
        for bet in daily_bets:
            best_choice = bet['best_choice']
            secondary_choice = bet['secondary_choice']
            odds = bet['odds']
            bet_type = bet['bet_type']
            best_ev = bet['best_ev']
            real_outcome = bet['real_outcome']
            
            if bet_type == 'single':
                net_odd = 1 + (odds[best_choice] - 1) * (1 - TAX_RETENTION_RATE)
                b = net_odd - 1
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if odds[best_choice] < 1.30: kelly_pct = min(kelly_pct, 0.01)
                
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.001: continue
                
                stake = liquid_bankroll * stake_pct
                if liquid_bankroll - stake < 0:
                    stake = liquid_bankroll
                    if stake <= 0: break
                
                liquid_bankroll -= stake
                if real_outcome == best_choice:
                    gross_profit = stake * (odds[best_choice] - 1)
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE)
                    day_profit += stake + net_profit
            
            elif bet_type == 'dutching':
                implied_prob_1 = 1 / odds[best_choice]
                implied_prob_X = 1 / odds[secondary_choice]
                total_implied = implied_prob_1 + implied_prob_X
                combined_odds = 1 / total_implied
                net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                b = net_combined_odds - 1
                
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if combined_odds < 1.30: kelly_pct = min(kelly_pct, 0.01)
                
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.001: continue
                
                total_dutch_stake = liquid_bankroll * stake_pct
                if liquid_bankroll - total_dutch_stake < 0:
                    total_dutch_stake = liquid_bankroll
                    if total_dutch_stake <= 0: break
                
                stake_1 = total_dutch_stake * (implied_prob_1 / total_implied)
                stake_X = total_dutch_stake * (implied_prob_X / total_implied)
                
                liquid_bankroll -= total_dutch_stake
                if real_outcome == best_choice:
                    profit_1 = stake_1 * (odds[best_choice] - 1)
                    net_profit_1 = profit_1 * (1.0 - TAX_RETENTION_RATE) if profit_1 > 0 else profit_1
                    net_profit = net_profit_1 - stake_X
                    day_profit += total_dutch_stake + net_profit
                elif real_outcome == secondary_choice:
                    profit_X = stake_X * (odds[secondary_choice] - 1)
                    net_profit_X = profit_X * (1.0 - TAX_RETENTION_RATE) if profit_X > 0 else profit_X
                    net_profit = net_profit_X - stake_1
                    day_profit += total_dutch_stake + net_profit
                    
        liquid_bankroll += day_profit
        if liquid_bankroll > historical_peak:
            historical_peak = liquid_bankroll
        current_dd = (historical_peak - liquid_bankroll) / historical_peak
        if current_dd > historical_mdd:
            historical_mdd = current_dd
            
        if liquid_bankroll <= 0:
            break
            
    roi = (liquid_bankroll - 1000.0) / 1000.0
    return roi, historical_mdd

def optimize_league(df, league_name):
    logger.info(f"Iniciando Optuna para {league_name} ({OPTUNA_TRIALS} trials)...")
    df_league = df[df['competition'] == league_name].copy()
    if df_league.empty:
        logger.warning(f"No hay datos para {league_name}.")
        return None, None
        
    def objective(trial):
        ev_thresh = trial.suggest_float('ev_thresh', 0.020, 0.150)
        # Configuración del kelly máximo para Optuna ajustado a ser REALMENTE conservador
        # Al limitar el rango superior a 0.12 (12% del Kelly óptimo), evitamos el overbetting
        # que causaba pérdidas en el Test Set.
        kelly_fraction = trial.suggest_float('kelly_fraction', 0.01, 0.25)
        
        roi, mdd = evaluate_league_params(df_league, ev_thresh, kelly_fraction)
        
        if roi <= 0.0:
            return roi - (mdd * 2.0) # Penalizar pérdida de forma continua
            
        # --- NUEVA LÓGICA DE UTILIDAD (Mean-Variance adaptada) ---
        # Un penalty_factor de 10.0 destruía el score de ligas rentables con varianza normal.
        # Las ligas individuales SIEMPRE tienen un MDD más alto que el portafolio global.
        # Un penalty de 2.0 logra el equilibrio perfecto: penaliza el exceso de drawdown 
        # sin forzar a Optuna a preferir un Score de 0 (es decir, sin anular la liga).
        
        penalty_factor = 1.15  # Nivel de aversión al riesgo balanceado
        score = roi - penalty_factor * (mdd ** 2)
        
        if mdd > 0.35:
            score -= (mdd - 0.35) * 1  # Penalización más holgada para ligas individuales
            
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"Mejores parámetros para {league_name} encontrados: EV={best_params['ev_thresh']:.4f}, Kelly={best_params['kelly_fraction']:.4f} | Score Ajustado: {best_value:.2f}")
    
    return best_params['ev_thresh'], best_params['kelly_fraction']

def run_simulation():
    PREDICTIONS_PATH = '../data/processed/test_predictions.parquet'
    
    if not os.path.exists(PREDICTIONS_PATH):
        logger.error(f"No se encontró el archivo de predicciones en {PREDICTIONS_PATH}.")
        logger.error("Asegúrate de ejecutar 'python core/train.py' primero para generar las predicciones.")
        return

    logger.info(f"Cargando predicciones del set de prueba desde: {PREDICTIONS_PATH}...")
    df = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    
    has_odds = 'odds_win' in df.columns
    if not has_odds:
        logger.warning("=== EVALUACIÓN FINANCIERA ===")
        logger.warning("No se encontraron cuotas en el dataset de predicciones. No se puede calcular ROI/Yield.")
        return

    # 1. OPTIMIZACIÓN O CARGA DE PARÁMETROS
    if OPTIMIZATION_MODE != 'NONE':
        TRAIN_PREDS_PATH = '../data/processed/train_predictions.parquet'
        if not os.path.exists(TRAIN_PREDS_PATH):
            logger.error("No se puede optimizar: falta train_predictions.parquet")
        else:
            logger.info(f"Cargando predicciones del set de TRAIN para Optimización: {TRAIN_PREDS_PATH}...")
            df_train_opt = pd.read_parquet(TRAIN_PREDS_PATH, engine='fastparquet')
            
            leagues_to_optimize = []
            if OPTIMIZATION_MODE == 'ALL':
                leagues_to_optimize = df_train_opt['competition'].unique()
            elif OPTIMIZATION_MODE == 'WHITELIST':
                leagues_to_optimize = [L for L in df_train_opt['competition'].unique() if L in WHITELIST_LEAGUES]
            else:
                leagues_to_optimize = [OPTIMIZATION_MODE] 
                
            logger.info(f"Modo Optimización Activado: Optimizando {len(leagues_to_optimize)} ligas en el set de ENTRENAMIENTO (OOF).")
            for comp in leagues_to_optimize:
                if FILTER_BY_WHITELIST and comp not in WHITELIST_LEAGUES:
                    continue
                ev_opt, kelly_opt = optimize_league(df_train_opt, comp)
                if ev_opt is not None:
                    EV_THRESHOLDS[comp] = ev_opt
                    KELLY_FRACTIONS[comp] = kelly_opt
                    
            # Guardar resultados para futuro uso
            save_optimized_params()
    else:
        # Cargar si existe el archivo
        load_optimized_params()

    logger.info("=== EVALUACIÓN FINANCIERA (Bankroll Simulation) ===")
    
    # ------------------ NUEVO: BRIER SCORE GLOBAL ------------------
    df_eval = df.copy()
    if FILTER_BY_WHITELIST:
        df_eval = df_eval[df_eval['competition'].isin(WHITELIST_LEAGUES)]
    
    if not df_eval.empty:
        # Probabilidades y resultados
        y_prob_eval = df_eval[['prob_loss', 'prob_draw', 'prob_win']].values
        y_true_eval = df_eval['outcome'].values
        
        # One-hot encoding manual para y_true
        y_true_oh = np.zeros_like(y_prob_eval)
        for idx_val, val in enumerate(y_true_eval):
            if not np.isnan(val) and val in [0, 1, 2]:
                y_true_oh[idx_val, int(val)] = 1
                
        # Brier Score (Multi-class)
        brier_score = np.mean(np.sum((y_prob_eval - y_true_oh)**2, axis=1))
        
        # Log Loss (Multi-class)
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob_eval, eps, 1 - eps)
        log_loss_val = -np.mean(np.sum(y_true_oh * np.log(y_prob_clipped), axis=1))
        
        # Descomposición Brier Score (aplanando a binario)
        p_flat = y_prob_eval.flatten()
        y_flat = y_true_oh.flatten()
        
        bins = np.linspace(0, 1, 21)
        digitized = np.digitize(p_flat, bins)
        
        reliability = 0.0
        resolution = 0.0
        base_rate = np.mean(y_flat)
        uncertainty = base_rate * (1 - base_rate)
        N_tot = len(p_flat)
        
        for b in range(1, len(bins)):
            idx_b = np.where(digitized == b)[0]
            if len(idx_b) > 0:
                p_k = np.mean(p_flat[idx_b])
                f_k = np.mean(y_flat[idx_b])
                n_k = len(idx_b)
                reliability += (n_k / N_tot) * (p_k - f_k)**2
                resolution += (n_k / N_tot) * (f_k - base_rate)**2
                
        logger.info("=== MÉTRICAS DE CALIBRACIÓN GLOBAL ===")
        logger.info(f"Log Loss: {log_loss_val:.4f}")
        logger.info(f"Brier Score Global: {brier_score:.4f}")
        logger.info(f" -> Reliability (menor es mejor): {reliability * 2:.4f}")
        logger.info(f" -> Resolution (mayor es mejor): {resolution * 2:.4f}")
        logger.info(f" -> Uncertainty: {uncertainty * 2:.4f}")
    # ---------------------------------------------------------------
    
    odds_win = df['odds_win'].values
    odds_draw = df['odds_draw'].values
    odds_loss = df['odds_loss'].values
    
    has_closing_odds = all(c in df.columns for c in ['closing_odds_win', 'closing_odds_draw', 'closing_odds_loss'])
    if has_closing_odds:
        c_odds_win = df['closing_odds_win'].values
        c_odds_draw = df['closing_odds_draw'].values
        c_odds_loss = df['closing_odds_loss'].values
    else:
        logger.warning("No se detectaron columnas de 'closing_odds_*' en el dataset. Se omitirá el cálculo de CLV.")
        
    dates = df['match_date'].values
    competitions = df['competition'].values if 'competition' in df.columns else np.array([None]*len(df))
    
    y_test = df['outcome'].values
    y_prob = df[['prob_loss', 'prob_draw', 'prob_win']].values
    
    liquid_bankroll = 1000.0  # Bankroll líquido disponible
    bankroll_history = [liquid_bankroll]
    daily_multipliers = []  # Para Bootstrapping Monte Carlo
    total_staked = 0.0
    bets_placed = 0
    bets_won = 0
    total_analyzed_matches = 0
    
    total_expected_profit = 0.0 
    historical_peak = liquid_bankroll
    historical_mdd = 0.0
    clv_list = []
    
    gross_profit_sum = 0.0
    gross_loss_sum = 0.0
    avg_odds_list = []
    
    unique_dates = np.unique(dates)
    league_stats = {}
    
    placed_bets_history = []

    for current_date in sorted(unique_dates):
        start_of_day_bankroll = liquid_bankroll
        day_indices = np.where(dates == current_date)[0]
        daily_bets = []
        
        for i in day_indices:
            comp = competitions[i]
            if FILTER_BY_WHITELIST and comp not in WHITELIST_LEAGUES:
                continue
                
            p_loss, p_draw, p_win = y_prob[i]
            real_outcome = y_test[i] 
            odds = [odds_loss[i], odds_draw[i], odds_win[i]]
            probs = [p_loss, p_draw, p_win]
            
            if np.isnan(odds).any():
                continue
                
            total_analyzed_matches += 1
                
            market_implied = [1 / odds[j] for j in range(3)]
            margin = sum(market_implied)
            market_probs = [p / margin for p in market_implied]
            
            blended_probs = []
            for j in range(3):
                divergence = abs(probs[j] - market_probs[j])
                dynamic_alpha = MARKET_BLEND_ALPHA
                if divergence > 0.20:
                    dynamic_alpha = 0.30  
                elif divergence > 0.15:
                    dynamic_alpha = 0.50
                elif divergence > 0.10:
                    dynamic_alpha = 0.70  
                blended_probs.append((dynamic_alpha * probs[j]) + ((1 - dynamic_alpha) * market_probs[j]))
            
            net_odds = [1 + (odds[j] - 1) * (1 - TAX_RETENTION_RATE) for j in range(3)]
            evs = [ (blended_probs[j] * net_odds[j]) - 1 - EXPECTED_CLV_DROP for j in range(3) ]
            
            if ENABLE_VALUE_TRAP_FILTER:
                for j in range(3):
                    divergence = abs(probs[j] - market_probs[j])
                    # Umbrales relajados: 15% para súper favoritos, 25% para general
                    if probs[j] > 0.65 and divergence > 0.15:
                        evs[j] = -1.0
                    elif divergence > 0.25:
                        evs[j] = -1.0
                        
            league_ev_thresh = EV_THRESHOLDS.get(comp, EV_THRESHOLDS.get('DEFAULT', 0.015))
            league_kelly = KELLY_FRACTIONS.get(comp, KELLY_FRACTIONS.get('DEFAULT', 0.015))
            
            has_pred_clv = all(c in df.columns for c in ['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win'])
            pred_clv = [
                df['pred_clv_loss'].iloc[i] if has_pred_clv else 0.0,
                df['pred_clv_draw'].iloc[i] if has_pred_clv else 0.0,
                df['pred_clv_win'].iloc[i] if has_pred_clv else 0.0
            ]
            
            ev_local = evs[2]
            ev_draw = evs[1]
            ev_away = evs[0]
            
            bet_type = 'single'
            best_choice = np.argmax(evs)
            best_ev = evs[best_choice]
            secondary_choice = None
            
            MIN_EXPECTED_CLV = 0.005
            if has_pred_clv and pred_clv[best_choice] < MIN_EXPECTED_CLV:
                best_ev = -1.0 
            
            if ev_local > league_ev_thresh and ev_draw > league_ev_thresh:
                if (not has_pred_clv) or (pred_clv[2] >= MIN_EXPECTED_CLV and pred_clv[1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    implied_prob_1 = 1 / odds[2]
                    implied_prob_X = 1 / odds[1]
                    total_implied = implied_prob_1 + implied_prob_X
                    combined_odds = 1 / total_implied
                    
                    blended_prob_1X = blended_probs[2] + blended_probs[1]
                    net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_1X * net_combined_odds) - 1 - EXPECTED_CLV_DROP
                    
                    best_choice = 2
                    secondary_choice = 1
            elif ev_away > league_ev_thresh and ev_draw > league_ev_thresh:
                if (not has_pred_clv) or (pred_clv[0] >= MIN_EXPECTED_CLV and pred_clv[1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    implied_prob_2 = 1 / odds[0]
                    implied_prob_X = 1 / odds[1]
                    total_implied = implied_prob_2 + implied_prob_X
                    combined_odds = 1 / total_implied
                    
                    blended_prob_X2 = blended_probs[0] + blended_probs[1]
                    net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_X2 * net_combined_odds) - 1 - EXPECTED_CLV_DROP
                    
                    best_choice = 0
                    secondary_choice = 1
            
            if best_ev > league_ev_thresh:
                daily_bets.append({
                    'kelly_fraction': league_kelly,
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
            kelly_fraction = bet['kelly_fraction']
            
            if bet_type == 'single':
                net_odd = 1 + (odds[best_choice] - 1) * (1 - TAX_RETENTION_RATE)
                b = net_odd - 1
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                
                if odds[best_choice] < 1.30:
                    kelly_pct = min(kelly_pct, 0.01) 
                    
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.001: continue  
                    
                stake = liquid_bankroll * stake_pct
                
                max_liquidity = MAX_BET_LIQUIDITY.get(comp, MAX_BET_LIQUIDITY.get('DEFAULT', 200.0))
                if stake > max_liquidity:
                    stake = max_liquidity

                if liquid_bankroll - stake < 0:
                    stake = liquid_bankroll
                    if stake <= 0: break
                        
                liquid_bankroll -= stake
                day_staked += stake
                total_staked += stake
                bets_placed += 1
                total_expected_profit += (stake * best_ev)
                avg_odds_list.append(odds[best_choice])
                
                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0, 'clv_list': []}

                if has_closing_odds:
                    idx = bet['index']
                    c_loss = c_odds_loss[idx]
                    c_draw = c_odds_draw[idx]
                    c_win = c_odds_win[idx]
                    
                    if not np.isnan(c_loss) and not np.isnan(c_draw) and not np.isnan(c_win) and c_loss > 0 and c_draw > 0 and c_win > 0:
                        c_margin = (1/c_loss) + (1/c_draw) + (1/c_win)
                        fair_closing_odds = [
                            1 / ((1/c_loss) / c_margin),
                            1 / ((1/c_draw) / c_margin),
                            1 / ((1/c_win) / c_margin)
                        ]
                        
                        fair_c_odd = fair_closing_odds[best_choice]
                        true_clv = (odds[best_choice] / fair_c_odd) - 1
                        clv_list.append(true_clv)
                        league_stats[comp]['clv_list'].append(true_clv)
                
                league_stats[comp]['bets'] += 1
                league_stats[comp]['staked'] += stake
                
                if real_outcome == best_choice:
                    gross_profit = stake * (odds[best_choice] - 1)
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE)
                    total_return = stake + net_profit
                    
                    day_profit += total_return
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                    gross_profit_sum += net_profit
                else:
                    net_profit = -stake
                    league_stats[comp]['profit'] -= stake
                    gross_loss_sum += stake
                    
                placed_bets_history.append({
                    'ev': best_ev,
                    'prob': blended_probs[best_choice],
                    'odds': odds[best_choice],
                    'stake': stake,
                    'is_win': int(real_outcome == best_choice),
                    'net_profit': net_profit
                })
                    
            elif bet_type == 'dutching':
                implied_prob_1 = 1 / odds[best_choice]
                implied_prob_X = 1 / odds[secondary_choice]
                total_implied = implied_prob_1 + implied_prob_X
                combined_odds = 1 / total_implied
                
                net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                b = net_combined_odds - 1
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if combined_odds < 1.30:
                    kelly_pct = min(kelly_pct, 0.01)
                    
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.001: continue 
                    
                total_dutch_stake = liquid_bankroll * stake_pct
                
                max_liquidity = MAX_BET_LIQUIDITY.get(comp, MAX_BET_LIQUIDITY.get('DEFAULT', 200.0))
                if total_dutch_stake > max_liquidity:
                    total_dutch_stake = max_liquidity

                if liquid_bankroll - total_dutch_stake < 0:
                    total_dutch_stake = liquid_bankroll
                    if total_dutch_stake <= 0: break
                        
                stake_1 = total_dutch_stake * (implied_prob_1 / total_implied)
                stake_X = total_dutch_stake * (implied_prob_X / total_implied)
                
                liquid_bankroll -= total_dutch_stake
                day_staked += total_dutch_stake
                total_staked += total_dutch_stake
                bets_placed += 1
                total_expected_profit += (total_dutch_stake * best_ev) 
                avg_odds_list.append(combined_odds)
                
                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0, 'clv_list': []}

                if has_closing_odds:
                    idx = bet['index']
                    c_loss = c_odds_loss[idx]
                    c_draw = c_odds_draw[idx]
                    c_win = c_odds_win[idx]
                    
                    if not np.isnan(c_loss) and not np.isnan(c_draw) and not np.isnan(c_win) and c_loss > 0 and c_draw > 0 and c_win > 0:
                        c_margin = (1/c_loss) + (1/c_draw) + (1/c_win)
                        fair_prob_1 = (1/c_win) / c_margin
                        fair_prob_X = (1/c_draw) / c_margin
                        fair_c_combined_odds = 1 / (fair_prob_1 + fair_prob_X)
                        true_clv = (combined_odds / fair_c_combined_odds) - 1
                        clv_list.append(true_clv)
                        league_stats[comp]['clv_list'].append(true_clv)
                
                league_stats[comp]['bets'] += 1
                league_stats[comp]['staked'] += total_dutch_stake
                
                if real_outcome == best_choice:
                    gross_profit = stake_1 * (odds[best_choice] - 1) - stake_X
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE) if gross_profit > 0 else gross_profit
                    total_return = total_dutch_stake + net_profit
                    day_profit += total_return
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                    if net_profit > 0:
                        gross_profit_sum += net_profit
                    else:
                        gross_loss_sum += abs(net_profit)
                    final_net_profit = net_profit
                elif real_outcome == secondary_choice:
                    gross_profit = stake_X * (odds[secondary_choice] - 1) - stake_1
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE) if gross_profit > 0 else gross_profit
                    total_return = total_dutch_stake + net_profit
                    day_profit += total_return
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                    if net_profit > 0:
                        gross_profit_sum += net_profit
                    else:
                        gross_loss_sum += abs(net_profit)
                    final_net_profit = net_profit
                else:
                    final_net_profit = -total_dutch_stake
                    league_stats[comp]['profit'] -= total_dutch_stake
                    gross_loss_sum += total_dutch_stake
                    
                placed_bets_history.append({
                    'ev': best_ev,
                    'prob': blended_probs[best_choice] + blended_probs[secondary_choice],
                    'odds': combined_odds,
                    'stake': total_dutch_stake,
                    'is_win': int(real_outcome in [best_choice, secondary_choice]),
                    'net_profit': final_net_profit
                })
        
        liquid_bankroll += day_profit
        bankroll_history.append(liquid_bankroll)
        
        if liquid_bankroll > historical_peak:
            historical_peak = liquid_bankroll
        current_dd = (historical_peak - liquid_bankroll) / historical_peak
        if current_dd > historical_mdd:
            historical_mdd = current_dd
        
        if day_staked > 0:
            daily_multiplier = liquid_bankroll / start_of_day_bankroll
            daily_multipliers.append(daily_multiplier)

    yield_pct = ((liquid_bankroll - 1000.0) / total_staked) * 100 if total_staked > 0 else 0
    roi_pct = ((liquid_bankroll - 1000.0) / 1000.0) * 100
    x_yield_pct = (total_expected_profit / total_staked) * 100 if total_staked > 0 else 0
    
    avg_clv = np.mean(clv_list) * 100 if len(clv_list) > 0 else 0.0
    median_clv = np.median(clv_list) * 100 if len(clv_list) > 0 else 0.0
    beat_close_rate = (sum(1 for clv in clv_list if clv > 0) / len(clv_list)) * 100 if len(clv_list) > 0 else 0.0
    
    avg_odds = np.mean(avg_odds_list) if len(avg_odds_list) > 0 else 0.0
    profit_factor = gross_profit_sum / gross_loss_sum if gross_loss_sum > 0 else float('inf')
    
    daily_returns = [mult - 1 for mult in daily_multipliers]
    mean_return = np.mean(daily_returns) if daily_returns else 0
    std_return = np.std(daily_returns) if daily_returns else 0
    sharpe_ratio = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0
    
    downside_returns = [r for r in daily_returns if r < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return / downside_std) * np.sqrt(365) if downside_std > 0 else 0

    bet_percentage = (bets_placed / total_analyzed_matches) * 100 if total_analyzed_matches > 0 else 0
    
    logger.info("=== RESULTADOS GLOBALES ===")
    logger.info(f"Capital Inicial: $1000.00 | Capital Final Líquido: ${liquid_bankroll:.2f}")
    logger.info(f"Partidos Analizados (Whitelist & Cuotas válidas): {total_analyzed_matches}")
    logger.info(f"Apuestas Realizadas: {bets_placed} ({bet_percentage:.1f}% de selectividad) | Apuestas Ganadas: {bets_won} ({(bets_won/bets_placed)*100:.1f}% WinRate)" if bets_placed > 0 else "Apuestas Realizadas: 0")
    logger.info(f"Cuota Promedio Apostada: {avg_odds:.2f}")
    logger.info(f"Volumen Apostado (Turnover): ${total_staked:.2f}")
    logger.info(f"Fricción de Mercado Simulada (Impuestos / Ganancias Neta): {TAX_RETENTION_RATE*100:.2f}%")
    logger.info(f"Yield Real (Beneficio Neto / Turnover): {yield_pct:.2f}% | Expected Yield (xYield): {x_yield_pct:.2f}%")
    logger.info(f"Profit Factor (Ganancias / Pérdidas): {profit_factor:.2f}")
    logger.info(f"ROI del Capital Inicial: {roi_pct:.2f}%")
    logger.info(f"Maximum Drawdown Histórico Real: {historical_mdd*100:.2f}%")
    
    if len(daily_returns) > 1:
        logger.info(f"Ratio de Sharpe Anualizado: {sharpe_ratio:.2f} | Ratio de Sortino: {sortino_ratio:.2f}")
    
    if has_closing_odds:
        logger.info("=== ANÁLISIS DE CLOSING LINE VALUE (CLV) ===")
        logger.info(f"Promedio CLV: {avg_clv:.2f}% | Mediana CLV: {median_clv:.2f}%")
        logger.info(f"Beat The Close Rate (Cuotas superadas al cierre): {beat_close_rate:.1f}%")
        clv_efficiency = yield_pct - avg_clv
        logger.info(f"Eficiencia del CLV vs Yield (Varianza o Suerte): {clv_efficiency:.2f}% (Positivo = Runneando por encima del EV, Negativo = Mala varianza)")
    
    logger.info("=== RENDIMIENTO POR LIGA ===")
    for comp, stats in sorted(league_stats.items(), key=lambda x: x[1]['profit'], reverse=True):
        if stats['bets'] > 0:
            l_winrate = (stats['won'] / stats['bets']) * 100
            l_yield = (stats['profit'] / stats['staked']) * 100
            l_clv = (np.mean(stats['clv_list']) * 100) if stats.get('clv_list') else 0.0
            clv_str = f" | CLV: {l_clv:.2f}%" if stats.get('clv_list') else ""
            logger.info(f"Liga {comp}: {stats['bets']} apuestas | WinRate: {l_winrate:.1f}% | Yield: {l_yield:.2f}% | Profit: ${stats['profit']:.2f} | Kelly: {KELLY_FRACTIONS.get(comp,0):.3f} | EV: {EV_THRESHOLDS.get(comp,0):.3f}{clv_str}")
    
    if yield_pct > 0:
        logger.info("EL MODELO ES RENTABLE. (Tiene Edge real contra el mercado).")
    else:
        logger.warning("EL MODELO PIERDE DINERO. (Las cuotas del mercado son más eficientes que el modelo).")
        
    if len(daily_multipliers) > 10:
        logger.info("=== PRUEBA DE RESISTENCIA (MONTE CARLO) ===")
        logger.info(f"Ejecutando 10,000 simulaciones de bootstrapping sobre {len(daily_multipliers)} bloques diarios...")
        
        n_sims = 10000
        ruin_count = 0
        max_drawdowns = []
        final_capitals = []
        
        for _ in range(n_sims):
            sim_bankroll = 1000.0
            peak = 1000.0
            max_dd = 0.0
            is_ruined = False
            
            sampled_multipliers = random.choices(daily_multipliers, k=len(daily_multipliers))
            
            for mult in sampled_multipliers:
                sim_bankroll *= mult
                if sim_bankroll > peak:
                    peak = sim_bankroll
                dd = (peak - sim_bankroll) / peak
                if dd > max_dd:
                    max_dd = dd
                if sim_bankroll <= 10.0: 
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

    # --- NUEVOS DIAGNÓSTICOS QUANT ---
    if placed_bets_history:
        df_bets = pd.DataFrame(placed_bets_history)
        
        logger.info("=== YIELD POR BUCKETS DE EXPECTED VALUE (EV) ===")
        ev_bins = [-np.inf, 0.05, 0.10, 0.15, 0.20, np.inf]
        ev_labels = ['<5%', '5-10%', '10-15%', '15-20%', '>20%']
        df_bets['ev_bucket'] = pd.cut(df_bets['ev'], bins=ev_bins, labels=ev_labels, right=False)
        
        ev_stats = df_bets.groupby('ev_bucket', observed=False).agg(
            bets=('ev', 'count'),
            staked=('stake', 'sum'),
            profit=('net_profit', 'sum')
        ).reset_index()
        
        for _, row in ev_stats.iterrows():
            if row['bets'] > 0:
                b_yield = (row['profit'] / row['staked']) * 100
                logger.info(f"EV Bucket {row['ev_bucket']}: {int(row['bets'])} apuestas | Yield: {b_yield:.2f}% | Profit: ${row['profit']:.2f}")

        logger.info("=== CURVA DE CALIBRACIÓN (RELIABILITY DIAGRAM) ===")
        prob_bins = np.linspace(0, 1, 21)
        df_bets['prob_bucket'] = pd.cut(df_bets['prob'], bins=prob_bins).astype(str)
        
        calib_stats = df_bets.groupby('prob_bucket', observed=False).agg(
            bets=('prob', 'count'),
            mean_pred=('prob', 'mean'),
            hit_rate=('is_win', 'mean')
        ).dropna().reset_index()
        
        calib_stats = calib_stats[calib_stats['bets'] > 0].copy()
        
        for _, row in calib_stats.iterrows():
            diff = (row['mean_pred'] - row['hit_rate']) * 100
            diff_str = f"+{diff:.1f}% (Overconfident)" if diff > 0 else f"{diff:.1f}% (Underconfident)"
            logger.info(f"Prob {row['prob_bucket']}: {int(row['bets'])} apuestas | Pred: {row['mean_pred']*100:.1f}% | Real: {row['hit_rate']*100:.1f}% | Diff: {diff_str}")
            
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(log_dir, f'calibration_data_{timestamp_str}.csv')
        calib_stats.to_csv(csv_path, index=False)
        logger.info(f"Datos de curva de calibración guardados en: {csv_path}")
        
        try:
            plt.figure(figsize=(8, 8))
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.plot(calib_stats['mean_pred'], calib_stats['hit_rate'], "s-", label="Model")
            for i, row in calib_stats.iterrows():
                plt.text(row['mean_pred'], row['hit_rate'], str(int(row['bets'])), fontsize=8, ha='right', va='bottom')
            plt.ylabel("Fraction of positives (Hit Rate)")
            plt.xlabel("Mean predicted probability")
            plt.title("Reliability Diagram (Calibration Curve)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plot_path = os.path.join(log_dir, f'calibration_plot_{timestamp_str}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Gráfica de calibración generada en: {plot_path}")
        except Exception as e:
            logger.error(f"Error generando gráfica de calibración: {e}")

        if len(placed_bets_history) > 10:
            logger.info("=== MONTE CARLO DE CALIBRACIÓN (EXPECTED DISTRIBUTION) ===")
            logger.info(f"Ejecutando 10,000 simulaciones sintéticas basadas en las probabilidades del modelo...")
            
            n_sims = 10000
            synthetic_yields = []
            synthetic_mdds = []
            
            probs_array = df_bets['prob'].values
            odds_array = df_bets['odds'].values
            stakes_array = df_bets['stake'].values
            total_staked_synth = np.sum(stakes_array)
            
            if total_staked_synth > 0:
                for _ in range(n_sims):
                    sim_wins = np.random.rand(len(probs_array)) < probs_array
                    net_profits = np.where(sim_wins, stakes_array * (odds_array - 1) * (1.0 - TAX_RETENTION_RATE), -stakes_array)
                    
                    total_profit_synth = np.sum(net_profits)
                    synthetic_yields.append(total_profit_synth / total_staked_synth)
                    
                    bankroll_path = 1000.0 + np.cumsum(net_profits)
                    peaks = np.maximum.accumulate(np.insert(bankroll_path, 0, 1000.0))
                    path_with_initial = np.insert(bankroll_path, 0, 1000.0)
                    drawdowns = (peaks - path_with_initial) / peaks
                    synthetic_mdds.append(np.max(drawdowns))
                    
                synthetic_yields = np.array(synthetic_yields) * 100
                synthetic_mdds = np.array(synthetic_mdds) * 100
                
                real_yield = yield_pct
                real_mdd = historical_mdd * 100
                
                percentile_yield = np.sum(synthetic_yields < real_yield) / n_sims * 100
                percentile_mdd = np.sum(synthetic_mdds < real_mdd) / n_sims * 100
                
                logger.info(f"Yield Real: {real_yield:.2f}% | xYield Mediano (Simulación): {np.median(synthetic_yields):.2f}%")
                logger.info(f"El Yield Real cae en el percentil {percentile_yield:.1f}% de la distribución de calibración.")
                
                logger.info(f"MDD Real: {real_mdd:.2f}% | xMDD Mediano (Simulación): {np.median(synthetic_mdds):.2f}%")
                logger.info(f"El MDD Real cae en el percentil {percentile_mdd:.1f}% de la distribución de calibración.")
                
                if percentile_yield < 10.0:
                    logger.warning("OVERCONFIDENCE CONFIRMADA: Tu Yield real está en el 10% inferior de lo que el modelo predecía. Las probabilidades están sobreestimadas.")
                elif percentile_yield > 90.0:
                    logger.warning("UNDERCONFIDENCE: Tu Yield real es excepcionalmente bueno frente a lo predicho. Posible varianza positiva ('Good Luck').")
                else:
                    logger.info("CALIBRACIÓN ADECUADA: El rendimiento está dentro del rango esperado por la varianza normal del modelo.")
                    
                if percentile_mdd > 90.0:
                    logger.warning("RIESGO EXTREMO: El MDD real fue mucho peor (Top 10%) que el peor escenario probabilístico del modelo.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_simulation()
