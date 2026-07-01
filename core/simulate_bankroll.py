import os
import json
import logging
import numpy as np
import pandas as pd
import random
import optuna

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración Quant
FILTER_BY_WHITELIST = False  # True: ignora partidos fuera de Whitelist en la simulación financiera
WHITELIST_LEAGUES = ['I1', 'D2', 'SP1', 'F2', 'G1', 'D1', 'T1', 'F1']

# --- CONFIGURACIÓN DE OPTIMIZACIÓN ---
# Opciones: 'NONE', 'ALL', 'WHITELIST', o una liga específica como 'I1'
OPTIMIZATION_MODE = 'NONE' 
OPTUNA_TRIALS = 200
OPTIMIZED_PARAMS_FILE = '../data/processed/optimal_bankroll_params.json'

# Diccionarios de riesgo por liga iniciales/por defecto
KELLY_FRACTIONS = {'D2': 0.03, 'I1': 0.01, 'SP1': 0.03, 'F2': 0.02, 'G1': 0.01, 'D1': 0.02, 'T1': 0.03, 'F1': 0.02, 'E1': 0.02, 'N1': 0.01, 'SP2': 0.01, 'P1': 0.01, 'DEFAULT': 0.015}
EV_THRESHOLDS = {'D2': 0.015, 'I1': 0.02, 'SP1': 0.01, 'F2': 0.015, 'G1': 0.02, 'D1': 0.015, 'T1': 0.01, 'F1': 0.015, 'E1': 0.015, 'N1': 0.02, 'SP2': 0.02, 'P1': 0.02, 'DEFAULT': 0.015}

MAX_STAKE_PCT = 0.03  # Cap reducido al 3% para mayor seguridad con el nuevo filtro

# Parámetros Institucionales/Fricción Añadidos
TAX_RETENTION_RATE = 0.07  # Retención del 7% sobre ganancias netas (Común en MX: 1% federal + 6% estatal, p.ej. Caliente/Draftea)
MARKET_BLEND_ALPHA = 0.85  # Peso del modelo vs mercado (85% modelo, 15% mercado)
EXPECTED_CLV_DROP = 0.015  # Penalización por slippage esperado del CLV (-1.5%)
MAX_BET_LIQUIDITY = {      # Límites de liquidez absolutos en USD o Unidad Base
    'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
    'D2': 1000.0, 'F2': 1000.0,
    'T1': 500.0,
    'DEFAULT': 200.0
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
                if divergence > 0.20: dynamic_alpha = 0.50
                elif divergence > 0.10: dynamic_alpha = 0.70
                blended_probs.append((dynamic_alpha * [p_loss, p_draw, p_win][j]) + ((1 - dynamic_alpha) * market_probs[j]))
            
            net_odds = [1 + (odds[j] - 1) * (1 - TAX_RETENTION_RATE) for j in range(3)]
            evs = [ (blended_probs[j] * net_odds[j]) - 1 - EXPECTED_CLV_DROP for j in range(3) ]
            
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
                kelly_pct = best_ev / b if b > 0 else 0
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
                
                kelly_pct = best_ev / b if b > 0 else 0
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
        ev_thresh = trial.suggest_float('ev_thresh', 0.001, 0.05)
        kelly_fraction = trial.suggest_float('kelly_fraction', 0.005, 0.15)
        
        roi, mdd = evaluate_league_params(df_league, ev_thresh, kelly_fraction)
        
        if roi <= -0.99:
            return -999.0 # Ruina penalizada
            
        # Riesgo Ajustado: ROI / (MDD + 1%)
        score = roi / (mdd + 0.01)
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
        leagues_to_optimize = []
        if OPTIMIZATION_MODE == 'ALL':
            leagues_to_optimize = df['competition'].unique()
        elif OPTIMIZATION_MODE == 'WHITELIST':
            leagues_to_optimize = [L for L in df['competition'].unique() if L in WHITELIST_LEAGUES]
        else:
            leagues_to_optimize = [OPTIMIZATION_MODE] # Asumimos que es el nombre de la liga (ej 'I1')
            
        logger.info(f"Modo Optimización Activado: Optimizando {len(leagues_to_optimize)} ligas.")
        for comp in leagues_to_optimize:
            if FILTER_BY_WHITELIST and comp not in WHITELIST_LEAGUES:
                continue
            ev_opt, kelly_opt = optimize_league(df, comp)
            if ev_opt is not None:
                EV_THRESHOLDS[comp] = ev_opt
                KELLY_FRACTIONS[comp] = kelly_opt
                
        # Guardar resultados para futuro uso
        save_optimized_params()
    else:
        # Cargar si existe el archivo
        load_optimized_params()

    logger.info("=== EVALUACIÓN FINANCIERA (Bankroll Simulation) ===")
    
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
    
    unique_dates = np.unique(dates)
    league_stats = {}

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
                    dynamic_alpha = 0.50  
                elif divergence > 0.10:
                    dynamic_alpha = 0.70  
                blended_probs.append((dynamic_alpha * probs[j]) + ((1 - dynamic_alpha) * market_probs[j]))
            
            net_odds = [1 + (odds[j] - 1) * (1 - TAX_RETENTION_RATE) for j in range(3)]
            evs = [ (blended_probs[j] * net_odds[j]) - 1 - EXPECTED_CLV_DROP for j in range(3) ]
            
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
                kelly_pct = best_ev / b if b > 0 else 0
                
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
                
                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0}
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
                else:
                    league_stats[comp]['profit'] -= stake
                    
            elif bet_type == 'dutching':
                implied_prob_1 = 1 / odds[best_choice]
                implied_prob_X = 1 / odds[secondary_choice]
                total_implied = implied_prob_1 + implied_prob_X
                combined_odds = 1 / total_implied
                
                net_combined_odds = 1 + (combined_odds - 1) * (1 - TAX_RETENTION_RATE)
                b = net_combined_odds - 1
                kelly_pct = best_ev / b if b > 0 else 0
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
                
                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0}
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
                elif real_outcome == secondary_choice:
                    gross_profit = stake_X * (odds[secondary_choice] - 1) - stake_1
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE) if gross_profit > 0 else gross_profit
                    total_return = total_dutch_stake + net_profit
                    day_profit += total_return
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                else:
                    league_stats[comp]['profit'] -= total_dutch_stake
        
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
    
    bet_percentage = (bets_placed / total_analyzed_matches) * 100 if total_analyzed_matches > 0 else 0
    
    logger.info(f"Capital Inicial: $1000.00 | Capital Final Líquido: ${liquid_bankroll:.2f}")
    logger.info(f"Partidos Analizados (Whitelist & Cuotas válidas): {total_analyzed_matches}")
    logger.info(f"Apuestas Realizadas: {bets_placed} ({bet_percentage:.1f}% de selectividad) | Apuestas Ganadas: {bets_won} ({(bets_won/bets_placed)*100:.1f}% WinRate)" if bets_placed > 0 else "Apuestas Realizadas: 0")
    logger.info(f"Volumen Apostado (Turnover): ${total_staked:.2f}")
    logger.info(f"Fricción de Mercado Simulada (Impuestos / Ganancias Neta): {TAX_RETENTION_RATE*100:.1f}%")
    logger.info(f"Yield Real (Beneficio Neto / Turnover): {yield_pct:.2f}% | Expected Yield (xYield): {x_yield_pct:.2f}%")
    if has_closing_odds:
        logger.info(f"Promedio Closing Line Value (CLV): {avg_clv:.2f}%")
    logger.info(f"ROI del Capital Inicial: {roi_pct:.2f}%")
    logger.info(f"Maximum Drawdown Histórico Real: {historical_mdd*100:.2f}%")
    
    logger.info("=== RENDIMIENTO POR LIGA ===")
    for comp, stats in sorted(league_stats.items(), key=lambda x: x[1]['profit'], reverse=True):
        if stats['bets'] > 0:
            l_winrate = (stats['won'] / stats['bets']) * 100
            l_yield = (stats['profit'] / stats['staked']) * 100
            logger.info(f"Liga {comp}: {stats['bets']} apuestas | WinRate: {l_winrate:.1f}% | Yield: {l_yield:.2f}% | Profit: ${stats['profit']:.2f} | Kelly: {KELLY_FRACTIONS.get(comp,0):.3f} | EV: {EV_THRESHOLDS.get(comp,0):.3f}")
    
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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_simulation()
