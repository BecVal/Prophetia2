import os
import logging
import numpy as np
import pandas as pd
import random

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración Quant
FILTER_BY_WHITELIST = True  # True: ignora partidos fuera de Whitelist en la simulación financiera
WHITELIST_LEAGUES = ['I1', 'D2', 'SP1', 'F2', 'G1', 'D1', 'T1', 'F1']

# Diccionarios de riesgo por liga (ajustados a liquidez y eficiencia)
KELLY_FRACTIONS = {'D2': 0.10, 'I1': 0.20, 'SP1': 0.05, 'F2': 0.05, 'G1': 0.05, 'D1': 0.05, 'T1': 0.04, 'F1': 0.02, 'DEFAULT': 0.02}
EV_THRESHOLDS = {'D2': 0.10, 'I1': 0.06, 'SP1': 0.10, 'F2': 0.10, 'G1': 0.08, 'D1': 0.07, 'T1': 0.08, 'F1': 0.10, 'DEFAULT': 0.12}

MAX_STAKE_PCT = 0.05  # Cap de apuesta por partido (5% del bankroll líquido)

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
        
    logger.info("=== EVALUACIÓN FINANCIERA (Bankroll Simulation) ===")
    
    odds_win = df['odds_win'].values
    odds_draw = df['odds_draw'].values
    odds_loss = df['odds_loss'].values
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
    
    # Agrupar índices por fecha (timestamp) para evitar Sequential Loop Bug
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
                
            evs = [ (probs[j] * odds[j]) - 1 for j in range(3) ]
            
            # Obtener umbrales dinámicos para esta liga
            league_ev_thresh = EV_THRESHOLDS.get(comp, EV_THRESHOLDS['DEFAULT'])
            league_kelly = KELLY_FRACTIONS.get(comp, KELLY_FRACTIONS['DEFAULT'])
            
            # Check for Dutching / Doble Oportunidad (Local y Empate)
            ev_local = evs[2]
            ev_draw = evs[1]
            
            bet_type = 'single'
            best_choice = np.argmax(evs)
            best_ev = evs[best_choice]
            secondary_choice = None
            
            # Dutching logic if both Local and Draw have EV > league_ev_thresh
            if ev_local > league_ev_thresh and ev_draw > league_ev_thresh:
                bet_type = 'dutching'
                implied_prob_1 = 1 / odds[2]
                implied_prob_X = 1 / odds[1]
                total_implied = implied_prob_1 + implied_prob_X
                combined_odds = 1 / total_implied
                best_ev = ((probs[2] + probs[1]) * combined_odds) - 1
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
            kelly_fraction = bet['kelly_fraction']
            
            if bet_type == 'single':
                b = odds[best_choice] - 1
                kelly_pct = best_ev / b if b > 0 else 0
                
                # Conservador en cuotas bajas para mitigar trampa isotónica
                if odds[best_choice] < 1.30:
                    kelly_pct = min(kelly_pct, 0.01) # Cap 1% Kelly
                    
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
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
                    
                stake_pct = min(kelly_pct * kelly_fraction, MAX_STAKE_PCT)
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
    logger.info(f"Apuestas Realizadas: {bets_placed} | Apuestas Ganadas: {bets_won} ({(bets_won/bets_placed)*100:.1f}% WinRate)" if bets_placed > 0 else "Apuestas Realizadas: 0")
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

    logger.info("--- Muestra de Predicciones de Valor (Set de Prueba) ---")
    
    # Muestra de predicciones
    # Verificamos si tenemos las columnas extra
    has_extra_info = all(c in df.columns for c in ['team', 'opponent', 'is_home', 'predicted_xg_scored', 'predicted_xg_conceded'])
    
    for i in range(min(5, len(df))):
        p_loss, p_draw, p_win = y_prob[i]
        real = y_test[i]
        
        if has_extra_info:
            team_name = df['team'].iloc[i]
            opp_name = df['opponent'].iloc[i]
            is_home_flag = df['is_home'].iloc[i]
            
            local = team_name if is_home_flag == 1 else opp_name
            visitante = opp_name if is_home_flag == 1 else team_name
            
            pred_xg = df['predicted_xg_scored'].iloc[i]
            pred_xg_opp = df['predicted_xg_conceded'].iloc[i]
            
            xg_local = pred_xg if is_home_flag == 1 else pred_xg_opp
            xg_visitante = pred_xg_opp if is_home_flag == 1 else pred_xg
            
            real_str = f"Victoria {team_name}" if real == 2 else "Empate" if real == 1 else f"Victoria {opp_name}"
            
            logger.info(f"Partido {i+1}: {local} ({xg_local:.2f} xG) vs {visitante} ({xg_visitante:.2f} xG)")
            logger.info(f"  -> Prob. {team_name}: {p_win*100:5.1f}% | Empate: {p_draw*100:5.1f}% | {opp_name}: {p_loss*100:5.1f}%")
            logger.info(f"  => Realidad: {real_str} (Clase {real})\n")
        else:
            real_str = "Victoria Local" if real == 2 else "Empate" if real == 1 else "Victoria Visitante"
            logger.info(f"Partido {i+1}:")
            logger.info(f"  -> Prob. Local: {p_win*100:5.1f}% | Empate: {p_draw*100:5.1f}% | Visitante: {p_loss*100:5.1f}%")
            logger.info(f"  => Realidad: {real_str} (Clase {real})\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_simulation()
