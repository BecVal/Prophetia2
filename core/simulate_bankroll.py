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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'simulate_bankroll')

# --- CONFIGURACIÓN QUANT DE NIVEL INSTITUCIONAL (10/10) ---
FILTER_BY_WHITELIST = False  # True: ignora partidos fuera de Whitelist en la simulación financiera
WHITELIST_LEAGUES = ['F1', 'F2', 'SP1', 'G1', 'B1', 'P1', 'SP2', 'SC0', 'D1', 'D2', 'E1', 'MLS', 'J1', 'E2']

# Límites de Exposición y Gestión de Riesgo de Portafolio
MAX_STAKE_PCT = 0.03               # Stake máximo por apuesta (3.0% del bankroll)
MAX_DAILY_PORTFOLIO_PCT = 0.15     # Presupuesto de riesgo diario simultáneo (máximo 15% expuesto por día)
MAX_DRAWDOWN_TARGET = 0.30         # Umbral de drawdown para desescalamiento reactivo de Kelly (30%)

# Parámetros Institucionales/Fricción de Mercado
TAX_RETENTION_RATE = 0.0075        # Retención del 0.75% sobre ganancias netas (Polymarket)
EXPECTED_CLV_DROP = 0.015         # Penalización por slippage esperado del CLV (-1.5%)
MARKET_IMPACT_GAMMA = 0.05        # Coeficiente de impacto de mercado (Square-root Law)

MAX_BET_LIQUIDITY = {             # Límites de liquidez absolutos por competición (USD)
    'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
    'D2': 2000.0, 'F2': 2000.0, 'T1': 2000.0, 'MLS': 1500.0, 'J1': 1500.0,
    'DEFAULT': 2000.0
}

# CONFIGURACIÓN DE OPTIMIZACIÓN
OPTIMIZATION_MODE = 'NONE'  # 'NONE', 'ALL', 'WHITELIST', o liga específica
OPTUNA_TRIALS = 1000
OPTIMIZED_PARAMS_FILE = '../data/processed/models_best_parameters/optimal_bankroll_params.json'

# Diccionarios de riesgo por defecto
KELLY_FRACTIONS = {'D2': 0.03, 'I1': 0.01, 'SP1': 0.03, 'F2': 0.02, 'G1': 0.01, 'D1': 0.02, 'T1': 0.03, 'F1': 0.02, 'E1': 0.02, 'N1': 0.01, 'SP2': 0.01, 'P1': 0.01, 'DEFAULT': 0.015}
EV_THRESHOLDS = {'D2': 0.015, 'I1': 0.02, 'SP1': 0.01, 'F2': 0.015, 'G1': 0.02, 'D1': 0.015, 'T1': 0.01, 'F1': 0.015, 'E1': 0.015, 'N1': 0.02, 'SP2': 0.02, 'P1': 0.02, 'DEFAULT': 0.015}

ALPHA_DIV_LOW = {'DEFAULT': 0.85}
ALPHA_DIV_MED = {'DEFAULT': 0.70}
ALPHA_DIV_HIGH = {'DEFAULT': 0.50}

# MAPEO ANTI-CORRUPCIÓN: Europa, EE.UU. (MLS), Japón (J1) e Internacionales UEFA/FIFA
COMPETITION_MAPPING = {
    # Ligas Europeas Principales
    'E0': 'Premier League', 'Premier League': 'E0',
    'SP1': 'La Liga', 'La Liga': 'SP1',
    'D1': '1. Bundesliga', '1. Bundesliga': 'D1',
    'I1': 'Serie A', 'Serie A': 'I1',
    'F1': 'Ligue 1', 'Ligue 1': 'F1',
    'E1': 'Championship', 'Championship': 'E1',
    'SP2': 'La Liga 2', 'La Liga 2': 'SP2',
    'D2': '2. Bundesliga', '2. Bundesliga': 'D2',
    'F2': 'Ligue 2', 'Ligue 2': 'F2',
    'I2': 'Serie B', 'Serie B': 'I2',
    'B1': 'Jupiler Pro League', 'Jupiler Pro League': 'B1',
    'N1': 'Eredivisie', 'Eredivisie': 'N1',
    'P1': 'Primeira Liga', 'Primeira Liga': 'P1',
    'SC0': 'Scottish Premiership', 'Scottish Premiership': 'SC0',
    'T1': 'Süper Lig', 'Süper Lig': 'T1',
    'E2': 'League One', 'League One': 'E2',
    # EE.UU. y Japón (Bajo índice de corrupción)
    'MLS': 'Major League Soccer', 'Major League Soccer': 'MLS',
    'J1': 'J-League 1', 'J-League 1': 'J1',
    # Torneos Internacionales UEFA / FIFA
    'CL': 'Champions League', 'Champions League': 'CL',
    'EL': 'Europa League', 'Europa League': 'EL',
    'WC': 'FIFA World Cup', 'FIFA World Cup': 'WC'
}

def get_param_by_comp(param_dict, comp, default_val=0.015):
    """ Busca un parámetro intentando el código corto y el nombre largo """
    if comp in param_dict:
        return param_dict[comp]
    alt_name = COMPETITION_MAPPING.get(comp)
    if alt_name and alt_name in param_dict:
        return param_dict[alt_name]
    return param_dict.get('DEFAULT', default_val)

def load_optimized_params():
    global KELLY_FRACTIONS, EV_THRESHOLDS, ALPHA_DIV_LOW, ALPHA_DIV_MED, ALPHA_DIV_HIGH
    if os.path.exists(OPTIMIZED_PARAMS_FILE):
        try:
            with open(OPTIMIZED_PARAMS_FILE, 'r') as f:
                data = json.load(f)
                if 'KELLY_FRACTIONS' in data: KELLY_FRACTIONS.update(data['KELLY_FRACTIONS'])
                if 'EV_THRESHOLDS' in data: EV_THRESHOLDS.update(data['EV_THRESHOLDS'])
                if 'ALPHA_DIV_LOW' in data: ALPHA_DIV_LOW.update(data['ALPHA_DIV_LOW'])
                if 'ALPHA_DIV_MED' in data: ALPHA_DIV_MED.update(data['ALPHA_DIV_MED'])
                if 'ALPHA_DIV_HIGH' in data: ALPHA_DIV_HIGH.update(data['ALPHA_DIV_HIGH'])
            logger.info("Parámetros optimizados cargados desde archivo local.")
        except Exception as e:
            logger.error(f"Error al cargar parámetros optimizados: {e}")

def save_optimized_params():
    data = {
        'KELLY_FRACTIONS': KELLY_FRACTIONS,
        'EV_THRESHOLDS': EV_THRESHOLDS,
        'ALPHA_DIV_LOW': ALPHA_DIV_LOW,
        'ALPHA_DIV_MED': ALPHA_DIV_MED,
        'ALPHA_DIV_HIGH': ALPHA_DIV_HIGH
    }
    try:
        os.makedirs(os.path.dirname(OPTIMIZED_PARAMS_FILE), exist_ok=True)
        with open(OPTIMIZED_PARAMS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Parámetros óptimos guardados exitosamente en {OPTIMIZED_PARAMS_FILE}")
    except Exception as e:
        logger.error(f"Error al guardar parámetros optimizados: {e}")

def calculate_dynamic_alpha(p_model, p_market, alpha_low, alpha_med, alpha_high):
    """ Función continua (smooth decay) para blending de probabilidades """
    divergence = abs(p_model - p_market)
    alpha_max = max(alpha_low, alpha_med, alpha_high)
    alpha_min = min(alpha_low, alpha_med, alpha_high)
    decay = np.exp(-25.0 * (divergence ** 2))
    return alpha_min + (alpha_max - alpha_min) * decay

def calculate_market_slippage(odds, stake, liquidity_cap):
    """ Modelo de impacto de mercado (Square-Root Law) """
    if stake <= 0 or liquidity_cap <= 0:
        return odds
    ratio = min(stake / liquidity_cap, 1.0)
    impact = MARKET_IMPACT_GAMMA * np.sqrt(ratio)
    effective_odds = 1.0 + (odds - 1.0) * (1.0 - impact)
    return max(effective_odds, 1.001)

def run_bankroll_engine(df, custom_ev_thresholds=None, custom_kelly_fractions=None,
                        custom_alpha_low=None, custom_alpha_med=None, custom_alpha_high=None):
    """
    Motor cuantitativo unificado de simulación financiera de bankroll.
    Incorpora:
      1. Presupuesto de riesgo diario simultáneo (MAX_DAILY_PORTFOLIO_PCT = 15%).
      2. De-risking dinámico por drawdown (Kelly Reactivo).
      3. Modelo de deslizamiento por impacto de mercado (Slippage).
      4. Tratamiento impositivo exacto en apuestas sencillas y dutching.
    """
    ev_dict = custom_ev_thresholds if custom_ev_thresholds is not None else EV_THRESHOLDS
    kelly_dict = custom_kelly_fractions if custom_kelly_fractions is not None else KELLY_FRACTIONS
    alpha_low_dict = custom_alpha_low if custom_alpha_low is not None else ALPHA_DIV_LOW
    alpha_med_dict = custom_alpha_med if custom_alpha_med is not None else ALPHA_DIV_MED
    alpha_high_dict = custom_alpha_high if custom_alpha_high is not None else ALPHA_DIV_HIGH

    odds_win = df['odds_win'].values
    odds_draw = df['odds_draw'].values
    odds_loss = df['odds_loss'].values
    
    has_closing_odds = all(c in df.columns for c in ['closing_odds_win', 'closing_odds_draw', 'closing_odds_loss'])
    if has_closing_odds:
        c_odds_win = df['closing_odds_win'].values
        c_odds_draw = df['closing_odds_draw'].values
        c_odds_loss = df['closing_odds_loss'].values

    dates = df['match_date'].values
    competitions = df['competition'].values if 'competition' in df.columns else np.array([None]*len(df))
    y_test = df['outcome'].values
    y_prob = df[['prob_loss', 'prob_draw', 'prob_win']].values

    has_pred_clv = all(c in df.columns for c in ['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win'])
    if has_pred_clv:
        pred_clv_vals = df[['pred_clv_loss', 'pred_clv_draw', 'pred_clv_win']].values

    liquid_bankroll = 1000.0
    bankroll_history = [liquid_bankroll]
    daily_multipliers = []
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
    league_stats = {}
    placed_bets_history = []

    unique_dates = np.unique(dates)

    for current_date in sorted(unique_dates):
        start_of_day_bankroll = liquid_bankroll
        day_indices = np.where(dates == current_date)[0]
        candidate_bets = []

        # 1. EVALUACIÓN Y SELECCIÓN DE OPORTUNIDADES EV+
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

            market_implied = [1.0 / odds[j] for j in range(3)]
            margin = sum(market_implied)
            market_probs = [p / margin for p in market_implied]

            a_low = get_param_by_comp(alpha_low_dict, comp, 0.85)
            a_med = get_param_by_comp(alpha_med_dict, comp, 0.70)
            a_high = get_param_by_comp(alpha_high_dict, comp, 0.50)

            blended_probs = []
            for j in range(3):
                dyn_alpha = calculate_dynamic_alpha(probs[j], market_probs[j], a_low, a_med, a_high)
                blended_probs.append((dyn_alpha * probs[j]) + ((1.0 - dyn_alpha) * market_probs[j]))

            net_odds = [1.0 + (odds[j] - 1.0) * (1.0 - TAX_RETENTION_RATE) for j in range(3)]
            evs = [(blended_probs[j] * net_odds[j]) - 1.0 - EXPECTED_CLV_DROP for j in range(3)]

            league_ev_thresh = get_param_by_comp(ev_dict, comp, 0.015)
            league_kelly = get_param_by_comp(kelly_dict, comp, 0.015)
            pred_clv = pred_clv_vals[i] if has_pred_clv else [0.0, 0.0, 0.0]

            bet_type = 'single'
            best_choice = np.argmax(evs)
            best_ev = evs[best_choice]
            secondary_choice = None

            MIN_EXPECTED_CLV = 0.0001
            if has_pred_clv and pred_clv[best_choice] < MIN_EXPECTED_CLV:
                best_ev = -1.0

            ev_local, ev_draw, ev_away = evs[2], evs[1], evs[0]

            if ev_local > league_ev_thresh and ev_draw > league_ev_thresh:
                if (not has_pred_clv) or (pred_clv[2] >= MIN_EXPECTED_CLV and pred_clv[1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    total_implied = (1.0 / odds[2]) + (1.0 / odds[1])
                    combined_odds = 1.0 / total_implied
                    blended_prob_1X = blended_probs[2] + blended_probs[1]
                    net_combined_odds = 1.0 + (combined_odds - 1.0) * (1.0 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_1X * net_combined_odds) - 1.0 - EXPECTED_CLV_DROP
                    best_choice = 2
                    secondary_choice = 1
            elif ev_away > league_ev_thresh and ev_draw > league_ev_thresh:
                if (not has_pred_clv) or (pred_clv[0] >= MIN_EXPECTED_CLV and pred_clv[1] >= MIN_EXPECTED_CLV):
                    bet_type = 'dutching'
                    total_implied = (1.0 / odds[0]) + (1.0 / odds[1])
                    combined_odds = 1.0 / total_implied
                    blended_prob_X2 = blended_probs[0] + blended_probs[1]
                    net_combined_odds = 1.0 + (combined_odds - 1.0) * (1.0 - TAX_RETENTION_RATE)
                    best_ev = (blended_prob_X2 * net_combined_odds) - 1.0 - EXPECTED_CLV_DROP
                    best_choice = 0
                    secondary_choice = 1

            if best_ev > league_ev_thresh:
                candidate_bets.append({
                    'kelly_fraction': league_kelly,
                    'index': i,
                    'best_choice': best_choice,
                    'secondary_choice': secondary_choice,
                    'best_ev': best_ev,
                    'odds': odds,
                    'probs': probs,
                    'blended_probs': blended_probs,
                    'bet_type': bet_type,
                    'real_outcome': real_outcome,
                    'comp': comp
                })

        if not candidate_bets:
            bankroll_history.append(liquid_bankroll)
            continue

        candidate_bets.sort(key=lambda x: x['best_ev'], reverse=True)

        # 2. CÁLCULO DE STAKES Y GESTIÓN DE RIESGO DE PORTAFOLIO SIMULTÁNEO
        current_dd = (historical_peak - start_of_day_bankroll) / historical_peak if historical_peak > 0 else 0
        reactive_risk_factor = max(0.25, 1.0 - (current_dd / MAX_DRAWDOWN_TARGET))

        unscaled_bets = []
        total_daily_unscaled_stake = 0.0

        for bet in candidate_bets:
            best_choice = bet['best_choice']
            secondary_choice = bet['secondary_choice']
            odds = bet['odds']
            bet_type = bet['bet_type']
            comp = bet['comp']
            best_ev = bet['best_ev']
            base_kelly = bet['kelly_fraction']

            adj_kelly_fraction = base_kelly * reactive_risk_factor
            max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

            if bet_type == 'single':
                raw_odd = odds[best_choice]
                net_odd = 1.0 + (raw_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
                b = net_odd - 1.0
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if raw_odd < 1.30: kelly_pct = min(kelly_pct, 0.01)

                stake_pct = min(kelly_pct * adj_kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.0001: continue

                target_stake = min(start_of_day_bankroll * stake_pct, max_liquidity)
            else: # dutching
                total_implied = (1.0 / odds[best_choice]) + (1.0 / odds[secondary_choice])
                combined_odds = 1.0 / total_implied
                net_combined_odds = 1.0 + (combined_odds - 1.0) * (1.0 - TAX_RETENTION_RATE)
                b = net_combined_odds - 1.0
                kelly_ev = min(best_ev, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if combined_odds < 1.30: kelly_pct = min(kelly_pct, 0.01)

                stake_pct = min(kelly_pct * adj_kelly_fraction, MAX_STAKE_PCT)
                if stake_pct < 0.001: continue

                target_stake = min(start_of_day_bankroll * stake_pct, max_liquidity)

            unscaled_bets.append((bet, target_stake))
            total_daily_unscaled_stake += target_stake

        if not unscaled_bets:
            bankroll_history.append(liquid_bankroll)
            continue

        # Cierre del Presupuesto Diario de Riesgo (Daily Portfolio Risk Cap = 15%)
        daily_cap_dollars = start_of_day_bankroll * MAX_DAILY_PORTFOLIO_PCT
        scale_ratio = (daily_cap_dollars / total_daily_unscaled_stake) if total_daily_unscaled_stake > daily_cap_dollars else 1.0

        day_profit = 0.0
        day_staked = 0.0

        # 3. EJECUCIÓN SIMULTÁNEA Y LIQUIDACIÓN
        for bet, unscaled_stake in unscaled_bets:
            stake = unscaled_stake * scale_ratio
            if liquid_bankroll - stake < 0:
                stake = liquid_bankroll
                if stake <= 0: break

            best_choice = bet['best_choice']
            secondary_choice = bet['secondary_choice']
            odds = bet['odds']
            bet_type = bet['bet_type']
            comp = bet['comp']
            real_outcome = bet['real_outcome']
            best_ev = bet['best_ev']

            max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

            if bet_type == 'single':
                eff_odds = calculate_market_slippage(odds[best_choice], stake, max_liquidity)

                liquid_bankroll -= stake
                day_staked += stake
                total_staked += stake
                bets_placed += 1
                total_expected_profit += (stake * best_ev)
                avg_odds_list.append(eff_odds)

                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0, 'clv_list': []}

                if has_closing_odds:
                    idx = bet['index']
                    c_loss, c_draw, c_win = c_odds_loss[idx], c_odds_draw[idx], c_odds_win[idx]
                    if not np.isnan([c_loss, c_draw, c_win]).any() and min(c_loss, c_draw, c_win) > 0:
                        c_margin = (1/c_loss) + (1/c_draw) + (1/c_win)
                        fair_closing_odds = [1 / ((1/c_loss)/c_margin), 1 / ((1/c_draw)/c_margin), 1 / ((1/c_win)/c_margin)]
                        true_clv = (eff_odds / fair_closing_odds[best_choice]) - 1.0
                        clv_list.append(true_clv)
                        league_stats[comp]['clv_list'].append(true_clv)

                league_stats[comp]['bets'] += 1
                league_stats[comp]['staked'] += stake

                if real_outcome == best_choice:
                    gross_profit = stake * (eff_odds - 1.0)
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE)
                    day_profit += stake + net_profit
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                    gross_profit_sum += net_profit
                    net_profit_record = net_profit
                else:
                    net_profit_record = -stake
                    league_stats[comp]['profit'] -= stake
                    gross_loss_sum += stake

                placed_bets_history.append({
                    'ev': best_ev,
                    'prob': bet['blended_probs'][best_choice],
                    'odds': eff_odds,
                    'stake': stake,
                    'stake_pct': stake / start_of_day_bankroll if start_of_day_bankroll > 0 else 0,
                    'is_win': int(real_outcome == best_choice),
                    'net_profit': net_profit_record
                })

            elif bet_type == 'dutching':
                total_implied = (1.0 / odds[best_choice]) + (1.0 / odds[secondary_choice])
                raw_combined_odds = 1.0 / total_implied
                eff_combined_odds = calculate_market_slippage(raw_combined_odds, stake, max_liquidity)

                liquid_bankroll -= stake
                day_staked += stake
                total_staked += stake
                bets_placed += 1
                total_expected_profit += (stake * best_ev)
                avg_odds_list.append(eff_combined_odds)

                if comp not in league_stats:
                    league_stats[comp] = {'bets': 0, 'won': 0, 'staked': 0.0, 'profit': 0.0, 'clv_list': []}

                if has_closing_odds:
                    idx = bet['index']
                    c_loss, c_draw, c_win = c_odds_loss[idx], c_odds_draw[idx], c_odds_win[idx]
                    if not np.isnan([c_loss, c_draw, c_win]).any() and min(c_loss, c_draw, c_win) > 0:
                        c_margin = (1/c_loss) + (1/c_draw) + (1/c_win)
                        fair_prob_1 = (1/c_win) / c_margin if best_choice == 2 else (1/c_loss) / c_margin
                        fair_prob_X = (1/c_draw) / c_margin
                        fair_c_combined = 1.0 / (fair_prob_1 + fair_prob_X)
                        true_clv = (eff_combined_odds / fair_c_combined) - 1.0
                        clv_list.append(true_clv)
                        league_stats[comp]['clv_list'].append(true_clv)

                league_stats[comp]['bets'] += 1
                league_stats[comp]['staked'] += stake

                if real_outcome in [best_choice, secondary_choice]:
                    gross_profit = stake * (eff_combined_odds - 1.0)
                    net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE)
                    day_profit += stake + net_profit
                    bets_won += 1
                    league_stats[comp]['profit'] += net_profit
                    league_stats[comp]['won'] += 1
                    gross_profit_sum += net_profit
                    net_profit_record = net_profit
                else:
                    net_profit_record = -stake
                    league_stats[comp]['profit'] -= stake
                    gross_loss_sum += stake

                placed_bets_history.append({
                    'ev': best_ev,
                    'prob': bet['blended_probs'][best_choice] + bet['blended_probs'][secondary_choice],
                    'odds': eff_combined_odds,
                    'stake': stake,
                    'stake_pct': stake / start_of_day_bankroll if start_of_day_bankroll > 0 else 0,
                    'is_win': int(real_outcome in [best_choice, secondary_choice]),
                    'net_profit': net_profit_record
                })

        liquid_bankroll += day_profit
        bankroll_history.append(liquid_bankroll)

        if liquid_bankroll > historical_peak:
            historical_peak = liquid_bankroll
        current_dd = (historical_peak - liquid_bankroll) / historical_peak if historical_peak > 0 else 0
        if current_dd > historical_mdd:
            historical_mdd = current_dd

        if day_staked > 0 and start_of_day_bankroll > 0:
            daily_multiplier = liquid_bankroll / start_of_day_bankroll
            daily_multipliers.append(daily_multiplier)

    roi = (liquid_bankroll - 1000.0) / 1000.0

    return {
        'liquid_bankroll': liquid_bankroll,
        'roi': roi,
        'historical_mdd': historical_mdd,
        'total_staked': total_staked,
        'bets_placed': bets_placed,
        'bets_won': bets_won,
        'total_analyzed_matches': total_analyzed_matches,
        'total_expected_profit': total_expected_profit,
        'clv_list': clv_list,
        'gross_profit_sum': gross_profit_sum,
        'gross_loss_sum': gross_loss_sum,
        'avg_odds_list': avg_odds_list,
        'daily_multipliers': daily_multipliers,
        'league_stats': league_stats,
        'placed_bets_history': placed_bets_history,
        'bankroll_history': bankroll_history
    }

def evaluate_league_params(df_league, ev_thresh, kelly_fraction, alpha_low, alpha_med, alpha_high):
    """ Evaluación para Optuna """
    league_name = df_league['competition'].iloc[0] if not df_league.empty else 'DEFAULT'
    res = run_bankroll_engine(
        df_league,
        custom_ev_thresholds={league_name: ev_thresh},
        custom_kelly_fractions={league_name: kelly_fraction},
        custom_alpha_low={league_name: alpha_low},
        custom_alpha_med={league_name: alpha_med},
        custom_alpha_high={league_name: alpha_high}
    )
    return res['roi'], res['historical_mdd']

def optimize_league(df, league_name):
    logger.info(f"Iniciando Optuna para {league_name} ({OPTUNA_TRIALS} trials)...")
    df_league = df[df['competition'] == league_name].copy()
    if df_league.empty:
        return None

    def objective(trial):
        ev_thresh = trial.suggest_float('ev_thresh', 0.015, 0.050)
        kelly_fraction = trial.suggest_float('kelly_fraction', 0.01, 0.25)
        alpha_low = trial.suggest_float('alpha_low', 0.10, 0.95)
        alpha_med = trial.suggest_float('alpha_med', 0.10, 0.95)
        alpha_high = trial.suggest_float('alpha_high', 0.10, 0.95)

        roi, mdd = evaluate_league_params(df_league, ev_thresh, kelly_fraction, alpha_low, alpha_med, alpha_high)
        if roi <= 0.0:
            return roi - (mdd * 2.0)

        # Score con aversión al riesgo ajustada por Calmar Ratio
        penalty_factor = 2.0
        score = roi - (penalty_factor * mdd)
        if mdd > 0.25:
            score -= (mdd - 0.25) * 3.0
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    best_params = study.best_params
    logger.info(f"Mejores parámetros para {league_name}: EV={best_params['ev_thresh']:.4f}, Kelly={best_params['kelly_fraction']:.4f}")
    return best_params

def run_simulation():
    PREDICTIONS_PATH = '../data/processed/test_predictions.parquet'
    if not os.path.exists(PREDICTIONS_PATH):
        logger.error(f"No se encontró predicciones en {PREDICTIONS_PATH}.")
        return

    logger.info(f"Cargando predicciones del set de prueba desde: {PREDICTIONS_PATH}...")
    df = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')

    if 'odds_win' not in df.columns:
        logger.warning("No se encontraron cuotas en el dataset.")
        return

    # OPTIMIZACIÓN O CARGA DE PARÁMETROS
    if OPTIMIZATION_MODE != 'NONE':
        TRAIN_PREDS_PATH = '../data/processed/train_predictions.parquet'
        if os.path.exists(TRAIN_PREDS_PATH):
            df_train_opt = pd.read_parquet(TRAIN_PREDS_PATH, engine='fastparquet')
            leagues = df_train_opt['competition'].unique() if OPTIMIZATION_MODE == 'ALL' else [OPTIMIZATION_MODE]
            for comp in leagues:
                if FILTER_BY_WHITELIST and comp not in WHITELIST_LEAGUES: continue
                best_params = optimize_league(df_train_opt, comp)
                if best_params:
                    EV_THRESHOLDS[comp] = best_params['ev_thresh']
                    KELLY_FRACTIONS[comp] = best_params['kelly_fraction']
                    ALPHA_DIV_LOW[comp] = best_params['alpha_low']
                    ALPHA_DIV_MED[comp] = best_params['alpha_med']
                    ALPHA_DIV_HIGH[comp] = best_params['alpha_high']
            save_optimized_params()
    else:
        load_optimized_params()

    logger.info("=== EVALUACIÓN FINANCIERA INSTITUCIONAL (Bankroll Simulation 10/10) ===")

    # CALIBRACIÓN GLOBAL
    df_eval = df[df['competition'].isin(WHITELIST_LEAGUES)].copy() if FILTER_BY_WHITELIST else df.copy()
    if not df_eval.empty:
        y_prob_eval = df_eval[['prob_loss', 'prob_draw', 'prob_win']].values
        y_true_eval = df_eval['outcome'].values
        y_true_oh = np.zeros_like(y_prob_eval)
        for idx_val, val in enumerate(y_true_eval):
            if not np.isnan(val) and val in [0, 1, 2]:
                y_true_oh[idx_val, int(val)] = 1

        brier_score = np.mean(np.sum((y_prob_eval - y_true_oh)**2, axis=1))
        eps = 1e-15
        log_loss_val = -np.mean(np.sum(y_true_oh * np.log(np.clip(y_prob_eval, eps, 1 - eps)), axis=1))

        logger.info("=== MÉTRICAS DE CALIBRACIÓN GLOBAL ===")
        logger.info(f"Log Loss: {log_loss_val:.4f} | Brier Score Global: {brier_score:.4f}")

    # EJECUCIÓN DEL MOTOR
    res = run_bankroll_engine(df)

    liquid_bankroll = res['liquid_bankroll']
    total_staked = res['total_staked']
    bets_placed = res['bets_placed']
    bets_won = res['bets_won']
    total_analyzed_matches = res['total_analyzed_matches']
    total_expected_profit = res['total_expected_profit']
    clv_list = res['clv_list']
    gross_profit_sum = res['gross_profit_sum']
    gross_loss_sum = res['gross_loss_sum']
    avg_odds_list = res['avg_odds_list']
    daily_multipliers = res['daily_multipliers']
    league_stats = res['league_stats']
    placed_bets_history = res['placed_bets_history']
    historical_mdd = res['historical_mdd']
    bankroll_history = res['bankroll_history']

    yield_pct = ((liquid_bankroll - 1000.0) / total_staked) * 100 if total_staked > 0 else 0
    roi_pct = ((liquid_bankroll - 1000.0) / 1000.0) * 100
    x_yield_pct = (total_expected_profit / total_staked) * 100 if total_staked > 0 else 0

    avg_clv = np.mean(clv_list) * 100 if len(clv_list) > 0 else 0.0
    median_clv = np.median(clv_list) * 100 if len(clv_list) > 0 else 0.0
    beat_close_rate = (sum(1 for clv in clv_list if clv > 0) / len(clv_list)) * 100 if len(clv_list) > 0 else 0.0

    avg_odds = np.mean(avg_odds_list) if len(avg_odds_list) > 0 else 0.0
    profit_factor = gross_profit_sum / gross_loss_sum if gross_loss_sum > 0 else float('inf')

    daily_returns = [mult - 1.0 for mult in daily_multipliers]
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
    logger.info(f"Fricción Simulada (Impuestos / Market Impact): {TAX_RETENTION_RATE*100:.2f}% / Gamma={MARKET_IMPACT_GAMMA}")
    logger.info(f"Yield Real (Beneficio Neto / Turnover): {yield_pct:.2f}% | Expected Yield (xYield): {x_yield_pct:.2f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f} | ROI del Capital Inicial: {roi_pct:.2f}%")
    logger.info(f"Maximum Drawdown Histórico Real: {historical_mdd*100:.2f}%")
    logger.info(f"Ratio de Sharpe Anualizado: {sharpe_ratio:.2f} | Ratio de Sortino: {sortino_ratio:.2f}")

    if 'closing_odds_win' in df.columns:
        logger.info("=== ANÁLISIS DE CLOSING LINE VALUE (CLV) ===")
        logger.info(f"Promedio CLV: {avg_clv:.2f}% | Mediana CLV: {median_clv:.2f}% | Beat The Close Rate: {beat_close_rate:.1f}%")

    logger.info("=== RENDIMIENTO POR LIGA (CON CONTRACCIÓN BAYESIANA EMPÍRICA) ===")
    M_bayes = 30.0  # Peso del prior global
    for comp, stats in sorted(league_stats.items(), key=lambda x: x[1]['profit'], reverse=True):
        if stats['bets'] > 0:
            raw_yield = (stats['profit'] / stats['staked']) * 100
            w_b = stats['bets'] / (stats['bets'] + M_bayes)
            shrunken_yield = w_b * raw_yield + (1.0 - w_b) * yield_pct

            l_winrate = (stats['won'] / stats['bets']) * 100
            l_clv = (np.mean(stats['clv_list']) * 100) if stats.get('clv_list') else 0.0
            clv_str = f" | CLV: {l_clv:.2f}%" if stats.get('clv_list') else ""
            k_val = get_param_by_comp(KELLY_FRACTIONS, comp, 0.015)
            ev_val = get_param_by_comp(EV_THRESHOLDS, comp, 0.015)
            logger.info(f"Liga {comp:4s}: {stats['bets']:3d} apuestas | WinRate: {l_winrate:4.1f}% | Raw Yield: {raw_yield:6.2f}% | Bayes Yield: {shrunken_yield:6.2f}% | Profit: ${stats['profit']:7.2f} | Kelly: {k_val:.3f}{clv_str}")

    # PRUEBA MONTE CARLO BOOTSTRAPPING
    if len(daily_multipliers) > 10:
        logger.info("=== PRUEBA DE RESISTENCIA (MONTE CARLO) ===")
        n_sims = 10000
        ruin_count = 0
        max_drawdowns, final_capitals = [], []
        for _ in range(n_sims):
            sim_bankroll, peak, max_dd, is_ruined = 1000.0, 1000.0, 0.0, False
            for mult in random.choices(daily_multipliers, k=len(daily_multipliers)):
                sim_bankroll *= mult
                if sim_bankroll > peak: peak = sim_bankroll
                dd = (peak - sim_bankroll) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
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

        logger.info(f"Probabilidad de Ruina (PoR): {por:.2f}% | MDD Promedio: {avg_mdd:.2f}% | MDD P95 Tail Risk: {p95_mdd:.2f}% | Capital Mediano: ${median_cap:.2f}")

    # MONTE CARLO DE CALIBRACIÓN VECTORIZADO
    if placed_bets_history:
        df_bets = pd.DataFrame(placed_bets_history)
        n_sims = 10000
        probs_array = df_bets['prob'].values
        odds_array = df_bets['odds'].values
        stake_pcts_array = df_bets['stake_pct'].values
        N_bets = len(probs_array)

        sim_wins = np.random.rand(n_sims, N_bets) < probs_array
        win_mults = 1.0 + stake_pcts_array * (odds_array - 1.0) * (1.0 - TAX_RETENTION_RATE)
        loss_mults = 1.0 - stake_pcts_array

        mults_matrix = np.where(sim_wins, win_mults, loss_mults)
        bankroll_paths = 1000.0 * np.cumprod(mults_matrix, axis=1)

        bankroll_paths_full = np.hstack([np.full((n_sims, 1), 1000.0), bankroll_paths])
        peaks = np.maximum.accumulate(bankroll_paths_full, axis=1)
        drawdowns = (peaks - bankroll_paths_full) / peaks
        synthetic_mdds = np.max(drawdowns, axis=1) * 100.0
        final_capitals = bankroll_paths[:, -1]
        synthetic_yields = ((final_capitals - 1000.0) / 1000.0) * 100.0

        percentile_yield = np.sum(synthetic_yields < yield_pct) / n_sims * 100.0
        percentile_mdd = np.sum(synthetic_mdds < (historical_mdd * 100.0)) / n_sims * 100.0

        logger.info("=== MONTE CARLO DE CALIBRACIÓN (EXPECTED DISTRIBUTION) ===")
        logger.info(f"Yield Real: {yield_pct:.2f}% | xYield Mediano (Sim): {np.median(synthetic_yields):.2f}% | Percentil Yield: {percentile_yield:.1f}%")
        logger.info(f"MDD Real: {historical_mdd*100:.2f}% | xMDD Mediano (Sim): {np.median(synthetic_mdds):.2f}% | Percentil MDD: {percentile_mdd:.1f}%")

        # GENERACIÓN Y EXPORTACIÓN DE GRÁFICAS DE EXECUTIVE DASHBOARD
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Subplot 1: Equity Curve
            ax1.plot(bankroll_history, color='#1f77b4', linewidth=1.8, label='Liquid Bankroll ($)')
            ax1.axhline(1000.0, color='gray', linestyle='--', alpha=0.6, label='Initial Capital ($1000)')
            ax1.set_title('Prophetia2 Quant Simulator - Equity Curve & Drawdown Waterfall', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Bankroll ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Drawdown Curve
            bh_arr = np.array(bankroll_history)
            peaks_arr = np.maximum.accumulate(bh_arr)
            dd_arr = (peaks_arr - bh_arr) / peaks_arr * 100.0
            ax2.fill_between(range(len(dd_arr)), 0, -dd_arr, color='#d62728', alpha=0.4, label='Drawdown (%)')
            ax2.plot(-dd_arr, color='#d62728', linewidth=1.0)
            ax2.set_xlabel('Bets Placed (Sequence)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim([-max(dd_arr)*1.1 - 1.0, 1.0])
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            dash_plot_path = os.path.join(log_dir, f'equity_drawdown_plot_{timestamp_str}.png')
            plt.savefig(dash_plot_path, dpi=150)
            plt.close()
            logger.info(f"Dashboard de Equity & Drawdown generado en: {dash_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar gráfica de dashboard: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_simulation()
