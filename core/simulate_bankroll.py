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

FILTER_BY_WHITELIST = False  # True: ignora partidos fuera de Whitelist en la simulación financiera

from core.league_mapping import CANONICAL_LEAGUES, COMPETITION_MAPPING, WHITELIST_LEAGUES, normalize_league
from core.markowitz_optimizer import MarkowitzPortfolioOptimizer

def is_in_whitelist(comp):
    if not FILTER_BY_WHITELIST:
        return True
    return normalize_league(comp) in WHITELIST_LEAGUES


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
    'MEX': 1500.0,
    'DEFAULT': 2000.0
}

# CONFIGURACIÓN DE OPTIMIZACIÓN
OPTIMIZATION_MODE = 'MEX'  # 'NONE', 'ALL', 'WHITELIST', o liga específica
OPTUNA_TRIALS = 10000
OPTIMIZED_PARAMS_FILE = '../data/processed/models_best_parameters/optimal_bankroll_params.json'

# Diccionarios de riesgo por defecto
KELLY_FRACTIONS = {'D2': 0.03, 'I1': 0.01, 'SP1': 0.03, 'F2': 0.02, 'G1': 0.01, 'D1': 0.02, 'T1': 0.03, 'F1': 0.02, 'E1': 0.02, 'N1': 0.01, 'SP2': 0.01, 'P1': 0.01, 'MEX': 0.02, 'DEFAULT': 0.015}
EV_THRESHOLDS = {'D2': 0.015, 'I1': 0.02, 'SP1': 0.01, 'F2': 0.015, 'G1': 0.02, 'D1': 0.015, 'T1': 0.01, 'F1': 0.015, 'E1': 0.015, 'N1': 0.02, 'SP2': 0.02, 'P1': 0.02, 'MEX': 0.015, 'DEFAULT': 0.015}

ALPHA_DIV_LOW = {'DEFAULT': 0.85}
ALPHA_DIV_MED = {'DEFAULT': 0.70}
ALPHA_DIV_HIGH = {'DEFAULT': 0.50}

def get_param_by_comp(param_dict, comp, default_val=0.015):
    """ Busca un parámetro usando el nombre canónico de la competición """
    canonical = normalize_league(comp)
    return param_dict.get(canonical, param_dict.get('DEFAULT', default_val))

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
            if not is_in_whitelist(comp):
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

        # 2. CÁLCULO DE STAKES Y GESTIÓN DE RIESGO DE PORTAFOLIO SIMULTÁNEO (HARRY MARKOWITZ 1952)
        current_dd = (historical_peak - start_of_day_bankroll) / historical_peak if historical_peak > 0 else 0
        reactive_risk_factor = max(0.25, 1.0 - (current_dd / MAX_DRAWDOWN_TARGET))

        ev_vec = []
        odds_vec = []
        prob_vec = []
        penalties = []
        liquidity_caps = []
        bet_metadata = []

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
                prob_est = (1.0 + best_ev) / net_odd if net_odd > 0 else 0.5
                ev_val = best_ev
                odd_val = net_odd
            else: # dutching
                total_implied = (1.0 / odds[best_choice]) + (1.0 / odds[secondary_choice])
                combined_odds = 1.0 / total_implied
                net_combined_odds = 1.0 + (combined_odds - 1.0) * (1.0 - TAX_RETENTION_RATE)
                prob_est = (1.0 + best_ev) / net_combined_odds if net_combined_odds > 0 else 0.5
                ev_val = best_ev
                odd_val = net_combined_odds

            ev_vec.append(ev_val)
            odds_vec.append(odd_val)
            prob_vec.append(prob_est)
            penalties.append(adj_kelly_fraction / 0.015)
            liquidity_caps.append(max_liquidity)
            bet_metadata.append(bet)

        markowitz = MarkowitzPortfolioOptimizer(
            risk_aversion=2.0,
            max_stake_pct=MAX_STAKE_PCT,
            max_daily_portfolio_pct=MAX_DAILY_PORTFOLIO_PCT
        )

        m_res = markowitz.optimize_mean_variance(
            ev_vec=ev_vec,
            odds_vec=odds_vec,
            prob_vec=prob_vec,
            bankroll=start_of_day_bankroll,
            uncertainty_penalties=penalties,
            liquidity_caps=liquidity_caps
        )

        unscaled_bets = []
        total_daily_unscaled_stake = 0.0

        for i, bet in enumerate(bet_metadata):
            target_stake = float(m_res['stakes'][i]) if i < len(m_res['stakes']) else 0.0
            if target_stake < 0.1:
                continue
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

                true_clv = np.nan
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

                outcome_labels = {0: 'Away (2)', 1: 'Draw (X)', 2: 'Home (1)'}
                placed_bets_history.append({
                    'date': str(current_date),
                    'comp': comp,
                    'bet_type': 'single',
                    'chosen_label': outcome_labels.get(best_choice, str(best_choice)),
                    'ev': best_ev,
                    'prob': bet['blended_probs'][best_choice],
                    'odds': eff_odds,
                    'stake': stake,
                    'stake_pct': stake / start_of_day_bankroll if start_of_day_bankroll > 0 else 0,
                    'is_win': int(real_outcome == best_choice),
                    'net_profit': net_profit_record,
                    'clv': true_clv
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

                true_clv = np.nan
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

                dutch_labels = {(2, 1): 'Home or Draw (1X)', (0, 1): 'Away or Draw (X2)'}
                placed_bets_history.append({
                    'date': str(current_date),
                    'comp': comp,
                    'bet_type': 'dutching',
                    'chosen_label': dutch_labels.get((best_choice, secondary_choice), 'Dutching'),
                    'ev': best_ev,
                    'prob': bet['blended_probs'][best_choice] + bet['blended_probs'][secondary_choice],
                    'odds': eff_combined_odds,
                    'stake': stake,
                    'stake_pct': stake / start_of_day_bankroll if start_of_day_bankroll > 0 else 0,
                    'is_win': int(real_outcome in [best_choice, secondary_choice]),
                    'net_profit': net_profit_record,
                    'clv': true_clv
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
                if not is_in_whitelist(comp): continue
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
    df_eval = df[df['competition'].apply(is_in_whitelist)].copy() if FILTER_BY_WHITELIST else df.copy()
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

        # GENERACIÓN Y EXPORTACIÓN AUTOMÁTICA DE REPORTES Y DASHBOARDS (PNG + CSV)
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        plot_dir = os.path.join(log_dir, 'plots')
        csv_dir = os.path.join(log_dir, 'plots', 'csv')
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. CALIBRATION DATA & RELIABILITY DIAGRAM (CSV + PNG)
        try:
            bins = np.linspace(0.0, 1.0, 11)
            df_bets['prob_bin'] = pd.cut(df_bets['prob'], bins=bins, include_lowest=True)
            
            calib_records = []
            for bin_interval, group in df_bets.groupby('prob_bin', observed=False):
                count = len(group)
                if count > 0:
                    pred_mean = group['prob'].mean()
                    win_rate = group['is_win'].mean()
                    staked = group['stake'].sum()
                    profit = group['net_profit'].sum()
                    brier = np.mean((group['prob'] - group['is_win']) ** 2)
                else:
                    pred_mean = bin_interval.mid
                    win_rate = np.nan
                    staked = 0.0
                    profit = 0.0
                    brier = np.nan
                
                calib_records.append({
                    'bin_lower': bin_interval.left,
                    'bin_upper': bin_interval.right,
                    'bin_midpoint': bin_interval.mid,
                    'predicted_prob_mean': pred_mean,
                    'actual_win_rate': win_rate,
                    'sample_count': count,
                    'total_staked': staked,
                    'total_profit': profit,
                    'brier_score': brier
                })
            
            calib_df = pd.DataFrame(calib_records)
            calib_csv_path = os.path.join(csv_dir, f'calibration_data_{timestamp_str}.csv')
            calib_df.to_csv(calib_csv_path, index=False)
            logger.info(f"Datos de calibración exportados exitosamente en: {calib_csv_path}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            valid_calib = calib_df.dropna(subset=['actual_win_rate'])
            
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (y = x)', alpha=0.7)
            ax1.plot(valid_calib['predicted_prob_mean'], valid_calib['actual_win_rate'], 's-', color='#2ca02c', linewidth=2.0, markersize=7, label='Empirical Win Rate')
            ax1.set_title(f'Prophetia2 - Model Calibration & Reliability Curve ({timestamp_str})', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Empirical Win Rate')
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            ax2.bar(valid_calib['bin_midpoint'], valid_calib['sample_count'], width=0.07, color='#1f77b4', alpha=0.7)
            ax2.set_xlabel('Predicted Probability Bin')
            ax2.set_ylabel('Bet Count')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            calib_plot_path = os.path.join(plot_dir, f'calibration_plot_{timestamp_str}.png')
            plt.savefig(calib_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de calibración guardada en: {calib_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar exportación de calibración: {e}")

        # 2. EQUITY & DRAWDOWN (CSV + PNG)
        try:
            bh_arr = np.array(bankroll_history)
            peaks_arr = np.maximum.accumulate(bh_arr)
            dd_arr = (peaks_arr - bh_arr) / peaks_arr * 100.0

            eq_df = pd.DataFrame({
                'sequence_index': np.arange(len(bh_arr)),
                'bankroll': bh_arr,
                'peak_bankroll': peaks_arr,
                'drawdown_pct': dd_arr
            })
            eq_csv_path = os.path.join(csv_dir, f'equity_drawdown_data_{timestamp_str}.csv')
            eq_df.to_csv(eq_csv_path, index=False)
            logger.info(f"Datos de curva de equidad exportados en: {eq_csv_path}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(bankroll_history, color='#1f77b4', linewidth=1.8, label='Liquid Bankroll ($)')
            ax1.axhline(1000.0, color='gray', linestyle='--', alpha=0.6, label='Initial Capital ($1000)')
            ax1.set_title('Prophetia2 Quant Simulator - Equity Curve & Drawdown Waterfall', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Bankroll ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            ax2.fill_between(range(len(dd_arr)), 0, -dd_arr, color='#d62728', alpha=0.4, label='Drawdown (%)')
            ax2.plot(-dd_arr, color='#d62728', linewidth=1.0)
            ax2.set_xlabel('Bets Placed (Sequence)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim([-max(dd_arr)*1.1 - 1.0, 1.0])
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            dash_plot_path = os.path.join(plot_dir, f'equity_drawdown_plot_{timestamp_str}.png')
            plt.savefig(dash_plot_path, dpi=150)
            plt.close()
            logger.info(f"Dashboard de Equity & Drawdown generado en: {dash_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar dashboard de equity & drawdown: {e}")

        # 3. MONTE CARLO DISTRIBUTION (CSV + PNG)
        try:
            mc_df = pd.DataFrame({
                'sim_id': np.arange(n_sims),
                'final_bankroll': final_capitals,
                'yield_pct': synthetic_yields,
                'max_drawdown_pct': synthetic_mdds
            })
            mc_csv_path = os.path.join(csv_dir, f'monte_carlo_data_{timestamp_str}.csv')
            mc_df.to_csv(mc_csv_path, index=False)
            logger.info(f"Datos de Monte Carlo exportados en: {mc_csv_path}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.hist(final_capitals, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
            ax1.axvline(np.median(final_capitals), color='red', linestyle='--', linewidth=1.5, label=f'Median (${np.median(final_capitals):.2f})')
            ax1.axvline(np.percentile(final_capitals, 5), color='orange', linestyle=':', linewidth=1.5, label=f'P5 (${np.percentile(final_capitals, 5):.2f})')
            ax1.set_title('Monte Carlo Final Bankroll Distribution', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Final Bankroll ($)')
            ax1.set_ylabel('Frequency')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            ax2.hist(synthetic_mdds, bins=50, color='#d62728', edgecolor='black', alpha=0.7)
            ax2.axvline(np.median(synthetic_mdds), color='black', linestyle='--', linewidth=1.5, label=f'Median MDD ({np.median(synthetic_mdds):.2f}%)')
            ax2.axvline(np.percentile(synthetic_mdds, 95), color='darkred', linestyle=':', linewidth=1.5, label=f'P95 Tail Risk ({np.percentile(synthetic_mdds, 95):.2f}%)')
            ax2.set_title('Monte Carlo Max Drawdown Distribution', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Max Drawdown (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            mc_plot_path = os.path.join(plot_dir, f'monte_carlo_distribution_plot_{timestamp_str}.png')
            plt.savefig(mc_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de distribución de Monte Carlo guardada en: {mc_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar distribución de Monte Carlo: {e}")

        # 4. LEAGUE PERFORMANCE BREAKDOWN (CSV + PNG)
        try:
            league_records = []
            for comp, stats in league_stats.items():
                b_cnt = stats['bets']
                if b_cnt == 0: continue
                w_rate = (stats['won'] / b_cnt) * 100.0
                r_yield = (stats['profit'] / stats['staked'] * 100.0) if stats['staked'] > 0 else 0.0
                w_b = b_cnt / (b_cnt + 30.0)
                bayes_y = float(w_b * r_yield + (1.0 - w_b) * yield_pct)
                avg_clv = float(np.mean(stats['clv_list']) * 100.0) if stats['clv_list'] else 0.0
                k_val = float(get_param_by_comp(KELLY_FRACTIONS, comp, 0.015))
                
                league_records.append({
                    'competition': comp,
                    'bets_placed': b_cnt,
                    'win_rate_pct': w_rate,
                    'raw_yield_pct': r_yield,
                    'bayes_yield_pct': bayes_y,
                    'net_profit_usd': stats['profit'],
                    'kelly_fraction': k_val,
                    'avg_clv_pct': avg_clv
                })
            
            league_df = pd.DataFrame(league_records).sort_values(by='net_profit_usd', ascending=False)
            league_csv_path = os.path.join(csv_dir, f'league_performance_data_{timestamp_str}.csv')
            league_df.to_csv(league_csv_path, index=False)
            logger.info(f"Datos de rendimiento por liga exportados en: {league_csv_path}")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(league_df['competition'], league_df['net_profit_usd'], color=np.where(league_df['net_profit_usd'] >= 0, '#2ca02c', '#d62728'))
            ax.set_title('Net Profit ($) by Competition / League', fontsize=12, fontweight='bold')
            ax.set_xlabel('Net Profit ($)')
            ax.set_ylabel('League')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            league_plot_path = os.path.join(plot_dir, f'league_performance_plot_{timestamp_str}.png')
            plt.savefig(league_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de rendimiento por liga guardada en: {league_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar rendimiento por liga: {e}")

        # 5. EV BUCKETING REPORT (CSV + PNG)
        try:
            ev_bins = [-np.inf, 0.02, 0.05, 0.10, 0.15, np.inf]
            ev_labels = ['0 - 2%', '2 - 5%', '5 - 10%', '10 - 15%', '15%+']
            df_bets['ev_bucket'] = pd.cut(df_bets['ev'], bins=ev_bins, labels=ev_labels, right=False)

            ev_records = []
            for label in ev_labels:
                group = df_bets[df_bets['ev_bucket'] == label]
                cnt = len(group)
                if cnt > 0:
                    w_rate = group['is_win'].mean() * 100.0
                    staked = group['stake'].sum()
                    profit = group['net_profit'].sum()
                    yield_val = (profit / staked * 100.0) if staked > 0 else 0.0
                    avg_clv_val = group['clv'].mean() * 100.0 if 'clv' in group and group['clv'].notna().any() else 0.0
                else:
                    w_rate, staked, profit, yield_val, avg_clv_val = 0.0, 0.0, 0.0, 0.0, 0.0
                
                ev_records.append({
                    'ev_range': label,
                    'bets_placed': cnt,
                    'win_rate_pct': w_rate,
                    'total_staked_usd': staked,
                    'net_profit_usd': profit,
                    'yield_pct': yield_val,
                    'avg_clv_pct': avg_clv_val
                })

            ev_df = pd.DataFrame(ev_records)
            ev_csv_path = os.path.join(csv_dir, f'ev_bucketing_data_{timestamp_str}.csv')
            ev_df.to_csv(ev_csv_path, index=False)
            logger.info(f"Datos de rangos de EV exportados en: {ev_csv_path}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.bar(ev_df['ev_range'], ev_df['net_profit_usd'], color=np.where(ev_df['net_profit_usd'] >= 0, '#2ca02c', '#d62728'), alpha=0.8)
            ax1.set_title('Net Profit ($) by EV Range', fontsize=11, fontweight='bold')
            ax1.set_xlabel('EV Range')
            ax1.set_ylabel('Net Profit ($)')
            ax1.grid(True, alpha=0.3)

            ax2.bar(ev_df['ev_range'], ev_df['yield_pct'], color='#1f77b4', alpha=0.8)
            ax2.set_title('Yield (%) by EV Range', fontsize=11, fontweight='bold')
            ax2.set_xlabel('EV Range')
            ax2.set_ylabel('Yield (%)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            ev_plot_path = os.path.join(plot_dir, f'ev_bucketing_plot_{timestamp_str}.png')
            plt.savefig(ev_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de rangos de EV guardada en: {ev_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar reporte de rangos de EV: {e}")

        # 6. ODDS BUCKETING REPORT (CSV + PNG)
        try:
            odds_bins = [1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
            odds_labels = [
                'Favoritos Fuertes (1.00 - 1.50)',
                'Favoritos Moderados (1.50 - 2.00)',
                'Underdogs Moderados (2.00 - 3.00)',
                'Underdogs Altos (3.00 - 5.00)',
                'Sorpresas / Longshots (5.00+)'
            ]
            df_bets['odds_bucket'] = pd.cut(df_bets['odds'], bins=odds_bins, labels=odds_labels, right=False)

            odds_records = []
            for label in odds_labels:
                group = df_bets[df_bets['odds_bucket'] == label]
                cnt = len(group)
                if cnt > 0:
                    w_rate = group['is_win'].mean() * 100.0
                    staked = group['stake'].sum()
                    profit = group['net_profit'].sum()
                    yield_val = (profit / staked * 100.0) if staked > 0 else 0.0
                    avg_clv_val = group['clv'].mean() * 100.0 if 'clv' in group and group['clv'].notna().any() else 0.0
                else:
                    w_rate, staked, profit, yield_val, avg_clv_val = 0.0, 0.0, 0.0, 0.0, 0.0

                odds_records.append({
                    'odds_range': label,
                    'bets_placed': cnt,
                    'win_rate_pct': w_rate,
                    'total_staked_usd': staked,
                    'net_profit_usd': profit,
                    'yield_pct': yield_val,
                    'avg_clv_pct': avg_clv_val
                })

            odds_df = pd.DataFrame(odds_records)
            odds_csv_path = os.path.join(csv_dir, f'odds_bucketing_data_{timestamp_str}.csv')
            odds_df.to_csv(odds_csv_path, index=False)
            logger.info(f"Datos de rangos de cuotas exportados en: {odds_csv_path}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.barh(odds_df['odds_range'], odds_df['net_profit_usd'], color=np.where(odds_df['net_profit_usd'] >= 0, '#2ca02c', '#d62728'), alpha=0.8)
            ax1.set_title('Net Profit ($) by Odds Range', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Net Profit ($)')
            ax1.set_ylabel('Odds Range')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3)

            ax2.barh(odds_df['odds_range'], odds_df['yield_pct'], color='#1f77b4', alpha=0.8)
            ax2.set_title('Yield (%) by Odds Range', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Yield (%)')
            ax2.set_ylabel('Odds Range')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            odds_plot_path = os.path.join(plot_dir, f'odds_bucketing_plot_{timestamp_str}.png')
            plt.savefig(odds_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de rangos de cuotas guardada en: {odds_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar reporte de rangos de cuotas: {e}")

        # 7. BET TYPES & STRATEGIES REPORT (CSV + PNG)
        try:
            type_records = []
            for b_type in ['single', 'dutching']:
                group = df_bets[df_bets['bet_type'] == b_type]
                cnt = len(group)
                if cnt > 0:
                    w_rate = group['is_win'].mean() * 100.0
                    staked = group['stake'].sum()
                    profit = group['net_profit'].sum()
                    yield_val = (profit / staked * 100.0) if staked > 0 else 0.0
                    avg_clv_val = group['clv'].mean() * 100.0 if 'clv' in group and group['clv'].notna().any() else 0.0
                else:
                    w_rate, staked, profit, yield_val, avg_clv_val = 0.0, 0.0, 0.0, 0.0, 0.0

                label_name = 'Single Bet (1X2)' if b_type == 'single' else 'Dutching (1X / X2)'
                type_records.append({
                    'bet_type': label_name,
                    'bets_placed': cnt,
                    'win_rate_pct': w_rate,
                    'total_staked_usd': staked,
                    'net_profit_usd': profit,
                    'yield_pct': yield_val,
                    'avg_clv_pct': avg_clv_val
                })

            type_df = pd.DataFrame(type_records)
            type_csv_path = os.path.join(csv_dir, f'bet_types_data_{timestamp_str}.csv')
            type_df.to_csv(type_csv_path, index=False)
            logger.info(f"Datos de tipos de apuesta exportados en: {type_csv_path}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.bar(type_df['bet_type'], type_df['net_profit_usd'], color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax1.set_title('Net Profit ($) by Bet Type', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Net Profit ($)')
            ax1.grid(True, alpha=0.3)

            ax2.bar(type_df['bet_type'], type_df['yield_pct'], color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax2.set_title('Yield (%) by Bet Type', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Yield (%)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            type_plot_path = os.path.join(plot_dir, f'bet_types_plot_{timestamp_str}.png')
            plt.savefig(type_plot_path, dpi=150)
            plt.close()
            logger.info(f"Gráfica de tipos de apuesta guardada en: {type_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar reporte de tipos de apuesta: {e}")

        # 8. TRADES EXECUTION LEDGER (CSV)
        try:
            ledger_records = []
            for idx_row, row in df_bets.iterrows():
                ledger_records.append({
                    'bet_index': idx_row + 1,
                    'date': row.get('date', ''),
                    'competition': row.get('comp', ''),
                    'bet_type': row.get('bet_type', ''),
                    'chosen_label': row.get('chosen_label', ''),
                    'odds': row['odds'],
                    'expected_value_ev': row['ev'],
                    'predicted_prob': row['prob'],
                    'stake_usd': row['stake'],
                    'stake_pct_bankroll': row['stake_pct'] * 100.0,
                    'is_win': row['is_win'],
                    'net_profit_usd': row['net_profit'],
                    'clv_pct': (row['clv'] * 100.0) if pd.notna(row.get('clv')) else np.nan
                })

            ledger_df = pd.DataFrame(ledger_records)
            ledger_csv_path = os.path.join(csv_dir, f'trades_ledger_data_{timestamp_str}.csv')
            ledger_df.to_csv(ledger_csv_path, index=False)
            logger.info(f"Libro de operaciones (Trades Ledger) exportado exitosamente en: {ledger_csv_path}")
        except Exception as e:
            logger.error(f"Error al exportar libro de operaciones: {e}")

        # 9. MONTHLY PERFORMANCE ANALYSIS (CSV + PNG)
        try:
            if 'date' in df_bets.columns and df_bets['date'].notna().any():
                df_bets['month'] = pd.to_datetime(df_bets['date']).dt.to_period('M')
                monthly_records = []
                for m_period, group in df_bets.groupby('month'):
                    cnt = len(group)
                    staked = group['stake'].sum()
                    profit = group['net_profit'].sum()
                    w_rate = group['is_win'].mean() * 100.0
                    yield_val = (profit / staked * 100.0) if staked > 0 else 0.0
                    
                    monthly_records.append({
                        'month': str(m_period),
                        'bets_placed': cnt,
                        'win_rate_pct': w_rate,
                        'total_staked_usd': staked,
                        'net_profit_usd': profit,
                        'yield_pct': yield_val
                    })
                
                monthly_df = pd.DataFrame(monthly_records)
                monthly_csv_path = os.path.join(csv_dir, f'monthly_performance_data_{timestamp_str}.csv')
                monthly_df.to_csv(monthly_csv_path, index=False)
                logger.info(f"Datos de rendimiento mensual exportados en: {monthly_csv_path}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(monthly_df['month'], monthly_df['net_profit_usd'], color=np.where(monthly_df['net_profit_usd'] >= 0, '#2ca02c', '#d62728'), alpha=0.8)
                ax.set_title('Monthly Net Profit ($)', fontsize=11, fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Net Profit ($)')
                plt.xticks(rotation=45)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                monthly_plot_path = os.path.join(plot_dir, f'monthly_performance_plot_{timestamp_str}.png')
                plt.savefig(monthly_plot_path, dpi=150)
                plt.close()
                logger.info(f"Gráfica de rendimiento mensual guardada en: {monthly_plot_path}")
        except Exception as e:
            logger.error(f"Error al generar reporte de rendimiento mensual: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    run_simulation()
