# tests/test_all_leagues.py
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Configurar rutas del sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from core.cli_predictor import load_data, load_all_models, predict_match
from core.league_mapping import CANONICAL_LEAGUES, normalize_league, get_param_by_comp

# Parámetros institucionales
TAX_RETENTION_RATE = 0.0075
MAX_STAKE_PCT = 0.03

def evaluate_league(comp_name, df_all, models_dict, sample_size=100):
    """
    Evalúa el rendimiento de una liga específica con una billetera inicial de $1,000 USD
    utilizando el pipeline institucional completo de 8 modelos + Markowitz Multi-Mercado.
    """
    # Encontrar todos los alias de la liga
    canonical = normalize_league(comp_name)
    aliases = CANONICAL_LEAGUES.get(canonical, [comp_name])
    all_names = [canonical] + aliases + [comp_name]
    
    # Filtrar partidos válidos
    comp_df = df_all[
        df_all['competition'].isin(all_names) & 
        df_all['odds_win'].notna() & 
        df_all['outcome'].notna()
    ].copy()
    
    if comp_df.empty or len(comp_df) < 20:
        return None
        
    if 'match_date' in comp_df.columns:
        comp_df['match_date_dt'] = pd.to_datetime(comp_df['match_date'])
        comp_df = comp_df.sort_values('match_date_dt').reset_index(drop=True)
        
    eval_df = comp_df.tail(min(sample_size, len(comp_df))).reset_index(drop=True)
    
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    bets_placed = 0
    bets_won = 0
    total_staked = 0.0
    profit = 0.0
    market_breakdown = {}
    
    for idx, row in eval_df.iterrows():
        home_team = row['team']
        away_team = row['opponent']
        comp = row['competition']
        
        odds_1 = row.get('open_odds_win')
        if pd.isna(odds_1) or odds_1 <= 1.01:
            odds_1 = float(row.get('odds_win', 0.0))
            
        odds_X = row.get('open_odds_draw')
        if pd.isna(odds_X) or odds_X <= 1.01:
            odds_X = float(row.get('odds_draw', 0.0))
            
        odds_2 = row.get('open_odds_loss')
        if pd.isna(odds_2) or odds_2 <= 1.01:
            odds_2 = float(row.get('odds_loss', 0.0))
            
        if pd.isna(odds_1) or pd.isna(odds_X) or pd.isna(odds_2) or odds_1 <= 1.01 or odds_X <= 1.01 or odds_2 <= 1.01:
            continue
            
        res_pred = predict_match(
            home_team, away_team, comp, odds_1, odds_X, odds_2,
            open_odds_win=odds_1, open_odds_draw=odds_X, open_odds_loss=odds_2,
            bankroll=bankroll, df=df_all, models_dict=models_dict
        )
        
        if res_pred is None or 'best_bet' not in res_pred:
            continue
            
        best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, blend_prob = res_pred['best_bet']
        league_ev_thresh = res_pred.get('league_ev_thresh', 0.015)
        
        if best_ev > league_ev_thresh and best_stake >= 1.0:
            actual_outcome = row.get('outcome')
            if actual_outcome == -1: actual_outcome = 0
            
            goals_home = row.get('goals_scored', 0)
            goals_away = row.get('goals_conceded', 0)
            total_goals = goals_home + goals_away
            
            is_win = False
            mkt_label = best_bet_name
            
            if best_bet_name.startswith("1 ") or "Local" in best_bet_name:
                mkt_label = "1 (Local)"
                if actual_outcome == 2: is_win = True
            elif best_bet_name.startswith("X ") or "Empate" in best_bet_name:
                mkt_label = "X (Empate)"
                if actual_outcome == 1: is_win = True
            elif best_bet_name.startswith("2 ") or "Visitante" in best_bet_name:
                mkt_label = "2 (Visitante)"
                if actual_outcome in [0, -1]: is_win = True
            elif "Doble Oportunidad 1X" in best_bet_name:
                mkt_label = "Doble Oportunidad 1X"
                if actual_outcome in [1, 2]: is_win = True
            elif "Doble Oportunidad X2" in best_bet_name:
                mkt_label = "Doble Oportunidad X2"
                if actual_outcome in [0, 1, -1]: is_win = True
            elif "Over 2.5" in best_bet_name:
                mkt_label = "Over 2.5 Goles"
                if total_goals > 2.5: is_win = True
            elif "BTTS" in best_bet_name:
                mkt_label = "BTTS"
                if goals_home > 0 and goals_away > 0: is_win = True
                
            stake = min(best_stake, bankroll * MAX_STAKE_PCT)
            if bankroll - stake < 0:
                stake = bankroll
                if stake <= 0: break
                
            bankroll -= stake
            total_staked += stake
            bets_placed += 1
            
            if is_win:
                gross_profit = stake * (best_eff_odd - 1.0)
                net_profit = gross_profit * (1.0 - TAX_RETENTION_RATE)
                bankroll += stake + net_profit
                bets_won += 1
                profit += net_profit
            else:
                profit -= stake
                
            if mkt_label not in market_breakdown:
                market_breakdown[mkt_label] = {'bets': 0, 'won': 0, 'profit': 0.0}
            market_breakdown[mkt_label]['bets'] += 1
            market_breakdown[mkt_label]['won'] += (1 if is_win else 0)
            market_breakdown[mkt_label]['profit'] += (net_profit if is_win else -stake)

    winrate = (bets_won / bets_placed * 100.0) if bets_placed > 0 else 0.0
    roi = (profit / initial_bankroll * 100.0)
    yield_val = (profit / total_staked * 100.0) if total_staked > 0 else 0.0
    
    return {
        'comp': comp_name,
        'canonical': canonical,
        'matches_evaluated': len(eval_df),
        'bets_placed': bets_placed,
        'bets_won': bets_won,
        'winrate': winrate,
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': profit,
        'roi': roi,
        'yield': yield_val,
        'total_staked': total_staked,
        'markets': market_breakdown
    }

def main():
    print("=" * 105)
    print("PROPHETIA2 - SUITE DE VALIDACION INTEGRAL MULTI-LIGA (TODAS LAS 28 COMPETICIONES)")
    print("Evaluando el rendimiento cuantitativo con $1,000 USD de Billetera Inicial por Liga...")
    print("=" * 105)
    
    df = load_data()
    models_dict = load_all_models()
    
    # Lista exhaustiva de las 28 ligas canónicas y sus alias exactos
    target_leagues = [
        # Ligas Europeas Principales
        ('E0', 'Premier League'),
        ('SP1', 'La Liga'),
        ('D1', '1. Bundesliga'),
        ('I1', 'Serie A'),
        ('F1', 'Ligue 1'),
        ('E1', 'Championship'),
        ('SP2', 'La Liga 2'),
        ('D2', '2. Bundesliga'),
        ('F2', 'Ligue 2'),
        ('I2', 'Serie B'),
        ('B1', 'Jupiler Pro League'),
        ('N1', 'Eredivisie'),
        ('P1', 'Primeira Liga'),
        ('SC0', 'Scottish Premiership'),
        ('T1', 'Süper Lig'),
        ('G1', 'Super League'),
        ('E2', 'League One'),
        # América & Internacionales
        ('MLS', 'Major League Soccer'),
        ('MEX', 'Liga MX'),
        ('J1', 'J-League 1'),
        ('SWE', 'Allsvenskan'),
        ('NOR', 'Eliteserien'),
        ('DNK', 'Superligaen'),
        ('SWZ', 'Swiss Super League'),
        ('AUT', 'Austrian Bundesliga'),
        # Torneos Internacionales UEFA / FIFA
        ('CL', 'Champions League'),
        ('EL', 'Europa League'),
        ('WC', 'FIFA World Cup')
    ]
    
    results = []
    
    for canon, display_name in target_leagues:
        print(f"\n[+] Evaluando {display_name} (ID: {canon})...")
        res = evaluate_league(display_name, df, models_dict, sample_size=100)
        if res is None:
            res = evaluate_league(canon, df, models_dict, sample_size=100)
            
        if res:
            results.append(res)
            status_icon = "[OK] RENTABLE" if res['profit'] >= 0 else "[-] AJUSTE"
            print(f"    {status_icon} | Apuestas: {res['bets_placed']:3d} | Aciertos: {res['bets_won']:2d} | WinRate: {res['winrate']:4.1f}% | Profit: ${res['profit']:>+8.2f} USD | ROI: {res['roi']:>+6.2f}% | Final: ${res['final_bankroll']:7.2f}")
        else:
            print(f"    [!] Sin partidos con cuotas en el histórico para {display_name} ({canon}).")

    print("\n" + "=" * 105)
    print(f"{'LIGA / COMPETICION':<26} | {'ID':<5} | {'APUESTAS':<8} | {'WINRATE':<8} | {'PROFIT ($)':<12} | {'ROI (%)':<9} | {'YIELD (%)':<9} | {'ESTADO'}")
    print("=" * 105)
    
    total_bets = 0
    total_won = 0
    total_profit = 0.0
    total_staked = 0.0
    profitable_leagues = 0
    
    for r in results:
        total_bets += r['bets_placed']
        total_won += r['bets_won']
        total_profit += r['profit']
        total_staked += r['total_staked']
        if r['profit'] >= 0:
            profitable_leagues += 1
            
        status = "RENTABLE" if r['profit'] >= 0 else "AJUSTE"
        print(f"{r['comp']:<26} | {r['canonical']:<5} | {r['bets_placed']:<8} | {r['winrate']:<7.1f}% | {r['profit']:>+10.2f} $ | {r['roi']:>+7.2f}% | {r['yield']:>+7.2f}% | {status}")

    global_winrate = (total_won / total_bets * 100.0) if total_bets > 0 else 0.0
    global_yield = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    
    print("=" * 105)
    print(f"RESUMEN GLOBAL DEL PORTAFOLIO MULTI-LIGA (TODAS LAS 28 COMPETICIONES):")
    print(f"- Ligas Rentables: {profitable_leagues}/{len(results)} ({profitable_leagues/len(results)*100:.1f}%)")
    print(f"- Total Apuestas Realizadas: {total_bets}")
    print(f"- WinRate Promedio Global: {global_winrate:.2f}%")
    print(f"- Beneficio Neto Acumulado: ${total_profit:+.2f} USD")
    print(f"- Yield Global del Portafolio: {global_yield:+.2f}%")
    print("=" * 105)

if __name__ == '__main__':
    main()
