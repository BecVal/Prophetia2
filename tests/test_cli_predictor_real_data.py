import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import poisson
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add core path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
core_dir = os.path.join(root_dir, 'core')
if core_dir not in sys.path:
    sys.path.append(core_dir)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.train_nn import SklearnPyTorchWrapper10, PyTorchGatedResNet10, GatedContext, ResidualBlock
from models.train_draws import HybridDrawsEnsemble
from models.train_stacker import TemperatureScaledClassifier, RegimeGatedMetaStacker
from models.train_shots_on_goal import (
    DualPerspectiveShotsModel,
    PerspectiveDistributionalEnsemble,
    PyTorchMLPRegressor,
    SklearnPyTorchRegressorWrapper
)
import __main__
__main__.SklearnPyTorchWrapper10 = SklearnPyTorchWrapper10
__main__.PyTorchGatedResNet10 = PyTorchGatedResNet10
__main__.GatedContext = GatedContext
__main__.ResidualBlock = ResidualBlock
__main__.SklearnPyTorchWrapper = SklearnPyTorchWrapper10
__main__.HybridDrawsEnsemble = HybridDrawsEnsemble
__main__.TemperatureScaledClassifier = TemperatureScaledClassifier
__main__.RegimeGatedMetaStacker = RegimeGatedMetaStacker
__main__.DualPerspectiveShotsModel = DualPerspectiveShotsModel
__main__.PerspectiveDistributionalEnsemble = PerspectiveDistributionalEnsemble
__main__.PyTorchMLPRegressor = PyTorchMLPRegressor
__main__.SklearnPyTorchRegressorWrapper = SklearnPyTorchRegressorWrapper

from cli_predictor import predict_match, load_all_models, load_data, TAX_RETENTION_RATE
from models.data_splitter import get_base_dataset, get_train_test_split

import logging
logging.disable(logging.INFO)

console = Console()

def run_cli_backtest_test():
    console.print(Panel.fit(
        "[bold cyan]PROPHETIA2 - SUITE DE PRUEBAS DEL CLI PREDICTOR (PARTIDOS REALES MULTI-LIGA)[/bold cyan]\n"
        "[dim white]Evaluando Winrate, Net Profit, Drawdown y ROI en el conjunto Out-Of-Sample hasta Agosto 2026...[/dim white]",
        border_style="bright_blue"
    ))
    
    full_df = load_data()
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Mapeo de estadísticas del visitante desde el dataset completo (is_home == 0)
    df_away_map = full_df[full_df['is_home'] == 0].drop_duplicates('match_id').set_index('match_id')
    
    if 'match_date' in test_df.columns:
        test_df['match_date_dt'] = pd.to_datetime(test_df['match_date'])
        # Filtrar hasta el 23 de agosto de 2026
        test_df = test_df[test_df['match_date_dt'] <= '2026-08-23 23:59:59']
        test_df = test_df.sort_values('match_date_dt').reset_index(drop=True)
    
    # Evaluar los 250 partidos más recientes del set de prueba
    if len(test_df) > 250:
        test_df = test_df.tail(250).reset_index(drop=True)

    # Mapear córneres, tarjetas y tiros al arco totales por partido
    eval_away_corners = test_df['match_id'].map(df_away_map['corners']).fillna(0) if 'corners' in df_away_map.columns else 0.0
    test_df['away_corners'] = eval_away_corners
    test_df['total_corners'] = test_df['corners'].fillna(0) + eval_away_corners if 'corners' in test_df.columns else 0.0
    
    eval_away_yellows = test_df['match_id'].map(df_away_map['yellow_cards']).fillna(0) if 'yellow_cards' in df_away_map.columns else 0.0
    eval_away_reds = test_df['match_id'].map(df_away_map['red_cards']).fillna(0) if 'red_cards' in df_away_map.columns else 0.0
    test_df['away_yellows'] = eval_away_yellows
    test_df['away_reds'] = eval_away_reds
    test_df['total_cards'] = test_df['yellow_cards'].fillna(0) + (test_df['red_cards'].fillna(0) * 2) + eval_away_yellows + (eval_away_reds * 2)

    eval_away_sot = test_df['match_id'].map(df_away_map['shots_on_target']).fillna(0) if 'shots_on_target' in df_away_map.columns else 0.0
    test_df['away_sot'] = eval_away_sot
    test_df['total_sot'] = test_df['shots_on_target'].fillna(0) + eval_away_sot if 'shots_on_target' in test_df.columns else 0.0

    first_date = str(test_df['match_date'].iloc[0])[:10] if 'match_date' in test_df.columns and len(test_df) > 0 else "N/A"
    last_date = str(test_df['match_date'].iloc[-1])[:10] if 'match_date' in test_df.columns and len(test_df) > 0 else "N/A"

    console.print(f"[cyan]📊 Muestras evaluadas en conjunto de prueba Out-Of-Sample:[/cyan] [bold bright_yellow]{len(test_df)}[/bold bright_yellow] partidos (Período: [bold cyan]{first_date}[/bold cyan] al [bold cyan]{last_date}[/bold cyan])")
    
    models_dict = load_all_models()
    if models_dict is None:
        console.print("[red]Error: No se pudieron cargar los modelos.[/red]")
        return
        
    bankroll = 10000.0
    initial_bankroll = bankroll
    bankroll_history = [bankroll]
    
    total_bets = 0
    wins = 0
    losses = 0
    total_staked = 0.0
    total_profit = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    ev_list = []
    odds_list = []
    
    market_stats = {}

    for idx, row in test_df.iterrows():
        home_team = row['team']
        away_team = row['opponent']
        comp = row['competition']
        match_date = row.get('match_date')
        
        odds_1 = row.get('open_odds_win')
        if pd.isna(odds_1) or odds_1 <= 1.01:
            odds_1 = row.get('odds_win')
        if pd.isna(odds_1) or odds_1 <= 1.01:
            odds_1 = row.get('closing_odds_win')
            
        odds_X = row.get('open_odds_draw')
        if pd.isna(odds_X) or odds_X <= 1.01:
            odds_X = row.get('odds_draw')
        if pd.isna(odds_X) or odds_X <= 1.01:
            odds_X = row.get('closing_odds_draw')
            
        odds_2 = row.get('open_odds_loss')
        if pd.isna(odds_2) or odds_2 <= 1.01:
            odds_2 = row.get('odds_loss')
        if pd.isna(odds_2) or odds_2 <= 1.01:
            odds_2 = row.get('closing_odds_loss')
            
        try:
            odds_1 = float(odds_1)
            odds_X = float(odds_X)
            odds_2 = float(odds_2)
        except (TypeError, ValueError):
            continue
            
        if odds_1 <= 1.01 or odds_X <= 1.01 or odds_2 <= 1.01:
            continue

        open_w = row.get('open_odds_win')
        open_d = row.get('open_odds_draw')
        open_l = row.get('open_odds_loss')
        if pd.isna(open_w) or open_w <= 1.01: open_w = odds_1
        if pd.isna(open_d) or open_d <= 1.01: open_d = odds_X
        if pd.isna(open_l) or open_l <= 1.01: open_l = odds_2

        # Cuotas de Mercados Derivados (Totales O/U 2.5, BTTS, SoT 7.5/8.5, Córneres 9.5, Tarjetas 4.5)
        # Calibradas con margen de casa de apuestas institucional (~4.5% de vig) desde la línea implícita 1X2
        inv_1 = 1.0 / max(odds_1, 1.01)
        inv_x = 1.0 / max(odds_X, 1.01)
        inv_2 = 1.0 / max(odds_2, 1.01)
        m_sum = inv_1 + inv_x + inv_2
        p1_mkt = inv_1 / m_sum
        px_mkt = inv_x / m_sum
        p2_mkt = inv_2 / m_sum
        fav_gap = abs(p1_mkt - p2_mkt)

        lam_mkt_g = float(np.clip(2.60 - 1.20 * (px_mkt - 0.26) + 0.25 * fav_gap, 2.15, 3.10))
        p_ov25 = float(np.clip(1.0 - poisson.cdf(2, lam_mkt_g), 0.35, 0.65))
        p_un25 = 1.0 - p_ov25
        p_btts = float(np.clip(0.53 - 0.15 * fav_gap + 0.05 * (lam_mkt_g - 2.60), 0.40, 0.60))

        lam_sot = float(np.clip(8.35 + 0.35 * fav_gap, 7.80, 9.10))
        p_sot75 = float(np.clip(1.0 - poisson.cdf(7, lam_sot), 0.50, 0.72))
        p_sot85 = float(np.clip(1.0 - poisson.cdf(8, lam_sot), 0.38, 0.60))

        extra_odds = {
            'ou25_over': round(float(np.clip(1.0 / (p_ov25 * 1.045), 1.65, 2.50)), 2),
            'ou25_under': round(float(np.clip(1.0 / (p_un25 * 1.045), 1.55, 2.45)), 2),
            'btts_yes': round(float(np.clip(1.0 / (p_btts * 1.045), 1.65, 2.25)), 2),
            'corners_over95': 1.95,
            'corners_under95': 1.85,
            'cards_over45': 1.95,
            'sot_over75': round(float(np.clip(1.0 / (p_sot75 * 1.045), 1.45, 1.85)), 2),
            'sot_over85': round(float(np.clip(1.0 / (p_sot85 * 1.045), 1.75, 2.25)), 2),
        }
            
        res = predict_match(
            home_team, away_team, comp, odds_1, odds_X, odds_2,
            open_odds_win=float(open_w), open_odds_draw=float(open_d), open_odds_loss=float(open_l),
            extra_odds=extra_odds,
            bankroll=bankroll, df=full_df, models_dict=models_dict,
            as_of_date=match_date
        )
        
        if res is None or 'best_bet' not in res:
            continue
            
        best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']
        
        # Filtro de Staking: Solo apostar si supera el umbral de EV y el stake es > $1.00
        if best_ev > res['league_ev_thresh'] and best_stake >= 1.0:
            stake = min(best_stake, bankroll * 0.03)
            if bankroll - stake < 0:
                stake = bankroll
            if stake <= 0:
                break

            total_bets += 1
            total_staked += stake
            ev_list.append(best_ev)
            odds_list.append(best_raw_odd)
            
            # Evaluar resultado real del partido
            # outcome: 1 = Home win, 0 = Draw, -1 = Away win (Loss)
            actual_outcome = row.get('outcome')
            
            goals_home = row.get('goals_scored', 0)
            goals_away = row.get('goals_conceded', 0)
            try:
                goals_home = float(goals_home) if pd.notna(goals_home) else 0.0
                goals_away = float(goals_away) if pd.notna(goals_away) else 0.0
            except (TypeError, ValueError):
                goals_home, goals_away = 0.0, 0.0

            total_goals = goals_home + goals_away
            total_corners = float(row.get('total_corners', 0) or 0)
            total_cards = float(row.get('total_cards', 0) or 0)
            total_sot = float(row.get('total_sot', 0) or 0)
            
            is_win = False
            mkt_category = best_bet_name
            
            if best_bet_name.startswith("1 ") or "Local" in best_bet_name:
                mkt_category = "1 (Local)"
                if actual_outcome == 1 or goals_home > goals_away: is_win = True
            elif best_bet_name.startswith("X ") or "Empate" in best_bet_name:
                mkt_category = "X (Empate)"
                if actual_outcome == 0 or goals_home == goals_away: is_win = True
            elif best_bet_name.startswith("2 ") or "Visitante" in best_bet_name:
                mkt_category = "2 (Visitante)"
                if actual_outcome == -1 or goals_away > goals_home: is_win = True
            elif "Doble Oportunidad 1X" in best_bet_name or "1X" in best_bet_name:
                mkt_category = "Doble Oportunidad 1X"
                if actual_outcome in [1, 0] or goals_home >= goals_away: is_win = True
            elif "Doble Oportunidad X2" in best_bet_name or "X2" in best_bet_name:
                mkt_category = "Doble Oportunidad X2"
                if actual_outcome in [-1, 0] or goals_away >= goals_home: is_win = True
            elif "Over 2.5" in best_bet_name:
                mkt_category = "Over 2.5 Goles"
                if total_goals > 2.5: is_win = True
            elif "Under 2.5" in best_bet_name:
                mkt_category = "Under 2.5 Goles"
                if total_goals < 2.5: is_win = True
            elif "BTTS" in best_bet_name:
                mkt_category = "BTTS (Ambos Anotan)"
                if goals_home > 0 and goals_away > 0: is_win = True
            elif "Under 9.5" in best_bet_name:
                mkt_category = "Under 9.5 Córneres"
                if total_corners < 9.5: is_win = True
            elif "Over 9.5" in best_bet_name or "Córneres" in best_bet_name or "Corners" in best_bet_name:
                mkt_category = "Over 9.5 Córneres"
                if total_corners > 9.5: is_win = True
            elif "Tarjetas" in best_bet_name or "Cards" in best_bet_name:
                mkt_category = "Over 4.5 Tarjetas"
                if total_cards > 4.5: is_win = True
            elif "7.5 Tiros" in best_bet_name or "7.5 SoT" in best_bet_name:
                if "Over" in best_bet_name:
                    mkt_category = "Over 7.5 Tiros al Arco"
                    if total_sot > 7.5: is_win = True
                else:
                    mkt_category = "Under 7.5 Tiros al Arco"
                    if total_sot < 7.5: is_win = True
            elif "8.5 Tiros" in best_bet_name or "8.5 SoT" in best_bet_name:
                if "Over" in best_bet_name:
                    mkt_category = "Over 8.5 Tiros al Arco"
                    if total_sot > 8.5: is_win = True
                else:
                    mkt_category = "Under 8.5 Tiros al Arco"
                    if total_sot < 8.5: is_win = True

            net_odd = 1.0 + (best_raw_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
            bankroll -= stake
            
            if is_win:
                wins += 1
                gross_payout = stake * (best_eff_odd - 1.0)
                net_profit = gross_payout * (1.0 - TAX_RETENTION_RATE)
                bankroll += stake + net_profit
                total_profit += net_profit
                gross_profit += net_profit
                profit_record = net_profit
            else:
                losses += 1
                profit_record = -stake
                total_profit -= stake
                gross_loss += stake

            bankroll_history.append(bankroll)

            if mkt_category not in market_stats:
                market_stats[mkt_category] = {'bets': 0, 'wins': 0, 'staked': 0.0, 'profit': 0.0}
            
            market_stats[mkt_category]['bets'] += 1
            market_stats[mkt_category]['wins'] += 1 if is_win else 0
            market_stats[mkt_category]['staked'] += stake
            market_stats[mkt_category]['profit'] += profit_record

    # Métricas de Riesgo y Portafolio
    peak = bankroll_history[0]
    max_drawdown = 0.0
    for b in bankroll_history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    winrate = (wins / total_bets * 100.0) if total_bets > 0 else 0.0
    roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    avg_ev = np.mean(ev_list) * 100.0 if ev_list else 0.0
    avg_odds = np.mean(odds_list) if odds_list else 0.0
    growth_pct = ((bankroll - initial_bankroll) / initial_bankroll) * 100.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    console.print("\n[bold bright_green]=== RESULTADOS GENERALES DE LA PRUEBA DEL CLI PREDICTOR (REAL DATA) ===[/bold bright_green]")
    console.print(f"Billetera Inicial: [bold white]${initial_bankroll:,.2f} USD[/bold white]")
    console.print(f"Billetera Final: [bold bright_yellow]${bankroll:,.2f} USD[/bold bright_yellow] ([bold bright_green]{growth_pct:+.2f}% Crecimiento Total[/bold bright_green])")
    console.print(f"Partidos Evaluados (Test Set): [bold]{len(test_df)}[/bold]")
    console.print(f"Apuestas Ejecutadas: [bold]{total_bets}[/bold]")
    console.print(f"Apuestas Ganadas: [bold bright_green]{wins}[/bold bright_green] | Apuestas Perdidas: [bold bright_red]{losses}[/bold bright_red]")
    console.print(f"Winrate Global: [bold bright_cyan]{winrate:.2f}%[/bold bright_cyan]")
    console.print(f"Capital Apostado Total: [bold]${total_staked:,.2f}[/bold]")
    
    if total_profit >= 0:
        console.print(f"Beneficio Neto (Profit): [bold bright_green]+${total_profit:,.2f} USD[/bold bright_green]")
        console.print(f"ROI Neto (Retorno sobre Inversión): [bold bright_green]+{roi:.2f}%[/bold bright_green]")
    else:
        console.print(f"Beneficio Neto (Profit): [bold bright_red]-${abs(total_profit):,.2f} USD[/bold bright_red]")
        console.print(f"ROI Neto (Retorno sobre Inversión): [bold bright_red]{roi:.2f}%[/bold bright_red]")
        
    console.print(f"Profit Factor: [bold bright_cyan]{profit_factor:.2f}[/bold bright_cyan]")
    console.print(f"Máximo Drawdown (MDD): [bold yellow]{max_drawdown*100:.2f}%[/bold yellow]")
    console.print(f"EV Promedio de Apuestas: [bold]+{avg_ev:.2f}%[/bold]")
    console.print(f"Cuota Promedio Apostada: [bold]{avg_odds:.2f}[/bold]\n")

    if market_stats:
        t = Table(title="Desglose por Mercado Recomendado en Partidos Reales", show_header=True, header_style="bold magenta", border_style="magenta")
        t.add_column("Mercado", style="bold white")
        t.add_column("Apuestas", justify="right")
        t.add_column("Aciertos", justify="right")
        t.add_column("Winrate", justify="right")
        t.add_column("Monto Apostado", justify="right")
        t.add_column("Profit ($)", justify="right")
        t.add_column("ROI (%)", justify="right")

        for m_name, m_data in market_stats.items():
            m_winrate = (m_data['wins'] / m_data['bets'] * 100.0) if m_data['bets'] > 0 else 0.0
            m_roi = (m_data['profit'] / m_data['staked'] * 100.0) if m_data['staked'] > 0 else 0.0
            profit_str = f"[bold green]+${m_data['profit']:,.2f}[/bold green]" if m_data['profit'] >= 0 else f"[bold red]-${abs(m_data['profit']):,.2f}[/bold red]"
            roi_str = f"[bold green]+{m_roi:.2f}%[/bold green]" if m_roi >= 0 else f"[bold red]{m_roi:.2f}%[/bold red]"
            t.add_row(m_name, str(m_data['bets']), str(m_data['wins']), f"{m_winrate:.1f}%", f"${m_data['staked']:,.2f}", profit_str, roi_str)

        console.print(t)

if __name__ == '__main__':
    run_cli_backtest_test()
