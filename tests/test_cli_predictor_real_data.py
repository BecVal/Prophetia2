import os
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add core path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
core_dir = os.path.join(root_dir, 'core')
if core_dir not in sys.path:
    sys.path.append(core_dir)

from models.train_nn import SklearnPyTorchWrapper10, PyTorchGatedResNet10, GatedContext, ResidualBlock
from models.train_draws import HybridDrawsEnsemble
import __main__
__main__.SklearnPyTorchWrapper10 = SklearnPyTorchWrapper10
__main__.PyTorchGatedResNet10 = PyTorchGatedResNet10
__main__.GatedContext = GatedContext
__main__.ResidualBlock = ResidualBlock
__main__.SklearnPyTorchWrapper = SklearnPyTorchWrapper10
__main__.HybridDrawsEnsemble = HybridDrawsEnsemble

from cli_predictor import predict_match, load_all_models, load_data, TAX_RETENTION_RATE
from models.data_splitter import get_base_dataset, get_train_test_split

import logging
logging.disable(logging.INFO)

console = Console()

def run_cli_backtest_test():
    console.print(Panel.fit("[bold cyan]PROPHETIA2 - SUITE DE PRUEBAS DEL CLI PREDICTOR (PARTIDOS REALES)[/bold cyan]\n[dim]Evaluando Winrate, Net Profit y ROI en el conjunto de prueba Out-Of-Sample...[/dim]"))
    
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Evaluar los 250 partidos más recientes del set de prueba para una ejecución veloz y completa
    if len(test_df) > 250:
        test_df = test_df.tail(250).reset_index(drop=True)

    console.print(f"[info]Muestras evaluadas en conjunto de prueba Out-Of-Sample: [bold]{len(test_df)}[/bold] partidos[/info]")
    
    models_dict = load_all_models()
    if models_dict is None:
        console.print("[red]Error: No se pudieron cargar los modelos.[/red]")
        return
        
    bankroll = 10000.0
    initial_bankroll = bankroll
    
    total_bets = 0
    wins = 0
    losses = 0
    total_staked = 0.0
    total_profit = 0.0
    ev_list = []
    odds_list = []
    
    market_stats = {}

    for idx, row in test_df.iterrows():
        home_team = row['team']
        away_team = row['opponent']
        comp = row['competition']
        
        odds_1 = row.get('open_odds_win', row.get('odds_win', 0.0))
        odds_X = row.get('open_odds_draw', row.get('odds_draw', 0.0))
        odds_2 = row.get('open_odds_loss', row.get('odds_loss', 0.0))
        
        if pd.isna(odds_1) or pd.isna(odds_X) or pd.isna(odds_2) or odds_1 <= 1.01 or odds_X <= 1.01 or odds_2 <= 1.01:
            continue
            
        res = predict_match(
            home_team, away_team, comp, odds_1, odds_X, odds_2,
            bankroll=bankroll, df=df, models_dict=models_dict
        )
        
        if res is None or 'best_bet' not in res:
            continue
            
        best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']
        
        # Filtro de Staking: Solo apostar si supera el umbral de EV y el stake es > $1.00
        if best_ev > res['league_ev_thresh'] and best_stake >= 1.0:
            total_bets += 1
            total_staked += best_stake
            ev_list.append(best_ev)
            odds_list.append(best_raw_odd)
            
            # Evaluar resultado real del partido
            # outcome: 2 = Home win, 1 = Draw, 0 = Away win (Loss)
            actual_outcome = row.get('outcome')
            if actual_outcome == -1: actual_outcome = 0
            
            goals_home = row.get('goals_scored', 0)
            goals_away = row.get('goals_conceded', 0)
            total_goals = goals_home + goals_away
            
            is_win = False
            if best_bet_name.startswith("1 (Local)") and actual_outcome == 2:
                is_win = True
            elif best_bet_name.startswith("X (Empate)") and actual_outcome == 1:
                is_win = True
            elif best_bet_name.startswith("2 (Visitante)") and actual_outcome == 0:
                is_win = True
            elif best_bet_name.startswith("Doble Oportunidad 1X") and actual_outcome in [1, 2]:
                is_win = True
            elif best_bet_name.startswith("Doble Oportunidad X2") and actual_outcome in [0, 1]:
                is_win = True
            elif best_bet_name.startswith("Over 2.5 Goles") and total_goals > 2.5:
                is_win = True
            elif best_bet_name.startswith("BTTS") and goals_home > 0 and goals_away > 0:
                is_win = True

            net_odd = 1.0 + (best_raw_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
            
            if is_win:
                wins += 1
                payout = best_stake * net_odd
                profit = payout - best_stake
                total_profit += profit
                bankroll += profit
            else:
                losses += 1
                profit = -best_stake
                total_profit += profit
                bankroll += profit

            if best_bet_name not in market_stats:
                market_stats[best_bet_name] = {'bets': 0, 'wins': 0, 'staked': 0.0, 'profit': 0.0}
            
            market_stats[best_bet_name]['bets'] += 1
            market_stats[best_bet_name]['wins'] += 1 if is_win else 0
            market_stats[best_bet_name]['staked'] += best_stake
            market_stats[best_bet_name]['profit'] += profit

    winrate = (wins / total_bets * 100.0) if total_bets > 0 else 0.0
    roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    avg_ev = np.mean(ev_list) * 100.0 if ev_list else 0.0
    avg_odds = np.mean(odds_list) if odds_list else 0.0

    console.print("\n[bold green]=== RESULTADOS GENERALES DE LA PRUEBA DEL CLI PREDICTOR ===[/bold green]")
    console.print(f"Partidos Evaluados (Test Set): [bold]{len(test_df)}[/bold]")
    console.print(f"Apuestas Recomendadas Ejecutadas: [bold]{total_bets}[/bold]")
    console.print(f"Apuestas Ganadas: [bold green]{wins}[/bold green] | Apuestas Perdidas: [bold red]{losses}[/bold red]")
    console.print(f"Winrate Global: [bold yellow]{winrate:.2f}%[/bold yellow]")
    console.print(f"Capital Apostado Total: [bold]${total_staked:,.2f}[/bold]")
    console.print(f"Beneficio Neto (Profit): [bold green]${total_profit:,.2f}[/bold green]" if total_profit >= 0 else f"Beneficio Neto (Profit): [bold red]${total_profit:,.2f}[/bold red]")
    console.print(f"ROI Neto (Retorno sobre Inversión): [bold yellow]{roi:+.2f}%[/bold yellow]")
    console.print(f"Net Expected Value Promedio (EV): [bold]+{avg_ev:.2f}%[/bold]")
    console.print(f"Cuota Promedio Apostada: [bold]{avg_odds:.2f}[/bold]\n")

    t = Table(title="Desglose por Mercado Recomendado", show_header=True, header_style="bold magenta")
    t.add_column("Mercado", style="cyan")
    t.add_column("Apuestas", justify="right")
    t.add_column("Winrate", justify="right")
    t.add_column("Monto Apostado", justify="right")
    t.add_column("Profit ($)", justify="right")
    t.add_column("ROI (%)", justify="right")

    for m_name, m_data in market_stats.items():
        m_winrate = (m_data['wins'] / m_data['bets'] * 100.0) if m_data['bets'] > 0 else 0.0
        m_roi = (m_data['profit'] / m_data['staked'] * 100.0) if m_data['staked'] > 0 else 0.0
        profit_str = f"[green]+${m_data['profit']:,.2f}[/green]" if m_data['profit'] >= 0 else f"[red]-${abs(m_data['profit']):,.2f}[/red]"
        roi_str = f"[green]+{m_roi:.2f}%[/green]" if m_roi >= 0 else f"[red]{m_roi:.2f}%[/red]"
        t.add_row(m_name, str(m_data['bets']), f"{m_winrate:.1f}%", f"${m_data['staked']:,.2f}", profit_str, roi_str)

    console.print(t)

if __name__ == '__main__':
    run_cli_backtest_test()
