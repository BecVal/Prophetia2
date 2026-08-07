import os
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add core path for imports
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
import __main__
__main__.SklearnPyTorchWrapper10 = SklearnPyTorchWrapper10
__main__.PyTorchGatedResNet10 = PyTorchGatedResNet10
__main__.GatedContext = GatedContext
__main__.ResidualBlock = ResidualBlock
__main__.SklearnPyTorchWrapper = SklearnPyTorchWrapper10
__main__.HybridDrawsEnsemble = HybridDrawsEnsemble

from cli_predictor import predict_match, load_all_models, load_data, TAX_RETENTION_RATE
from models.data_splitter import get_base_dataset

import logging
logging.disable(logging.INFO)

console = Console()

def run_liga_mx_cli_test():
    console.print(Panel.fit(
        "[bold bright_yellow]PROPHETIA2 - SUITE DE PRUEBAS EXCLUSIVA LIGA MX[/bold bright_yellow]\n"
        "[dim white]Evaluando Winrate, Beneficio Neto y ROI con una Billetera Inicial de $1,000 USD...[/dim white]",
        border_style="bright_blue"
    ))
    
    df = load_data()
    if df is None:
        console.print("[red]Error: No se pudo cargar el dataset.[/red]")
        return
        
    # Filtrar únicamente partidos de Liga MX que tengan cuotas y resultado definido
    mex_df = df[
        df['competition'].isin(['Liga MX', 'MEX', 'MEX1']) & 
        df['odds_win'].notna() & 
        df['outcome'].notna()
    ].copy()
    
    if 'match_date' in mex_df.columns:
        mex_df['match_date_dt'] = pd.to_datetime(mex_df['match_date'])
        mex_df = mex_df.sort_values('match_date_dt').reset_index(drop=True)
    
    # Tomar los 150 partidos más recientes de Liga MX
    sample_size = min(150, len(mex_df))
    eval_df = mex_df.tail(sample_size).reset_index(drop=True)
    
    first_date = str(eval_df['match_date'].iloc[0])[:10] if 'match_date' in eval_df.columns else "N/A"
    last_date = str(eval_df['match_date'].iloc[-1])[:10] if 'match_date' in eval_df.columns else "N/A"
    
    console.print(f"[cyan]📊 Partidos evaluados de Liga MX:[/cyan] [bold bright_yellow]{sample_size}[/bold bright_yellow] (Período: [bold cyan]{first_date}[/bold cyan] a [bold cyan]{last_date}[/bold cyan])")
    
    models_dict = load_all_models()
    if models_dict is None:
        console.print("[red]Error: No se pudieron cargar los modelos del sistema.[/red]")
        return
        
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    
    total_bets = 0
    wins = 0
    losses = 0
    total_staked = 0.0
    total_profit = 0.0
    ev_list = []
    odds_list = []
    
    market_stats = {}

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
            
        res = predict_match(
            home_team, away_team, comp, odds_1, odds_X, odds_2,
            open_odds_win=odds_1, open_odds_draw=odds_X, open_odds_loss=odds_2,
            bankroll=bankroll, df=df, models_dict=models_dict
        )
        
        if res is None or 'best_bet' not in res:
            continue
            
        best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']
        
        # Filtro de Staking: Apostar solo si supera el umbral EV de Liga MX y stake >= $1.00
        if best_ev > res['league_ev_thresh'] and best_stake >= 1.0:
            total_bets += 1
            total_staked += best_stake
            ev_list.append(best_ev)
            odds_list.append(best_raw_odd)
            
            # Outcome: 2 = Home Win (Local), 1 = Draw (Empate), 0 / -1 = Away Win (Visitante)
            actual_outcome = row.get('outcome')
            if actual_outcome == -1:
                actual_outcome = 0
            
            goals_home = row.get('goals_scored', 0)
            goals_away = row.get('goals_conceded', 0)
            total_goals = goals_home + goals_away
            
            is_win = False
            mkt_category = best_bet_name
            
            if best_bet_name.startswith("1 ") or "Local" in best_bet_name:
                mkt_category = "1 (Local)"
                if actual_outcome == 2: is_win = True
            elif best_bet_name.startswith("X ") or "Empate" in best_bet_name:
                mkt_category = "X (Empate)"
                if actual_outcome == 1: is_win = True
            elif best_bet_name.startswith("2 ") or "Visitante" in best_bet_name:
                mkt_category = "2 (Visitante)"
                if actual_outcome == 0: is_win = True
            elif "Doble Oportunidad 1X" in best_bet_name:
                mkt_category = "Doble Oportunidad 1X"
                if actual_outcome in [1, 2]: is_win = True
            elif "Doble Oportunidad X2" in best_bet_name:
                mkt_category = "Doble Oportunidad X2"
                if actual_outcome in [0, 1]: is_win = True
            elif "Over 2.5" in best_bet_name:
                mkt_category = "Over 2.5 Goles"
                if total_goals > 2.5: is_win = True
            elif "BTTS" in best_bet_name:
                mkt_category = "BTTS (Ambos Anotan)"
                if goals_home > 0 and goals_away > 0: is_win = True

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

            if mkt_category not in market_stats:
                market_stats[mkt_category] = {'bets': 0, 'wins': 0, 'staked': 0.0, 'profit': 0.0}
            
            market_stats[mkt_category]['bets'] += 1
            market_stats[mkt_category]['wins'] += 1 if is_win else 0
            market_stats[mkt_category]['staked'] += best_stake
            market_stats[mkt_category]['profit'] += profit

    winrate = (wins / total_bets * 100.0) if total_bets > 0 else 0.0
    roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    avg_ev = np.mean(ev_list) * 100.0 if ev_list else 0.0
    avg_odds = np.mean(odds_list) if odds_list else 0.0
    growth_pct = ((bankroll - initial_bankroll) / initial_bankroll) * 100.0

    console.print("\n[bold bright_green]=== RENDIMIENTO FINAL DEL CLI PREDICTOR EN LIGA MX ===[/bold bright_green]")
    console.print(f"Billetera Inicial: [bold white]${initial_bankroll:,.2f} USD[/bold white]")
    console.print(f"Billetera Final: [bold bright_yellow]${bankroll:,.2f} USD[/bold bright_yellow] ([bold bright_green]{growth_pct:+.2f}% Crecimiento Total[/bold bright_green])")
    console.print(f"Partidos Evaluados: [bold]{sample_size}[/bold]")
    console.print(f"Apuestas Recomendadas Ejecutadas: [bold]{total_bets}[/bold]")
    console.print(f"Apuestas Ganadas: [bold bright_green]{wins}[/bold bright_green] | Apuestas Perdidas: [bold bright_red]{losses}[/bold bright_red]")
    console.print(f"Winrate Global: [bold bright_cyan]{winrate:.2f}%[/bold bright_cyan]")
    console.print(f"Capital Total Apostado: [bold]${total_staked:,.2f} USD[/bold]")
    
    if total_profit >= 0:
        console.print(f"Beneficio Neto Generado (Profit): [bold bright_green]+${total_profit:,.2f} USD[/bold bright_green]")
        console.print(f"Retorno sobre Inversión (ROI): [bold bright_green]+{roi:.2f}%[/bold bright_green]")
    else:
        console.print(f"Beneficio Neto Generado (Profit): [bold bright_red]-${abs(total_profit):,.2f} USD[/bold bright_red]")
        console.print(f"Retorno sobre Inversión (ROI): [bold bright_red]{roi:.2f}%[/bold bright_red]")
        
    console.print(f"EV Promedio de Apuestas: [bold]+{avg_ev:.2f}%[/bold]")
    console.print(f"Cuota Promedio Apostada: [bold]{avg_odds:.2f}[/bold]\n")

    if market_stats:
        t = Table(title="Desglose por Mercado Recomendado en Liga MX", show_header=True, header_style="bold white on blue", border_style="cyan")
        t.add_column("Mercado Recomendado", style="bold white")
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
    run_liga_mx_cli_test()
