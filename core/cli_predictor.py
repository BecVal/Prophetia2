import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from scipy.stats import poisson

console = Console()

MODEL_PATH = 'save_models/prophetia_xgb_model.pkl'
DATASET_PATH = '../data/processed/matches_with_odds.parquet'
FALLBACK_DATASET = '../data/processed/matches_dataset.parquet'

def calc_dixon_coles_draw(lam_scored, lam_conceded, rho=-0.15):
    prob = 0
    for i in range(6):
        p_scored = poisson.pmf(i, lam_scored)
        p_conceded = poisson.pmf(i, lam_conceded)
        base_prob = p_scored * p_conceded
        
        tau = 1.0
        if i == 0:
            tau = 1 - (lam_scored * lam_conceded * rho)
        elif i == 1:
            tau = 1 - rho
            
        tau = max(0, tau)
        prob += base_prob * tau
    return prob

def load_data():
    path = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    if not os.path.exists(path):
        console.print(f"[red]Error: Dataset no encontrado en {path}[/red]")
        return None
    return pd.read_parquet(path)

def get_latest_team_stats(df, team_name):
    # Buscar partidos donde el equipo sea 'team' (puede ser local o visitante original, 
    # pero nuestra DB duplica o asume perspectiva de 'team')
    team_df = df[(df['team'] == team_name)].sort_values('match_date')
    if team_df.empty:
        # Intentar buscar como opponent si no está como team
        opp_df = df[(df['opponent'] == team_name)].sort_values('match_date')
        if opp_df.empty:
            return None
        return opp_df.iloc[-1]
    return team_df.iloc[-1]

def calculate_ev(prob, odds):
    return (prob * odds) - 1

def main():
    console.print(Panel.fit("[bold cyan]Prophetia2 - Quant Value Betting CLI[/bold cyan]\n[dim]Initializing stochastic models and feature stores...[/dim]"))
    
    # Cargar Modelo
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]Error: Modelo no encontrado en {MODEL_PATH}[/red]")
        return
    
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
    
    # Cargar Data
    df = load_data()
    if df is None:
        return
        
    # Extraer lista de competiciones y equipos
    competitions = df['competition'].dropna().unique().tolist()
    
    comp = questionary.select("Selecciona la Liga:", choices=sorted(competitions)).ask()
    
    teams_in_comp = df[df['competition'] == comp]['team'].dropna().unique().tolist()
    
    home_team = questionary.select("Equipo Local:", choices=sorted(teams_in_comp)).ask()
    away_team = questionary.select("Equipo Visitante:", choices=sorted(teams_in_comp)).ask()
    
    if not home_team or not away_team:
        console.print("[red]Debes seleccionar ambos equipos.[/red]")
        return
        
    # Inputs manuales del mercado
    try:
        odds_1 = float(questionary.text("Cuota (Odds) Local [1]:").ask())
        odds_X = float(questionary.text("Cuota (Odds) Empate [X]:").ask())
        odds_2 = float(questionary.text("Cuota (Odds) Visitante [2]:").ask())
    except ValueError:
        console.print("[red]Cuotas inválidas.[/red]")
        return
        
    injuries_home = int(questionary.text("Lesiones clave Local [0-5]:", default="0").ask())
    injuries_away = int(questionary.text("Lesiones clave Visitante [0-5]:", default="0").ask())
    bankroll = float(questionary.text("Bankroll actual ($):", default="1000").ask())
    kelly_fraction = float(questionary.text("Kelly Fraction (ej. 0.20 para 20%):", default="0.20").ask())
    
    # Obtener estado actual
    home_stats = get_latest_team_stats(df, home_team)
    away_stats = get_latest_team_stats(df, away_team)
    
    if home_stats is None or away_stats is None:
        console.print(f"DEBUG: home_team='{home_team}', away_team='{away_team}'")
        console.print(f"DEBUG: home_stats is None: {home_stats is None}, away_stats is None: {away_stats is None}")
        console.print("[red]No se encontró información histórica suficiente para uno de los equipos.[/red]")
        return
        
    # Construir el Feature Vector
    input_data = {}
    
    for f in features:
        input_data[f] = 0.0 # Default fallback
        
    # Llenar datos base (del local)
    for f in features:
        if f in home_stats.index:
            input_data[f] = home_stats[f]
            
    # Llenar datos cruzados
    input_data['is_home'] = 1
    input_data['team_elo'] = home_stats['team_elo'] if 'team_elo' in home_stats else 1500
    input_data['opp_elo'] = away_stats['team_elo'] if 'team_elo' in away_stats else 1500
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']
    
    input_data['team_att_rating'] = home_stats['team_att_rating'] if 'team_att_rating' in home_stats else 1.0
    input_data['team_def_rating'] = home_stats['team_def_rating'] if 'team_def_rating' in home_stats else 1.0
    input_data['opp_att_rating'] = away_stats['team_att_rating'] if 'team_att_rating' in away_stats else 1.0
    input_data['opp_def_rating'] = away_stats['team_def_rating'] if 'team_def_rating' in away_stats else 1.0
    
    input_data['team_squad_value'] = home_stats['team_squad_value'] if 'team_squad_value' in home_stats else 0
    input_data['opp_squad_value'] = away_stats['team_squad_value'] if 'team_squad_value' in away_stats else 0
    input_data['squad_value_diff'] = input_data['team_squad_value'] - input_data['opp_squad_value']
    
    # Calcular Poisson aproximado (ya que el XGBPoisson no se guardó)
    # Aproximación: xG_created del local multiplicado por xG_conceded del visitante (escalado)
    lam_scored = home_stats.get('xg_created_ema5', 1.0) * (away_stats.get('xg_conceded_ema5', 1.0) / 1.5)
    lam_conceded = away_stats.get('xg_created_ema5', 1.0) * (home_stats.get('xg_conceded_ema5', 1.0) / 1.5)
    
    # Penalizaciones por lesiones (Ajuste Estocástico Manual)
    # 2% menos de xG a favor y ELO por lesión, 2% más de xG en contra por lesión clave.
    if injuries_home > 0:
        penalty = injuries_home * 0.02
        lam_scored *= (1 - penalty)
        lam_conceded *= (1 + penalty)
        input_data['team_elo'] *= (1 - penalty)
        
    if injuries_away > 0:
        penalty = injuries_away * 0.02
        lam_conceded *= (1 - penalty)
        lam_scored *= (1 + penalty)
        input_data['opp_elo'] *= (1 - penalty)
        
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']
    input_data['predicted_xg_scored'] = lam_scored
    input_data['predicted_xg_conceded'] = lam_conceded
    input_data['poisson_draw_prob'] = calc_dixon_coles_draw(lam_scored, lam_conceded)
    
    # Mercado implícito
    impl_win = 1/odds_1
    impl_draw = 1/odds_X
    impl_loss = 1/odds_2
    
    input_data['open_prob_win'] = impl_win / (impl_win + impl_draw + impl_loss)
    input_data['open_prob_draw'] = impl_draw / (impl_win + impl_draw + impl_loss)
    input_data['open_prob_loss'] = impl_loss / (impl_win + impl_draw + impl_loss)

    # Inferencia
    df_input = pd.DataFrame([input_data])[features]
    
    # Predecir Probabilidades
    y_prob = model.predict_proba(df_input)[0]
    y_prob = y_prob / np.sum(y_prob) # Normalizar
    
    prob_loss, prob_draw, prob_win = y_prob
    
    # Quant Blending Parámetros
    TAX_RETENTION_RATE = 0.07
    MARKET_BLEND_ALPHA = 0.85
    EXPECTED_CLV_DROP = 0.015
    
    # Probabilidades de Mercado
    margin = impl_win + impl_draw + impl_loss
    market_prob_win = impl_win / margin
    market_prob_draw = impl_draw / margin
    market_prob_loss = impl_loss / margin
    
    # Blended Probs
    blend_win = (MARKET_BLEND_ALPHA * prob_win) + ((1 - MARKET_BLEND_ALPHA) * market_prob_win)
    blend_draw = (MARKET_BLEND_ALPHA * prob_draw) + ((1 - MARKET_BLEND_ALPHA) * market_prob_draw)
    blend_loss = (MARKET_BLEND_ALPHA * prob_loss) + ((1 - MARKET_BLEND_ALPHA) * market_prob_loss)
    
    # Net Odds
    net_odds_1 = 1 + (odds_1 - 1) * (1 - TAX_RETENTION_RATE)
    net_odds_X = 1 + (odds_X - 1) * (1 - TAX_RETENTION_RATE)
    net_odds_2 = 1 + (odds_2 - 1) * (1 - TAX_RETENTION_RATE)
    
    # Net EV
    ev_win = (blend_win * net_odds_1) - 1 - EXPECTED_CLV_DROP
    ev_draw = (blend_draw * net_odds_X) - 1 - EXPECTED_CLV_DROP
    ev_loss = (blend_loss * net_odds_2) - 1 - EXPECTED_CLV_DROP
    
    # Dutching 1X (Doble Oportunidad Local o Empate)
    total_implied_1X = impl_win + impl_draw
    combined_odds_1X = 1 / total_implied_1X
    net_combined_odds_1X = 1 + (combined_odds_1X - 1) * (1 - TAX_RETENTION_RATE)
    blend_1X = blend_win + blend_draw
    ev_1X = (blend_1X * net_combined_odds_1X) - 1 - EXPECTED_CLV_DROP
    
    # Kelly Criterion
    def calc_kelly_stake(ev, net_odd):
        b = net_odd - 1
        return (ev / b) if b > 0 and ev > 0 else 0
        
    k_win = calc_kelly_stake(ev_win, net_odds_1)
    k_draw = calc_kelly_stake(ev_draw, net_odds_X)
    k_loss = calc_kelly_stake(ev_loss, net_odds_2)
    k_1X = calc_kelly_stake(ev_1X, net_combined_odds_1X)
    
    # Mostrar resultados en tabla
    console.print("\n[bold]=== ANÁLISIS CUANTITATIVO DEL PARTIDO ===[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Mercado", style="cyan")
    table.add_column("Odds (Brutas)")
    table.add_column("Prob. Bookie", justify="right")
    table.add_column("Prob. Blended", justify="right")
    table.add_column("Net EV", justify="right")
    table.add_column("Kelly %", justify="right")
    
    def color_ev(ev):
        return f"[green]+{ev*100:.1f}%[/green]" if ev > 0 else f"[red]{ev*100:.1f}%[/red]"
        
    def color_prob(p_mod, p_book):
        return f"[green]{p_mod*100:.1f}%[/green]" if p_mod > p_book else f"[dim]{p_mod*100:.1f}%[/dim]"

    table.add_row(
        f"1 (Local - {home_team})", f"{odds_1:.2f}", f"{(market_prob_win)*100:.1f}%", color_prob(blend_win, market_prob_win), color_ev(ev_win), f"{k_win*100:.2f}%"
    )
    table.add_row(
        "X (Empate)", f"{odds_X:.2f}", f"{(market_prob_draw)*100:.1f}%", color_prob(blend_draw, market_prob_draw), color_ev(ev_draw), f"{k_draw*100:.2f}%"
    )
    table.add_row(
        f"2 (Visita - {away_team})", f"{odds_2:.2f}", f"{(market_prob_loss)*100:.1f}%", color_prob(blend_loss, market_prob_loss), color_ev(ev_loss), f"{k_loss*100:.2f}%"
    )
    table.add_row(
        "1X (Local o Empate)", f"{combined_odds_1X:.2f}", f"{(market_prob_win + market_prob_draw)*100:.1f}%", color_prob(blend_1X, market_prob_win + market_prob_draw), color_ev(ev_1X), f"{k_1X*100:.2f}%"
    )
    
    console.print(table)
    
    # Recomendación Final
    console.print("\n[bold]=== RECOMENDACIÓN DE STAKING ===[/bold]")
    
    best_ev = max(ev_win, ev_draw, ev_loss, ev_1X)
    if best_ev > 0:
        if best_ev == ev_win:
            selection = "Local (1)"
            k_pct = k_win * kelly_fraction
            odds = odds_1
        elif best_ev == ev_draw:
            selection = "Empate (X)"
            k_pct = k_draw * kelly_fraction
            odds = odds_X
        elif best_ev == ev_1X:
            selection = "Doble Oportunidad (1X) / Dutching"
            k_pct = k_1X * kelly_fraction
            odds = combined_odds_1X
        else:
            selection = "Visitante (2)"
            k_pct = k_loss * kelly_fraction
            odds = odds_2
            
        # Hard cap at 5% of bankroll per bet
        if k_pct > 0.05:
            k_pct = 0.05
            
        stake_amount = bankroll * k_pct
        
        if k_pct < 0.005:
            console.print(f"[yellow]El Edge existe pero es muy marginal para el riesgo (Kelly < 0.5%).[/yellow] -> [bold]PASS / NO BET[/bold]")
        else:
            rec = Panel(
                f"[bold green]VALUE DETECTADO[/bold green]\n"
                f"Selección: [bold]{selection}[/bold] @ {odds:.2f} (Cuota Bruta Mínima)\n"
                f"Stake Recomendado: [bold]${stake_amount:.2f}[/bold] ({k_pct*100:.2f}% del bankroll)",
                title="SISTEMA DE STAKING", border_style="green"
            )
            console.print(rec)
    else:
        console.print(Panel("[bold red]NO HAY VALUE EN ESTE PARTIDO.[/bold red]\nEl mercado es más eficiente que nuestra proyección tras descontar impuestos y CLV slippage.", title="SISTEMA DE STAKING", border_style="red"))

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
