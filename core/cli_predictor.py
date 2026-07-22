import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from scipy.stats import entropy

# Add core path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.train_quant_advanced import predict_from_map
from models.train_gbm_model import compute_gbm_features
from models.train_nn import SklearnPyTorchWrapper, PyTorchMLP
import __main__
__main__.SklearnPyTorchWrapper = SklearnPyTorchWrapper
__main__.PyTorchMLP = PyTorchMLP

console = Console()

MODEL_DIR = 'save_models'
QUANT_PATH = os.path.join(MODEL_DIR, 'quant_advanced_model.pkl')
CONTEXT_PATH = os.path.join(MODEL_DIR, 'context_model.pkl')
NN_PATH = os.path.join(MODEL_DIR, 'nn_model.pkl')
DRAWS_PATH = os.path.join(MODEL_DIR, 'draws_model.pkl')
MARKET_PATH = os.path.join(MODEL_DIR, 'market_model.pkl')
GBM_PATH = os.path.join(MODEL_DIR, 'gbm_model.pkl')
FUNDAMENTAL_PATH = os.path.join(MODEL_DIR, 'stacker_fundamental_model.pkl')
FINAL_PATH = os.path.join(MODEL_DIR, 'stacker_final_model.pkl')

DATASET_PATH = '../data/processed/matches_with_odds.parquet'
FALLBACK_DATASET = '../data/processed/matches_dataset.parquet'
OPTIMIZED_PARAMS_FILE = '../data/processed/models_best_parameters/optimal_bankroll_params.json'

def load_data():
    path = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    if not os.path.exists(path):
        console.print(f"[red]Error: Dataset no encontrado en {path}[/red]")
        return None
    return pd.read_parquet(path)

def get_latest_team_stats(df, team_name):
    team_df = df[(df['team'] == team_name)].sort_values('match_date')
    if team_df.empty:
        opp_df = df[(df['opponent'] == team_name)].sort_values('match_date')
        if opp_df.empty:
            return None
        return opp_df.iloc[-1]
    return team_df.iloc[-1]

def calculate_ev(prob, odds):
    return (prob * odds) - 1

def compute_meta_features_live(df_base, open_odds_loss, open_odds_draw, open_odds_win, comp_id=0):
    meta = pd.DataFrame(index=df_base.index)
    cols_loss = [c for c in df_base.columns if 'loss' in c.lower()]
    cols_draw = [c for c in df_base.columns if 'draw' in c.lower()]
    cols_win = [c for c in df_base.columns if 'win' in c.lower()]
    
    meta['meta_std_loss'] = df_base[cols_loss].std(axis=1).fillna(0) if cols_loss else 0
    meta['meta_std_draw'] = df_base[cols_draw].std(axis=1).fillna(0) if cols_draw else 0
    meta['meta_std_win'] = df_base[cols_win].std(axis=1).fillna(0) if cols_win else 0
    
    mean_loss = df_base[cols_loss].mean(axis=1) if cols_loss else 0
    mean_draw = df_base[cols_draw].mean(axis=1) if cols_draw else 0
    mean_win = df_base[cols_win].mean(axis=1) if cols_win else 1
    
    sums = mean_loss + mean_draw + mean_win
    mean_loss = mean_loss / np.where(sums > 0, sums, 1)
    mean_draw = mean_draw / np.where(sums > 0, sums, 1)
    mean_win = mean_win / np.where(sums > 0, sums, 1)
    
    def calc_entropy(row):
        return entropy([row['loss'] + 1e-9, row['draw'] + 1e-9, row['win'] + 1e-9])
        
    mean_probs = pd.DataFrame({'loss': mean_loss, 'draw': mean_draw, 'win': mean_win})
    meta['meta_entropy'] = mean_probs.apply(calc_entropy, axis=1)
    
    meta['implied_open_loss'] = 1 / max(open_odds_loss, 1.01)
    meta['implied_open_draw'] = 1 / max(open_odds_draw, 1.01)
    meta['implied_open_win'] = 1 / max(open_odds_win, 1.01)
    meta['competition_id'] = comp_id
    
    return pd.concat([df_base, meta], axis=1)

def main():
    console.print(Panel.fit("[bold cyan]Prophetia2 - Quant Value Betting CLI[/bold cyan]\n[dim]Initializing Quant Models & Stacker...[/dim]"))
    
    models = [QUANT_PATH, CONTEXT_PATH, NN_PATH, DRAWS_PATH, MARKET_PATH, GBM_PATH, FUNDAMENTAL_PATH, FINAL_PATH]
    if not all(os.path.exists(p) for p in models):
        console.print(f"[red]Error: Faltan modelos entrenados. Asegúrate de correr los scripts de /models/.[/red]")
        return
        
    quant_data = joblib.load(QUANT_PATH)
    context_data = joblib.load(CONTEXT_PATH)
    nn_data = joblib.load(NN_PATH)
    draws_data = joblib.load(DRAWS_PATH)
    market_data = joblib.load(MARKET_PATH)
    gbm_data = joblib.load(GBM_PATH)
    fund_data = joblib.load(FUNDAMENTAL_PATH)
    final_data = joblib.load(FINAL_PATH)
    
    df = load_data()
    if df is None:
        return
        
    competitions = df['competition'].dropna().unique().tolist()
    comp = questionary.select("Selecciona la Liga:", choices=sorted(competitions)).ask()
    
    teams_in_comp = df[df['competition'] == comp]['team'].dropna().unique().tolist()
    home_team = questionary.select("Equipo Local:", choices=sorted(teams_in_comp)).ask()
    away_team = questionary.select("Equipo Visitante:", choices=sorted(teams_in_comp)).ask()
    
    if not home_team or not away_team:
        console.print("[red]Debes seleccionar ambos equipos.[/red]")
        return
        
    try:
        odds_1 = float(questionary.text("Cuota de Apertura Local [1]:").ask())
        odds_X = float(questionary.text("Cuota de Apertura Empate [X]:").ask())
        odds_2 = float(questionary.text("Cuota de Apertura Visitante [2]:").ask())
    except ValueError:
        console.print("[red]Cuotas inválidas.[/red]")
        return
        
    injuries_home = int(questionary.text("Lesiones clave Local [0-5]:", default="0").ask())
    injuries_away = int(questionary.text("Lesiones clave Visitante [0-5]:", default="0").ask())
    bankroll = float(questionary.text("Bankroll actual ($):", default="1000").ask())
    
    kelly_fractions = {}
    ev_thresholds = {}
    alpha_div_low_dict = {}
    alpha_div_med_dict = {}
    alpha_div_high_dict = {}
    if os.path.exists(OPTIMIZED_PARAMS_FILE):
        with open(OPTIMIZED_PARAMS_FILE, 'r') as f:
            data = json.load(f)
            kelly_fractions = data.get('KELLY_FRACTIONS', {})
            ev_thresholds = data.get('EV_THRESHOLDS', {})
            alpha_div_low_dict = data.get('ALPHA_DIV_LOW', {})
            alpha_div_med_dict = data.get('ALPHA_DIV_MED', {})
            alpha_div_high_dict = data.get('ALPHA_DIV_HIGH', {})
            
    league_kelly = kelly_fractions.get(comp, kelly_fractions.get('DEFAULT', 0.015))
    league_ev_thresh = ev_thresholds.get(comp, ev_thresholds.get('DEFAULT', 0.015))
    league_alpha_low = alpha_div_low_dict.get(comp, alpha_div_low_dict.get('DEFAULT', 0.85))
    league_alpha_med = alpha_div_med_dict.get(comp, alpha_div_med_dict.get('DEFAULT', 0.70))
    league_alpha_high = alpha_div_high_dict.get(comp, alpha_div_high_dict.get('DEFAULT', 0.50))
    
    console.print(f"[dim]Parámetros cargados para la liga {comp}: EV Threshold = {league_ev_thresh:.4f}, Kelly = {league_kelly:.4f}, Alphas = [{league_alpha_low:.2f}, {league_alpha_med:.2f}, {league_alpha_high:.2f}][/dim]")
    
    home_stats = get_latest_team_stats(df, home_team)
    away_stats = get_latest_team_stats(df, away_team)
    
    if home_stats is None or away_stats is None:
        console.print("[red]No se encontró información histórica suficiente para uno de los equipos.[/red]")
        return
        
    console.print(f"\n[bold green]✓[/bold green] Histórico y Estadísticas H2H cargadas para: {home_team} vs {away_team}")
    
    input_data = {}
    all_features = set(context_data['features'] + nn_data['features'] + draws_data['features'])
    
    for f in all_features:
        input_data[f] = 0.0
        if f in home_stats.index:
            input_data[f] = home_stats[f]
            
    input_data['is_home'] = 1
    input_data['team'] = home_team
    input_data['opponent'] = away_team
    input_data['competition'] = comp
    input_data['competition_id'] = pd.factorize(df['competition'])[1].get_loc(comp) if comp in pd.factorize(df['competition'])[1] else 0
    
    input_data['team_elo'] = home_stats['team_elo'] if 'team_elo' in home_stats else 1500
    input_data['opp_elo'] = away_stats['team_elo'] if 'team_elo' in away_stats else 1500
    if injuries_home > 0: input_data['team_elo'] *= (1 - (injuries_home * 0.02))
    if injuries_away > 0: input_data['opp_elo'] *= (1 - (injuries_away * 0.02))
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']
    
    input_data['team_squad_value'] = home_stats['team_squad_value'] if 'team_squad_value' in home_stats else 0
    input_data['opp_squad_value'] = away_stats['team_squad_value'] if 'team_squad_value' in away_stats else 0
    input_data['squad_value_diff'] = input_data['team_squad_value'] - input_data['opp_squad_value']
    
    # Mercado (Al no tener cuotas de cierre, sumimos que Open = Close -> Steam=0)
    impl_win = 1/odds_1
    impl_draw = 1/odds_X
    impl_loss = 1/odds_2
    margin = impl_win + impl_draw + impl_loss
    
    input_data['open_odds_win'] = odds_1
    input_data['open_odds_draw'] = odds_X
    input_data['open_odds_loss'] = odds_2
    input_data['odds_win'] = odds_1
    input_data['odds_draw'] = odds_X
    input_data['odds_loss'] = odds_2
    
    input_data['open_prob_win'] = impl_win / margin
    input_data['open_prob_draw'] = impl_draw / margin
    input_data['open_prob_loss'] = impl_loss / margin
    
    input_data['prob_win_implied'] = input_data['open_prob_win']
    input_data['prob_draw_implied'] = input_data['open_prob_draw']
    input_data['prob_loss_implied'] = input_data['open_prob_loss']
    
    input_data['vig_open'] = margin - 1
    input_data['vig_close'] = margin - 1
    input_data['steam_win'] = 0.0
    input_data['steam_draw'] = 0.0
    input_data['steam_loss'] = 0.0
    
    # Historico para GBM (Volatilidad de Drift)
    team_hist = df[df['team'] == home_team].copy()
    current_match_df = pd.DataFrame([input_data])
    current_match_df['match_date'] = pd.Timestamp.now()
    combined_hist = pd.concat([team_hist, current_match_df], ignore_index=True)
    
    combined_gbm = compute_gbm_features(combined_hist)
    if combined_gbm is not None:
        last_gbm = combined_gbm.iloc[-1:]
        for col in gbm_data['features']:
            if col in last_gbm:
                input_data[col] = last_gbm[col].values[0]
            elif col.startswith('gbm_base_prob_'):
                clean_col = col.replace('gbm_base_', '')
                input_data[col] = current_match_df[clean_col].values[0] if clean_col in current_match_df else 0.33
    else:
        for col in gbm_data['features']: input_data[col] = 0.0
        
    df_input_full = pd.DataFrame([input_data])
    
    # 1. Quant Advanced
    # Ajuste manual por lesiones antes de Quant
    if injuries_home > 0: df_input_full['team_elo'] *= (1 - (injuries_home * 0.02))
    if injuries_away > 0: df_input_full['opp_elo'] *= (1 - (injuries_away * 0.02))
    
    # Predict from map requires specific columns, which df_input_full has.
    q_win, q_draw, q_loss, xg_scored, xg_conceded = predict_from_map(quant_data['map_estimate'], quant_data['team_mapping'], df_input_full)
    xg_s = xg_scored[0]
    xg_c = xg_conceded[0]
    
    # Ajuste simple de lesiones directo sobre xG (fuera del modelo matemático estricto)
    if injuries_home > 0:
        xg_s *= (1 - (injuries_home * 0.02))
        xg_c *= (1 + (injuries_home * 0.02))
    if injuries_away > 0:
        xg_c *= (1 - (injuries_away * 0.02))
        xg_s *= (1 + (injuries_away * 0.02))
        
    console.print(f"\n[cyan]>> Fase 1: Inferencia de Modelos Base[/cyan]")
    console.print(f"  [dim]Quant ZINB (PyMC): xG Home = {xg_s:.3f}, xG Away = {xg_c:.3f} | Probs [L:{q_loss[0]:.2f}, D:{q_draw[0]:.2f}, W:{q_win[0]:.2f}][/dim]")
    
    # 2. Context
    df_ctx = df_input_full[context_data['features']]
    if 'competition_id' in df_ctx.columns:
        df_ctx['competition_id'] = df_ctx['competition_id'].astype('category')
    ctx_probs = context_data['model'].predict_proba(df_ctx)[0]
    console.print(f"  [dim]Contextual (XGB):  Probs [L:{ctx_probs[0]:.2f}, D:{ctx_probs[1]:.2f}, W:{ctx_probs[2]:.2f}][/dim]")
    
    # 3. NN
    nn_probs = nn_data['model'].predict_proba(df_input_full[nn_data['features']])[0]
    console.print(f"  [dim]Red Neuronal (PT):  Probs [L:{nn_probs[0]:.2f}, D:{nn_probs[1]:.2f}, W:{nn_probs[2]:.2f}][/dim]")
    
    # 4. Draws
    draws_prob = draws_data['model'].predict_proba(df_input_full[draws_data['features']])[0][1]
    console.print(f"  [dim]Caza-Empates (XGB):Prob Empate {draws_prob:.3f}[/dim]")
    
    # 5. Market
    mkt_probs = market_data['model'].predict_proba(df_input_full[market_data['features']])[0]
    console.print(f"  [dim]Dinámica Mercado:  Probs [L:{mkt_probs[0]:.2f}, D:{mkt_probs[1]:.2f}, W:{mkt_probs[2]:.2f}][/dim]")
    
    # 6. GBM
    gbm_probs = gbm_data['model'].predict_proba(df_input_full[gbm_data['features']])[0]
    console.print(f"  [dim]Deriva GBM:        Probs [L:{gbm_probs[0]:.2f}, D:{gbm_probs[1]:.2f}, W:{gbm_probs[2]:.2f}][/dim]")
    
    # 7. Stacker Fundamental
    df_fund_input = pd.DataFrame({
        'predicted_xg_scored_quant': xg_scored,
        'predicted_xg_conceded_quant': xg_conceded,
        'quant_win_prob': q_win,
        'quant_draw_prob': q_draw,
        'quant_loss_prob': q_loss,
        'prob_loss_ctx': [ctx_probs[0]],
        'prob_draw_ctx': [ctx_probs[1]],
        'prob_win_ctx': [ctx_probs[2]],
        'prob_loss_nn': [nn_probs[0]],
        'prob_draw_nn': [nn_probs[1]],
        'prob_win_nn': [nn_probs[2]],
        'prob_is_draw': [draws_prob]
    })
    
    # Asegurar orden exacto de features
    df_fund_input = df_fund_input[fund_data['features']]
    fund_probs = fund_data['model'].predict_proba(df_fund_input)[0]
    
    console.print(f"\n[cyan]>> Fase 2: Stacking Fundamental (Nivel 1)[/cyan]")
    console.print(f"  [dim]Consenso:          Probs [L:{fund_probs[0]:.2f}, D:{fund_probs[1]:.2f}, W:{fund_probs[2]:.2f}][/dim]")
    
    df_fund_out = pd.DataFrame({
        'fund_prob_loss': [fund_probs[0]],
        'fund_prob_draw': [fund_probs[1]],
        'fund_prob_win': [fund_probs[2]]
    })
    
    # 8. Stacker Final
    df_mkt = pd.DataFrame({
        'prob_loss_mkt': [mkt_probs[0]], 'prob_draw_mkt': [mkt_probs[1]], 'prob_win_mkt': [mkt_probs[2]],
        'prob_loss_gbm': [gbm_probs[0]], 'prob_draw_gbm': [gbm_probs[1]], 'prob_win_gbm': [gbm_probs[2]]
    })
    
    df_meta_input = pd.concat([df_fund_out, df_mkt], axis=1)
    df_final = compute_meta_features_live(df_meta_input, odds_2, odds_X, odds_1, input_data['competition_id'])
    
    console.print(f"\n[cyan]>> Fase 3: Ingeniería de Meta-Features (Medición de Incertidumbre)[/cyan]")
    console.print(f"  [dim]Varianza Modelos:  Loss={df_final['meta_std_loss'].values[0]:.4f}, Draw={df_final['meta_std_draw'].values[0]:.4f}, Win={df_final['meta_std_win'].values[0]:.4f}[/dim]")
    console.print(f"  [dim]Entropía Consenso: {df_final['meta_entropy'].values[0]:.4f} bits[/dim]")
    console.print(f"  [dim]Cuota Implícita:   L={df_final['implied_open_loss'].values[0]:.3f}, D={df_final['implied_open_draw'].values[0]:.3f}, W={df_final['implied_open_win'].values[0]:.3f}[/dim]")
    
    df_final = df_final[final_data['features']]
    if 'competition_id' in df_final.columns:
        df_final['competition_id'] = df_final['competition_id'].astype('category')
        
    final_probs = final_data['model'].predict_proba(df_final)[0]
    final_probs = final_probs / np.sum(final_probs)
    
    prob_loss, prob_draw, prob_win = final_probs
    console.print(f"\n[cyan]>> Fase 4: Stacking Final de Mercado (HGB)[/cyan]")
    console.print(f"  [dim]Proyección Final:  Probs [L:{prob_loss:.3f}, D:{prob_draw:.3f}, W:{prob_win:.3f}][/dim]")
    
    # Quant Blending Parámetros
    TAX_RETENTION_RATE = 0.0075
    EXPECTED_CLV_DROP = 0.015
    
    market_prob_win = impl_win / margin
    market_prob_draw = impl_draw / margin
    market_prob_loss = impl_loss / margin
    
    def get_dynamic_alpha(prob, market_prob):
        divergence = abs(prob - market_prob)
        if divergence > 0.20:
            return 0.30
        elif divergence > 0.15:
            return 0.50
        elif divergence > 0.10:
            return league_alpha_high
        elif divergence > 0.05:
            return league_alpha_med
        else:
            return league_alpha_low

    alpha_win = get_dynamic_alpha(prob_win, market_prob_win)
    alpha_draw = get_dynamic_alpha(prob_draw, market_prob_draw)
    alpha_loss = get_dynamic_alpha(prob_loss, market_prob_loss)
    
    blend_win = (alpha_win * prob_win) + ((1 - alpha_win) * market_prob_win)
    blend_draw = (alpha_draw * prob_draw) + ((1 - alpha_draw) * market_prob_draw)
    blend_loss = (alpha_loss * prob_loss) + ((1 - alpha_loss) * market_prob_loss)
    
    net_odds_1 = 1 + (odds_1 - 1) * (1 - TAX_RETENTION_RATE)
    net_odds_X = 1 + (odds_X - 1) * (1 - TAX_RETENTION_RATE)
    net_odds_2 = 1 + (odds_2 - 1) * (1 - TAX_RETENTION_RATE)
    
    ev_win = (blend_win * net_odds_1) - 1 - EXPECTED_CLV_DROP
    ev_draw = (blend_draw * net_odds_X) - 1 - EXPECTED_CLV_DROP
    ev_loss = (blend_loss * net_odds_2) - 1 - EXPECTED_CLV_DROP
    
    total_implied_1X = impl_win + impl_draw
    combined_odds_1X = 1 / total_implied_1X
    net_combined_odds_1X = 1 + (combined_odds_1X - 1) * (1 - TAX_RETENTION_RATE)
    blend_1X = blend_win + blend_draw
    ev_1X = (blend_1X * net_combined_odds_1X) - 1 - EXPECTED_CLV_DROP
    
    total_implied_X2 = impl_draw + impl_loss
    combined_odds_X2 = 1 / total_implied_X2
    net_combined_odds_X2 = 1 + (combined_odds_X2 - 1) * (1 - TAX_RETENTION_RATE)
    blend_X2 = blend_draw + blend_loss
    ev_X2 = (blend_X2 * net_combined_odds_X2) - 1 - EXPECTED_CLV_DROP
    
    def calc_kelly_stake(ev, net_odd, raw_odd):
        b = net_odd - 1
        kelly_ev = min(ev, 0.15)
        kelly_pct = (kelly_ev / b) if b > 0 and kelly_ev > 0 else 0
        if raw_odd < 1.30:
            kelly_pct = min(kelly_pct, 0.01)
        return kelly_pct
        
    k_win = calc_kelly_stake(ev_win, net_odds_1, odds_1)
    k_draw = calc_kelly_stake(ev_draw, net_odds_X, odds_X)
    k_loss = calc_kelly_stake(ev_loss, net_odds_2, odds_2)
    k_1X = calc_kelly_stake(ev_1X, net_combined_odds_1X, combined_odds_1X)
    k_X2 = calc_kelly_stake(ev_X2, net_combined_odds_X2, combined_odds_X2)
    
    console.print("\n[bold]=== ANÁLISIS CUANTITATIVO DEL PARTIDO ===[/bold]")
    console.print(f"[bold cyan]Marcador Esperado Quant (xG):[/bold cyan] {home_team} [bold yellow]{xg_s:.2f} - {xg_c:.2f}[/bold yellow] {away_team}\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Mercado", style="cyan")
    table.add_column("Odds")
    table.add_column("Prob. Bookie", justify="right")
    table.add_column("Prob. Blended", justify="right")
    table.add_column("Net EV", justify="right")
    table.add_column("Kelly %", justify="right")
    
    def color_ev(ev):
        return f"[green]+{ev*100:.1f}%[/green]" if ev > 0 else f"[red]{ev*100:.1f}%[/red]"
        
    def color_prob(p_mod, p_book):
        return f"[green]{p_mod*100:.1f}%[/green]" if p_mod > p_book else f"[dim]{p_mod*100:.1f}%[/dim]"

    table.add_row(f"1 (Local - {home_team})", f"{odds_1:.2f}", f"{(market_prob_win)*100:.1f}%", color_prob(blend_win, market_prob_win), color_ev(ev_win), f"{k_win*100:.2f}%")
    table.add_row("X (Empate)", f"{odds_X:.2f}", f"{(market_prob_draw)*100:.1f}%", color_prob(blend_draw, market_prob_draw), color_ev(ev_draw), f"{k_draw*100:.2f}%")
    table.add_row(f"2 (Visita - {away_team})", f"{odds_2:.2f}", f"{(market_prob_loss)*100:.1f}%", color_prob(blend_loss, market_prob_loss), color_ev(ev_loss), f"{k_loss*100:.2f}%")
    table.add_row("1X (Local o Empate)", f"{combined_odds_1X:.2f}", f"{(market_prob_win + market_prob_draw)*100:.1f}%", color_prob(blend_1X, market_prob_win + market_prob_draw), color_ev(ev_1X), f"{k_1X*100:.2f}%")
    table.add_row("X2 (Empate o Visita)", f"{combined_odds_X2:.2f}", f"{(market_prob_draw + market_prob_loss)*100:.1f}%", color_prob(blend_X2, market_prob_draw + market_prob_loss), color_ev(ev_X2), f"{k_X2*100:.2f}%")
    
    console.print(table)
    
    console.print("\n[bold]=== RECOMENDACIÓN DE STAKING ===[/bold]")
    best_ev = max(ev_win, ev_draw, ev_loss, ev_1X, ev_X2)
    
    if best_ev > league_ev_thresh:
        if best_ev == ev_win:
            selection, k_pct, odds = "Local (1)", k_win * league_kelly, odds_1
        elif best_ev == ev_draw:
            selection, k_pct, odds = "Empate (X)", k_draw * league_kelly, odds_X
        elif best_ev == ev_1X:
            selection, k_pct, odds = "Doble Oportunidad (1X)", k_1X * league_kelly, combined_odds_1X
        elif best_ev == ev_X2:
            selection, k_pct, odds = "Doble Oportunidad (X2)", k_X2 * league_kelly, combined_odds_X2
        else:
            selection, k_pct, odds = "Visitante (2)", k_loss * league_kelly, odds_2
            
        MAX_STAKE_PCT = 0.6
        if k_pct > MAX_STAKE_PCT: k_pct = MAX_STAKE_PCT
            
        stake_amount = bankroll * k_pct
        
        MAX_BET_LIQUIDITY = {
            'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
            'D2': 2000.0, 'F2': 2000.0,
            'T1': 2000.0,
            'DEFAULT': 2000.0
        }
        max_liquidity = MAX_BET_LIQUIDITY.get(comp, MAX_BET_LIQUIDITY.get('DEFAULT', 200.0))
        if stake_amount > max_liquidity:
            stake_amount = max_liquidity
        
        if k_pct < 0.001:
            console.print(f"[yellow]Edge > umbral pero Kelly es muy bajo (< 0.1%).[/yellow] -> [bold]PASS / NO BET[/bold]")
        else:
            rec = Panel(
                f"[bold green]VALUE DETECTADO[/bold green]\n"
                f"Selección: [bold]{selection}[/bold] @ {odds:.2f}\n"
                f"EV Proyectado: [bold]{best_ev*100:.2f}%[/bold] (Threshold: {league_ev_thresh*100:.2f}%)\n"
                f"Stake Recomendado: [bold]${stake_amount:.2f}[/bold] ({k_pct*100:.2f}% del bankroll)",
                title="SISTEMA DE STAKING", border_style="green"
            )
            console.print(rec)
    elif best_ev > 0:
        console.print(Panel(f"[yellow]EDGE INSUFICIENTE.[/yellow]\nEV máximo: {best_ev*100:.2f}%. Threshold liga: {league_ev_thresh*100:.2f}%. -> [bold]PASS[/bold]", title="SISTEMA", border_style="yellow"))
    else:
        console.print(Panel("[bold red]NO HAY VALUE EN ESTE PARTIDO.[/bold red]\nEl mercado es más eficiente que nuestra proyección.", title="SISTEMA", border_style="red"))

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Suprimir warnings de pandas SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    main()
