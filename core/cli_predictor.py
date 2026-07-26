import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import entropy, poisson
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add core path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models.train_quant_advanced import predict_from_model, KalmanDixonColesQuantModel
from market_models.train_gbm_model import compute_gbm_features
from models.train_nn import SklearnPyTorchWrapper10, PyTorchGatedResNet10, GatedContext, ResidualBlock
from models.train_draws import HybridDrawsEnsemble

import __main__
__main__.SklearnPyTorchWrapper10 = SklearnPyTorchWrapper10
__main__.PyTorchGatedResNet10 = PyTorchGatedResNet10
__main__.GatedContext = GatedContext
__main__.ResidualBlock = ResidualBlock
__main__.SklearnPyTorchWrapper = SklearnPyTorchWrapper10
__main__.HybridDrawsEnsemble = HybridDrawsEnsemble

console = Console()

MODEL_DIR = os.path.join(script_dir, 'save_models')
QUANT_PATH = os.path.join(MODEL_DIR, 'quant_advanced_model.pkl')
CONTEXT_PATH = os.path.join(MODEL_DIR, 'context_model.pkl')
NN_PATH = os.path.join(MODEL_DIR, 'nn_model.pkl')
DRAWS_PATH = os.path.join(MODEL_DIR, 'draws_model.pkl')
MARKET_PATH = os.path.join(MODEL_DIR, 'market_model.pkl')
GBM_PATH = os.path.join(MODEL_DIR, 'gbm_model.pkl')
FUNDAMENTAL_PATH = os.path.join(MODEL_DIR, 'stacker_fundamental_model.pkl')
FINAL_PATH = os.path.join(MODEL_DIR, 'stacker_final_model.pkl')
CORNERS_PATH = os.path.join(MODEL_DIR, 'corners_model.pkl')
CARDS_PATH = os.path.join(MODEL_DIR, 'cards_total_xgboost_model.pkl')

DATASET_PATH = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_with_odds.parquet'))
FALLBACK_DATASET = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_dataset.parquet'))
OPTIMIZED_PARAMS_FILE = os.path.abspath(os.path.join(script_dir, '../data/processed/models_best_parameters/optimal_bankroll_params.json'))

# --- CONFIGURACIÓN QUANT INSTITUCIONAL (10/10) ---
TAX_RETENTION_RATE = 0.0075        # Retención del 0.75% sobre ganancias netas (Polymarket)
EXPECTED_CLV_DROP = 0.015         # Penalización por slippage esperado del CLV (-1.5%)
MAX_STAKE_PCT = 0.03               # Stake máximo por apuesta (3.0% del bankroll)
MARKET_IMPACT_GAMMA = 0.05        # Coeficiente de impacto de mercado (Square-root Law)

MAX_BET_LIQUIDITY = {             # Límites de liquidez absolutos por competición (USD)
    'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
    'D2': 2000.0, 'F2': 2000.0, 'T1': 2000.0, 'MLS': 1500.0, 'J1': 1500.0,
    'DEFAULT': 2000.0
}

COMPETITION_MAPPING = {
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
    'MLS': 'Major League Soccer', 'Major League Soccer': 'MLS',
    'J1': 'J-League 1', 'J-League 1': 'J1',
    'CL': 'Champions League', 'Champions League': 'CL',
    'EL': 'Europa League', 'Europa League': 'EL',
    'WC': 'FIFA World Cup', 'FIFA World Cup': 'WC'
}

def get_param_by_comp(param_dict, comp, default_val=0.015):
    if comp in param_dict:
        return param_dict[comp]
    alt_name = COMPETITION_MAPPING.get(comp)
    if alt_name and alt_name in param_dict:
        return param_dict[alt_name]
    return param_dict.get('DEFAULT', default_val)

def calculate_dynamic_alpha(p_model, p_market, alpha_low, alpha_med, alpha_high):
    divergence = abs(p_model - p_market)
    if divergence < 0.05:
        return alpha_low
    elif divergence < 0.12:
        return alpha_med
    else:
        return alpha_high

def calculate_market_slippage(odds, stake, liquidity_cap):
    if stake <= 0 or liquidity_cap <= 0:
        return odds
    ratio = min(stake / liquidity_cap, 1.0)
    impact = MARKET_IMPACT_GAMMA * np.sqrt(ratio)
    effective_odds = 1.0 + (odds - 1.0) * (1.0 - impact)
    return max(effective_odds, 1.001)

def calculate_bayesian_uncertainty(probs_list):
    """Calcula la media, varianza, error estándar, intervalo de credibilidad 95% y factor de penalización."""
    arr = np.asarray(probs_list, dtype=np.float64)
    mean_p = float(np.mean(arr))
    var_p = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = float(np.sqrt(var_p / max(len(arr), 1)))
    ci_low = float(np.clip(mean_p - 1.96 * se, 0.0, 1.0))
    ci_high = float(np.clip(mean_p + 1.96 * se, 0.0, 1.0))
    uncertainty_penalty = float(np.exp(-4.0 * se))
    return mean_p, se, ci_low, ci_high, uncertainty_penalty

def solve_multi_asset_kelly(ev_vec, odds_vec, prob_vec, corr_matrix, bankroll, max_stake_pct, penalty=1.0, max_liquidity=2000.0, league_kelly=0.015):
    """
    Resuelve la asignación Kelly Multi-Activo con matriz de covarianza regularizada y penalizador Bayesiano.
    """
    n = len(ev_vec)
    ev_arr = np.maximum(np.asarray(ev_vec, dtype=np.float64), 0.0)
    prob_arr = np.asarray(prob_vec, dtype=np.float64)
    odds_arr = np.asarray(odds_vec, dtype=np.float64)
    
    std_vec = np.sqrt(np.clip(prob_arr * (1.0 - prob_arr), 1e-4, 1.0))
    cov_matrix = np.outer(std_vec, std_vec) * corr_matrix
    reg_cov = cov_matrix + np.eye(n) * 1e-3
    
    try:
        inv_cov = np.linalg.inv(reg_cov)
        b_vec = np.maximum(odds_arr - 1.0, 0.01)
        raw_weights = inv_cov @ (ev_arr / b_vec)
        raw_weights = np.maximum(raw_weights, 0.0)
    except np.linalg.LinAlgError:
        raw_weights = ev_arr / np.maximum(odds_arr - 1.0, 0.01)
        
    scaled_weights = raw_weights * penalty * league_kelly
    stakes = []
    pcts = []
    for i in range(n):
        pct = float(np.clip(scaled_weights[i], 0.0, max_stake_pct))
        if odds_arr[i] < 1.30: pct = min(pct, 0.01)
        stake = min(bankroll * pct, max_liquidity)
        stakes.append(stake)
        pcts.append(pct)
        
    return stakes, pcts

def compute_poisson_market_correlations(poisson_matrix):
    """Calcula la matriz de correlación empírica exacta entre los mercados de un partido."""
    rows, cols = poisson_matrix.shape
    weights = poisson_matrix.flatten()
    
    indicators = []
    # 0: Home Win (i > j)
    # 1: Draw (i == j)
    # 2: Away Win (i < j)
    # 3: 1X (i >= j)
    # 4: X2 (i <= j)
    # 5: Over 2.5 (i + j > 2)
    # 6: BTTS (i >= 1 and j >= 1)
    
    ind_win = np.fromfunction(lambda i, j: i > j, (rows, cols)).flatten().astype(float)
    ind_draw = np.fromfunction(lambda i, j: i == j, (rows, cols)).flatten().astype(float)
    ind_loss = np.fromfunction(lambda i, j: i < j, (rows, cols)).flatten().astype(float)
    ind_1X = np.fromfunction(lambda i, j: i >= j, (rows, cols)).flatten().astype(float)
    ind_X2 = np.fromfunction(lambda i, j: i <= j, (rows, cols)).flatten().astype(float)
    ind_ou25 = np.fromfunction(lambda i, j: (i + j) > 2, (rows, cols)).flatten().astype(float)
    ind_btts = np.fromfunction(lambda i, j: (i >= 1) & (j >= 1), (rows, cols)).flatten().astype(float)
    ind_corn = np.ones(rows * cols) * 0.5
    ind_card = np.ones(rows * cols) * 0.5

    matrix_ind = np.vstack([ind_win, ind_draw, ind_loss, ind_1X, ind_X2, ind_ou25, ind_btts, ind_corn, ind_card])
    
    # Covarianza ponderada
    mean_vec = matrix_ind @ weights
    dev = matrix_ind - mean_vec[:, None]
    weighted_cov = (dev * weights[None, :]) @ dev.T
    
    std_vec = np.sqrt(np.diag(weighted_cov))
    std_vec[std_vec == 0] = 1.0
    corr_matrix = weighted_cov / np.outer(std_vec, std_vec)
    np.fill_diagonal(corr_matrix, 1.0)
    return np.clip(corr_matrix, -1.0, 1.0)

def export_trade_signals(res, filepath='trade_signals.json'):
    """Exporta las proyecciones cuantitativas, CIs 95% y asignaciones Kelly a archivo JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'match': f"{res['home_team']} vs {res['away_team']}",
        'competition': res['competition'],
        'xg_expected': {'home': res['xg_scored'], 'away': res['xg_conceded']},
        'bayesian_ci_95': {
            'win': [res['ci_win_low'], res['ci_win_high']],
            'draw': [res['ci_draw_low'], res['ci_draw_high']],
            'loss': [res['ci_loss_low'], res['ci_loss_high']]
        },
        'recommended_bet': {
            'market': res['best_bet'][0],
            'net_ev': res['best_bet'][1],
            'stake_usd': res['best_bet'][2],
            'stake_pct': res['best_bet'][3],
            'entry_odd': res['best_bet'][4]
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    return filepath

def load_data():
    path = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    if not os.path.exists(path):
        console.print(f"[red]Error: Dataset no encontrado en {path}[/red]")
        return None
    return pd.read_parquet(path)

def load_all_models():
    """Carga y empaqueta todos los modelos entrenados guardados."""
    models_dict = {}
    required = [QUANT_PATH, CONTEXT_PATH, NN_PATH, DRAWS_PATH, MARKET_PATH, GBM_PATH, FUNDAMENTAL_PATH, FINAL_PATH]
    if not all(os.path.exists(p) for p in required):
        missing = [p for p in required if not os.path.exists(p)]
        console.print(f"[red]Error: Faltan modelos entrenados: {missing}[/red]")
        return None

    quant_raw = joblib.load(QUANT_PATH)
    quant_model = KalmanDixonColesQuantModel(n_teams=quant_raw['n_teams'])
    quant_model.load_state_dict(quant_raw['model_state'])
    quant_model.eval()
    
    models_dict['quant'] = {
        'model': quant_model,
        'team_mapping': quant_raw['team_mapping'],
        'n_teams': quant_raw['n_teams'],
        'kalman_gamma': quant_raw.get('kalman_gamma', 0.001)
    }
    models_dict['context'] = joblib.load(CONTEXT_PATH)
    models_dict['nn'] = joblib.load(NN_PATH)
    models_dict['draws'] = joblib.load(DRAWS_PATH)
    models_dict['market'] = joblib.load(MARKET_PATH)
    models_dict['gbm'] = joblib.load(GBM_PATH)
    models_dict['fundamental'] = joblib.load(FUNDAMENTAL_PATH)
    models_dict['final'] = joblib.load(FINAL_PATH)
    models_dict['corners'] = joblib.load(CORNERS_PATH) if os.path.exists(CORNERS_PATH) else None
    models_dict['cards'] = joblib.load(CARDS_PATH) if os.path.exists(CARDS_PATH) else None

    return models_dict

def get_latest_team_row(df, team_name):
    """Obtiene la última fila histórica de un equipo."""
    team_df = df[(df['team'] == team_name)].sort_values('match_date')
    if not team_df.empty:
        return team_df.iloc[-1]
    opp_df = df[(df['opponent'] == team_name)].sort_values('match_date')
    if not opp_df.empty:
        return opp_df.iloc[-1]
    return None

def compute_meta_features_live(df_base, open_odds_win, open_odds_draw, open_odds_loss, comp_id=0):
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
    
    mean_probs = pd.DataFrame({'loss': mean_loss, 'draw': mean_draw, 'win': mean_win})
    meta['meta_entropy'] = mean_probs.apply(lambda row: entropy([row['loss'] + 1e-9, row['draw'] + 1e-9, row['win'] + 1e-9]), axis=1)
    
    meta['implied_open_loss'] = 1.0 / max(open_odds_loss, 1.01)
    meta['implied_open_draw'] = 1.0 / max(open_odds_draw, 1.01)
    meta['implied_open_win'] = 1.0 / max(open_odds_win, 1.01)
    meta['meta_competition_id'] = comp_id
    meta['competition_id'] = comp_id
    
    return pd.concat([df_base, meta], axis=1)

def build_live_match_features(df, home_team, away_team, comp, odds_1, odds_X, odds_2):
    """Construye un vector de características completo e insesgado para la predicción en vivo."""
    home_row = get_latest_team_row(df, home_team)
    away_row = get_latest_team_row(df, away_team)
    
    if home_row is None or away_row is None:
        return None

    input_data = {}
    
    # 1. Copiar métricas históricas de rendimiento del local
    for col in home_row.index:
        if isinstance(home_row[col], (int, float, np.number)):
            input_data[col] = float(home_row[col])

    # 2. Reemplazar métricas del rival con el equipo visitante real
    input_data['is_home'] = 1
    input_data['team'] = home_team
    input_data['opponent'] = away_team
    input_data['competition'] = comp
    
    comp_categories = pd.factorize(df['competition'])[1]
    input_data['competition_id'] = comp_categories.get_loc(comp) if comp in comp_categories else 0

    input_data['team_elo'] = float(home_row.get('team_elo', 1500))
    input_data['opp_elo'] = float(away_row.get('team_elo', 1500))
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']

    input_data['team_squad_value'] = float(home_row.get('team_squad_value', 0))
    input_data['opp_squad_value'] = float(away_row.get('team_squad_value', 0))
    input_data['squad_value_diff'] = input_data['team_squad_value'] - input_data['opp_squad_value']

    if 'xg_created_ema5' in away_row:
        input_data['opp_xg_created_ema5'] = float(away_row['xg_created_ema5'])
    if 'xg_conceded_ema5' in away_row:
        input_data['opp_xg_conceded_ema5'] = float(away_row['xg_conceded_ema5'])

    # 3. Cuotas de Apertura, Implícitas y Presión de Mercado (Steam / Line Movement Velocity)
    impl_win = 1.0 / odds_1
    impl_draw = 1.0 / odds_X
    impl_loss = 1.0 / odds_2
    margin = impl_win + impl_draw + impl_loss

    open_1 = float(home_row.get('open_odds_win', odds_1))
    open_X = float(home_row.get('open_odds_draw', odds_X))
    open_2 = float(home_row.get('open_odds_loss', odds_2))

    input_data['open_odds_win'] = open_1
    input_data['open_odds_draw'] = open_X
    input_data['open_odds_loss'] = open_2
    input_data['odds_win'] = odds_1
    input_data['odds_draw'] = odds_X
    input_data['odds_loss'] = odds_2

    input_data['open_prob_win'] = (1.0 / max(open_1, 1.01))
    input_data['open_prob_draw'] = (1.0 / max(open_X, 1.01))
    input_data['open_prob_loss'] = (1.0 / max(open_2, 1.01))
    input_data['prob_win_implied'] = impl_win / margin
    input_data['prob_draw_implied'] = impl_draw / margin
    input_data['prob_loss_implied'] = impl_loss / margin
    input_data['vig_open'] = margin - 1.0
    input_data['vig_close'] = margin - 1.0

    # Inferencia de presión de mercado (Steam)
    input_data['steam_win'] = float((1.0 / max(odds_1, 1.01)) - (1.0 / max(open_1, 1.01)))
    input_data['steam_draw'] = float((1.0 / max(odds_X, 1.01)) - (1.0 / max(open_X, 1.01)))
    input_data['steam_loss'] = float((1.0 / max(odds_2, 1.01)) - (1.0 / max(open_2, 1.01)))

    return input_data

def predict_match(
    home_team,
    away_team,
    comp,
    odds_1,
    odds_X,
    odds_2,
    extra_odds=None,
    bankroll=1000.0,
    injuries_home=0,
    injuries_away=0,
    df=None,
    models_dict=None
):
    """Ejecuta la inferencia cuantitativa completa con Incertidumbre Bayesiana y Kelly Multi-Activo (10/10)."""
    if df is None:
        df = load_data()
        if df is None: return None

    if models_dict is None:
        models_dict = load_all_models()
        if models_dict is None: return None

    extra_odds = extra_odds or {}
    
    # 1. Cargar parámetros cuantitativos por liga
    kelly_fractions, ev_thresholds = {}, {}
    alpha_div_low_dict, alpha_div_med_dict, alpha_div_high_dict = {}, {}, {}
    if os.path.exists(OPTIMIZED_PARAMS_FILE):
        with open(OPTIMIZED_PARAMS_FILE, 'r') as f:
            data = json.load(f)
            kelly_fractions = data.get('KELLY_FRACTIONS', {})
            ev_thresholds = data.get('EV_THRESHOLDS', {})
            alpha_div_low_dict = data.get('ALPHA_DIV_LOW', {})
            alpha_div_med_dict = data.get('ALPHA_DIV_MED', {})
            alpha_div_high_dict = data.get('ALPHA_DIV_HIGH', {})

    league_kelly = get_param_by_comp(kelly_fractions, comp, 0.015)
    league_ev_thresh = get_param_by_comp(ev_thresholds, comp, 0.015)
    league_alpha_low = get_param_by_comp(alpha_div_low_dict, comp, 0.85)
    league_alpha_med = get_param_by_comp(alpha_div_med_dict, comp, 0.70)
    league_alpha_high = get_param_by_comp(alpha_div_high_dict, comp, 0.50)

    # 2. Construir vector de características
    input_data = build_live_match_features(df, home_team, away_team, comp, odds_1, odds_X, odds_2)
    if input_data is None:
        return None

    if injuries_home > 0: input_data['team_elo'] *= (1 - (injuries_home * 0.02))
    if injuries_away > 0: input_data['opp_elo'] *= (1 - (injuries_away * 0.02))
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']

    context_data = models_dict['context']
    nn_data = models_dict['nn']
    draws_data = models_dict['draws']
    gbm_data = models_dict['gbm']
    market_data = models_dict['market']
    fund_data = models_dict['fundamental']
    final_data = models_dict['final']
    quant_dict = models_dict['quant']

    all_req_features = set(context_data['features'] + nn_data['features'] + draws_data['features'])
    for f in all_req_features:
        if f not in input_data:
            input_data[f] = 0.0

    df_input_full = pd.DataFrame([input_data])

    # 3. Características GBM (Bypass si ya están precalculadas)
    has_all_gbm = all(col in input_data for col in gbm_data['features'])
    if not has_all_gbm:
        import logging
        logging.getLogger('train_gbm_model').setLevel(logging.WARNING)
        team_hist = df[df['team'] == home_team].copy()
        current_match_df = pd.DataFrame([input_data])
        current_match_df['match_date'] = pd.Timestamp.now()
        combined_hist = pd.concat([team_hist, current_match_df], ignore_index=True)
        combined_gbm = compute_gbm_features(combined_hist)

        if combined_gbm is not None and not combined_gbm.empty:
            last_gbm = combined_gbm.iloc[-1:]
            for col in gbm_data['features']:
                if col in last_gbm: input_data[col] = last_gbm[col].values[0]
                elif col.startswith('gbm_base_prob_'):
                    clean_col = col.replace('gbm_base_', '')
                    input_data[col] = current_match_df[clean_col].values[0] if clean_col in current_match_df else 0.33
        else:
            for col in gbm_data['features']: input_data[col] = 0.0

    df_input_full = pd.DataFrame([input_data])

    # 4. Predicciones Modelo Quant ZINB Kalman Dixon-Coles
    w_q, d_q, l_q, mu_h_np, mu_a_np = predict_from_model(quant_dict['model'], quant_dict['team_mapping'], df_input_full)
    q_win, q_draw, q_loss = w_q[0], d_q[0], l_q[0]
    xg_s, xg_c = float(mu_h_np[0]), float(mu_a_np[0])

    if injuries_home > 0: xg_s *= (1 - injuries_home * 0.02); xg_c *= (1 + injuries_home * 0.02)
    if injuries_away > 0: xg_c *= (1 - injuries_away * 0.02); xg_s *= (1 + injuries_away * 0.02)

    # 5. Inferencia Submodelos
    df_ctx = df_input_full[context_data['features']].copy()
    if 'competition_id' in df_ctx.columns: df_ctx['competition_id'] = df_ctx['competition_id'].astype('category')
    ctx_probs = context_data['model'].predict_proba(df_ctx)[0]
    
    nn_probs = nn_data['model'].predict_proba(df_input_full[nn_data['features']])[0]
    draws_prob = draws_data['model'].predict_proba(df_input_full[draws_data['features']])[0][1]
    mkt_probs = market_data['model'].predict_proba(df_input_full[market_data['features']])[0]
    gbm_probs = gbm_data['model'].predict_proba(df_input_full[gbm_data['features']])[0]

    # Stacker Fundamental
    df_fund_input = pd.DataFrame({
        'predicted_xg_scored_quant': [xg_s], 'predicted_xg_conceded_quant': [xg_c],
        'quant_win_prob': [q_win], 'quant_draw_prob': [q_draw], 'quant_loss_prob': [q_loss],
        'prob_loss_ctx': [ctx_probs[0]], 'prob_draw_ctx': [ctx_probs[1]], 'prob_win_ctx': [ctx_probs[2]],
        'prob_loss_nn': [nn_probs[0]], 'prob_draw_nn': [nn_probs[1]], 'prob_win_nn': [nn_probs[2]],
        'prob_is_draw': [draws_prob]
    })[fund_data['features']]
    fund_probs = fund_data['model'].predict_proba(df_fund_input)[0]

    # Stacker Final (Nivel 2)
    df_fund_out = pd.DataFrame({'fund_prob_loss': [fund_probs[0]], 'fund_prob_draw': [fund_probs[1]], 'fund_prob_win': [fund_probs[2]]})
    df_mkt = pd.DataFrame({'prob_loss_mkt': [mkt_probs[0]], 'prob_draw_mkt': [mkt_probs[1]], 'prob_win_mkt': [mkt_probs[2]],
                           'prob_loss_gbm': [gbm_probs[0]], 'prob_draw_gbm': [gbm_probs[1]], 'prob_win_gbm': [gbm_probs[2]]})
    
    df_final = compute_meta_features_live(pd.concat([df_fund_out, df_mkt], axis=1), odds_1, odds_X, odds_2, input_data['competition_id'])
    
    for feat in final_data['features']:
        if feat not in df_final.columns:
            df_final[feat] = 0.0

    df_final = df_final[final_data['features']]
    if 'competition_id' in df_final.columns: df_final['competition_id'] = df_final['competition_id'].astype('category')

    final_probs = final_data['model'].predict_proba(df_final)[0]
    final_probs /= np.sum(final_probs)
    prob_loss, prob_draw, prob_win = final_probs[0], final_probs[1], final_probs[2]

    # --- INCERTIDUMBRE BAYESIANA & INTERVALOS DE CREDIBILIDAD 95% ---
    all_win_probs = [q_win, ctx_probs[2], nn_probs[2], mkt_probs[2], gbm_probs[2], fund_probs[2], prob_win]
    all_draw_probs = [q_draw, ctx_probs[1], nn_probs[1], mkt_probs[1], gbm_probs[1], fund_probs[1], prob_draw]
    all_loss_probs = [q_loss, ctx_probs[0], nn_probs[0], mkt_probs[0], gbm_probs[0], fund_probs[0], prob_loss]

    _, se_win, ci_win_low, ci_win_high, pen_win = calculate_bayesian_uncertainty(all_win_probs)
    _, se_draw, ci_draw_low, ci_draw_high, pen_draw = calculate_bayesian_uncertainty(all_draw_probs)
    _, se_loss, ci_loss_low, ci_loss_high, pen_loss = calculate_bayesian_uncertainty(all_loss_probs)

    # 6. Motor Poisson & Mercados Derivados (Fórmulas Matemáticas Exactas)
    rho_val = float(quant_dict['model'].rho.item()) if hasattr(quant_dict['model'], 'rho') else 0.0
    poisson_matrix = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            poisson_matrix[i, j] = poisson.pmf(i, xg_s) * poisson.pmf(j, xg_c)

    if rho_val != 0.0:
        tau_00 = max(1.0 - xg_s * xg_c * rho_val, 0.0)
        tau_10 = max(1.0 + xg_c * rho_val, 0.0)
        tau_01 = max(1.0 + xg_s * rho_val, 0.0)
        tau_11 = max(1.0 - rho_val, 0.0)
        poisson_matrix[0, 0] *= tau_00
        poisson_matrix[1, 0] *= tau_10
        poisson_matrix[0, 1] *= tau_01
        poisson_matrix[1, 1] *= tau_11
        p_sum = np.sum(poisson_matrix)
        if p_sum > 0: poisson_matrix /= p_sum

    goals_grid = np.arange(11)[:, None] + np.arange(11)[None, :]
    prob_over25_goals = float(np.sum(poisson_matrix[goals_grid > 2]))
    prob_under25_goals = 1.0 - prob_over25_goals
    prob_btts_yes = float(np.sum(poisson_matrix[1:, 1:]))
    prob_btts_no = 1.0 - prob_btts_yes

    corners_data = models_dict['corners']
    exp_corners = 9.8
    if corners_data and 'model_corners' in corners_data:
        try:
            df_c_feat = df_input_full[corners_data['features']] if all(c in df_input_full.columns for c in corners_data['features']) else None
            if df_c_feat is not None: exp_corners = float(corners_data['model_corners'].predict(df_c_feat)[0])
        except Exception: pass

    cards_data = models_dict['cards']
    exp_cards = 4.2
    if cards_data:
        try:
            exp_cards = float(cards_data.predict(df_input_full)[0]) if hasattr(cards_data, 'predict') else 4.2
        except Exception: pass

    prob_over95_corners = float(1.0 - sum(poisson.pmf(k, exp_corners) for k in range(10)))
    prob_over45_cards = float(1.0 - sum(poisson.pmf(k, exp_cards) for k in range(5)))

    # 7. Dynamic Alpha Blending & Net Expected Values (EV)
    impl_win, impl_draw, impl_loss = 1.0/odds_1, 1.0/odds_X, 1.0/odds_2
    margin = impl_win + impl_draw + impl_loss
    market_prob_win, market_prob_draw, market_prob_loss = impl_win/margin, impl_draw/margin, impl_loss/margin

    alpha_win = calculate_dynamic_alpha(prob_win, market_prob_win, league_alpha_low, league_alpha_med, league_alpha_high)
    alpha_draw = calculate_dynamic_alpha(prob_draw, market_prob_draw, league_alpha_low, league_alpha_med, league_alpha_high)
    alpha_loss = calculate_dynamic_alpha(prob_loss, market_prob_loss, league_alpha_low, league_alpha_med, league_alpha_high)

    blend_win = (alpha_win * prob_win) + ((1.0 - alpha_win) * market_prob_win)
    blend_draw = (alpha_draw * prob_draw) + ((1.0 - alpha_draw) * market_prob_draw)
    blend_loss = (alpha_loss * prob_loss) + ((1.0 - alpha_loss) * market_prob_loss)

    net_odds_1 = 1.0 + (odds_1 - 1.0) * (1.0 - TAX_RETENTION_RATE)
    net_odds_X = 1.0 + (odds_X - 1.0) * (1.0 - TAX_RETENTION_RATE)
    net_odds_2 = 1.0 + (odds_2 - 1.0) * (1.0 - TAX_RETENTION_RATE)

    ev_win = (blend_win * net_odds_1) - 1.0 - EXPECTED_CLV_DROP
    ev_draw = (blend_draw * net_odds_X) - 1.0 - EXPECTED_CLV_DROP
    ev_loss = (blend_loss * net_odds_2) - 1.0 - EXPECTED_CLV_DROP

    # Dutching 1X & X2
    combined_odds_1X = 1.0 / (impl_win + impl_draw)
    net_combined_1X = 1.0 + (combined_odds_1X - 1.0) * (1.0 - TAX_RETENTION_RATE)
    blend_1X = blend_win + blend_draw
    ev_1X = (blend_1X * net_combined_1X) - 1.0 - EXPECTED_CLV_DROP

    combined_odds_X2 = 1.0 / (impl_draw + impl_loss)
    net_combined_X2 = 1.0 + (combined_odds_X2 - 1.0) * (1.0 - TAX_RETENTION_RATE)
    blend_X2 = blend_draw + blend_loss
    ev_X2 = (blend_X2 * net_combined_X2) - 1.0 - EXPECTED_CLV_DROP

    # Mercados Derivados
    odd_ou25 = extra_odds.get('ou25_over', 1.0 / max(prob_over25_goals, 0.05))
    ev_ou25 = (prob_over25_goals * (1.0 + (odd_ou25 - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    odd_btts = extra_odds.get('btts_yes', 1.0 / max(prob_btts_yes, 0.05))
    ev_btts = (prob_btts_yes * (1.0 + (odd_btts - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    odd_corners = extra_odds.get('corners_over95', 1.0 / max(prob_over95_corners, 0.05))
    ev_corners = (prob_over95_corners * (1.0 + (odd_corners - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    odd_cards = extra_odds.get('cards_over45', 1.0 / max(prob_over45_cards, 0.05))
    ev_cards = (prob_over45_cards * (1.0 + (odd_cards - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

    # --- OPTIMIZACIÓN DE PORTAFOLIO KELLY MULTI-ACTIVO CON MATRIZ DE COVARIANZAS ---
    corr_matrix = compute_poisson_market_correlations(poisson_matrix)
    ev_vector = [ev_win, ev_draw, ev_loss, ev_1X, ev_X2, ev_ou25, ev_btts, ev_corners, ev_cards]
    odds_vector = [odds_1, odds_X, odds_2, combined_odds_1X, combined_odds_X2, odd_ou25, odd_btts, odd_corners, odd_cards]
    prob_vector = [blend_win, blend_draw, blend_loss, blend_1X, blend_X2, prob_over25_goals, prob_btts_yes, prob_over95_corners, prob_over45_cards]

    avg_penalty = np.mean([pen_win, pen_draw, pen_loss])
    multi_stakes, multi_pcts = solve_multi_asset_kelly(
        ev_vector, odds_vector, prob_vector, corr_matrix, bankroll, MAX_STAKE_PCT,
        penalty=avg_penalty, max_liquidity=max_liquidity, league_kelly=league_kelly
    )

    s_win, s_draw, s_loss, s_1X, s_X2, s_ou25, s_btts, s_corn, s_cards = multi_stakes
    p_win_stk, p_draw_stk, p_loss_stk, p_1X_stk, p_X2_stk, p_ou25_stk, p_btts_stk, p_corn_stk, p_cards_stk = multi_pcts

    eff_1 = calculate_market_slippage(odds_1, s_win, max_liquidity)
    eff_X = calculate_market_slippage(odds_X, s_draw, max_liquidity)
    eff_2 = calculate_market_slippage(odds_2, s_loss, max_liquidity)
    eff_1X = calculate_market_slippage(combined_odds_1X, s_1X, max_liquidity)
    eff_X2 = calculate_market_slippage(combined_odds_X2, s_X2, max_liquidity)

    all_bets = [
        ("1 (Local)", ev_win, s_win, p_win_stk, odds_1, eff_1, blend_win),
        ("X (Empate)", ev_draw, s_draw, p_draw_stk, odds_X, eff_X, blend_draw),
        ("2 (Visitante)", ev_loss, s_loss, p_loss_stk, odds_2, eff_2, blend_loss),
        ("Doble Oportunidad 1X", ev_1X, s_1X, p_1X_stk, combined_odds_1X, eff_1X, blend_1X),
        ("Doble Oportunidad X2", ev_X2, s_X2, p_X2_stk, combined_odds_X2, eff_X2, blend_X2),
        ("Over 2.5 Goles", ev_ou25, s_ou25, p_ou25_stk, odd_ou25, odd_ou25, prob_over25_goals),
        ("BTTS (Ambos Anotan)", ev_btts, s_btts, p_btts_stk, odd_btts, odd_btts, prob_btts_yes),
        ("Over 9.5 Córneres", ev_corners, s_corn, p_corn_stk, odd_corners, odd_corners, prob_over95_corners),
        ("Over 4.5 Tarjetas", ev_cards, s_cards, p_cards_stk, odd_cards, odd_cards, prob_over45_cards)
    ]
    
    all_bets.sort(key=lambda x: x[1], reverse=True)
    best_bet = all_bets[0]

    return {
        'home_team': home_team, 'away_team': away_team, 'competition': comp,
        'xg_scored': xg_s, 'xg_conceded': xg_c,
        'exp_corners': exp_corners, 'exp_cards': exp_cards,
        'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
        'ci_win_low': ci_win_low, 'ci_win_high': ci_win_high, 'se_win': se_win,
        'ci_draw_low': ci_draw_low, 'ci_draw_high': ci_draw_high, 'se_draw': se_draw,
        'ci_loss_low': ci_loss_low, 'ci_loss_high': ci_loss_high, 'se_loss': se_loss,
        'blend_win': blend_win, 'blend_draw': blend_draw, 'blend_loss': blend_loss,
        'prob_over25_goals': prob_over25_goals, 'prob_under25_goals': prob_under25_goals,
        'prob_btts_yes': prob_btts_yes, 'prob_btts_no': prob_btts_no,
        'prob_over95_corners': prob_over95_corners, 'prob_over45_cards': prob_over45_cards,
        'ev_win': ev_win, 'ev_draw': ev_draw, 'ev_loss': ev_loss,
        'ev_1X': ev_1X, 'ev_X2': ev_X2, 'ev_ou25': ev_ou25, 'ev_btts': ev_btts,
        'ev_corners': ev_corners, 'ev_cards': ev_cards,
        'stake_win': s_win, 'stake_draw': s_draw, 'stake_loss': s_loss,
        'stake_1X': s_1X, 'stake_X2': s_X2, 'stake_ou25': s_ou25, 'stake_btts': s_btts,
        'stake_corners': s_corn, 'stake_cards': s_cards,
        'pct_win': p_win_stk, 'pct_draw': p_draw_stk, 'pct_loss': p_loss_stk,
        'pct_1X': p_1X_stk, 'pct_X2': p_X2_stk, 'pct_ou25': p_ou25_stk, 'pct_btts': p_btts_stk,
        'pct_corners': p_corn_stk, 'pct_cards': p_cards_stk,
        'eff_1': eff_1, 'eff_X': eff_X, 'eff_2': eff_2, 'eff_1X': eff_1X, 'eff_X2': eff_X2,
        'league_ev_thresh': league_ev_thresh, 'league_kelly': league_kelly,
        'bayesian_penalty': avg_penalty, 'all_bets': all_bets, 'best_bet': best_bet
    }

def main():
    console.print(Panel.fit("[bold cyan]Prophetia2 - Institutional Multi-Market Quant Value CLI (10/10)[/bold cyan]\n[dim]Initializing Stacker Meta-Model, Bayesian Credible Intervals & Multi-Asset Kelly Portfolio...[/dim]"))
    
    models_dict = load_all_models()
    if models_dict is None: return

    df = load_data()
    if df is None: return

    competitions = df['competition'].dropna().unique().tolist()
    comp = questionary.select("Selecciona la Liga:", choices=sorted(competitions)).ask()
    if not comp: return
    
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
    except (ValueError, TypeError):
        console.print("[red]Cuotas inválidas.[/red]")
        return
        
    want_extra_odds = questionary.confirm("¿Deseas ingresar cuotas para mercados derivados (Goles O/U 2.5, BTTS, Handicap, Córneres, Tarjetas)?", default=False).ask()

    extra_odds = {}
    if want_extra_odds:
        try:
            extra_odds['ou25_over'] = float(questionary.text("Cuota Over 2.5 Goles:", default="1.95").ask())
            extra_odds['ou25_under'] = float(questionary.text("Cuota Under 2.5 Goles:", default="1.95").ask())
            extra_odds['btts_yes'] = float(questionary.text("Cuota Both Teams to Score (BTTS Yes):", default="1.85").ask())
            extra_odds['ah_minus05_home'] = float(questionary.text(f"Cuota Handicap Asiático -0.5 {home_team}:", default=str(odds_1)).ask())
            extra_odds['corners_over95'] = float(questionary.text("Cuota Over 9.5 Córneres Totales:", default="1.90").ask())
            extra_odds['cards_over45'] = float(questionary.text("Cuota Over 4.5 Tarjetas Totales:", default="1.90").ask())
        except (ValueError, TypeError):
            console.print("[yellow]Cuotas adicionales inválidas, usando estimaciones por modelo.[/yellow]")
            extra_odds = {}

    try:
        injuries_home = int(questionary.text("Lesiones clave Local [0-5]:", default="0").ask())
        injuries_away = int(questionary.text("Lesiones clave Visitante [0-5]:", default="0").ask())
        bankroll = float(questionary.text("Bankroll actual ($):", default="1000").ask())
    except (ValueError, TypeError):
        injuries_home, injuries_away, bankroll = 0, 0, 1000.0

    res = predict_match(
        home_team, away_team, comp, odds_1, odds_X, odds_2,
        extra_odds=extra_odds, bankroll=bankroll,
        injuries_home=injuries_home, injuries_away=injuries_away,
        df=df, models_dict=models_dict
    )

    if res is None:
        console.print("[red]Error ejecutando predicción.[/red]")
        return

    console.print(f"[dim]Parámetros cuantitativos cargados para {comp}: EV Threshold = {res['league_ev_thresh']*100:.2f}%, Kelly = {res['league_kelly']:.4f} | Penalizador Incertidumbre = {res['bayesian_penalty']:.3f}[/dim]")
    console.print("\n[bold]=== PROYECCIONES MULTI-MERCADO & EXPECTED VALUES ===[/bold]")
    console.print(f"[bold cyan]Marcador Esperado Quant (xG):[/bold cyan] {home_team} [bold yellow]{res['xg_scored']:.2f} - {res['xg_conceded']:.2f}[/bold yellow] {away_team}")
    console.print(f"[bold cyan]Córneres Totales Esperados:[/bold cyan] [bold yellow]{res['exp_corners']:.1f}[/bold yellow] | [bold cyan]Tarjetas Totales Esperadas:[/bold cyan] [bold yellow]{res['exp_cards']:.1f}[/bold yellow]\n")

    def c_ev(ev): return f"[green]+{ev*100:.1f}%[/green]" if ev > res['league_ev_thresh'] else (f"[yellow]+{ev*100:.1f}%[/yellow]" if ev > 0 else f"[red]{ev*100:.1f}%[/red]")

    t1 = Table(title="1. Mercados de Partido (1X2, Doble Oportunidad & Intervalos de Credibilidad 95%)", show_header=True, header_style="bold magenta")
    t1.add_column("Mercado", style="cyan")
    t1.add_column("Odds (Efectiva)")
    t1.add_column("Prob. Blended", justify="right")
    t1.add_column("IC 95% Bayesiano", justify="center")
    t1.add_column("Net EV", justify="right")
    t1.add_column("Stake Kelly Multi ($ / %)", justify="right")

    t1.add_row(f"1 (Local - {home_team})", f"{odds_1:.2f} ({res['eff_1']:.2f})", f"{res['blend_win']*100:.1f}%", f"[{res['ci_win_low']*100:.1f}% - {res['ci_win_high']*100:.1f}%]", c_ev(res['ev_win']), f"${res['stake_win']:.2f} ({res['pct_win']*100:.2f}%)")
    t1.add_row("X (Empate)", f"{odds_X:.2f} ({res['eff_X']:.2f})", f"{res['blend_draw']*100:.1f}%", f"[{res['ci_draw_low']*100:.1f}% - {res['ci_draw_high']*100:.1f}%]", c_ev(res['ev_draw']), f"${res['stake_draw']:.2f} ({res['pct_draw']*100:.2f}%)")
    t1.add_row(f"2 (Visita - {away_team})", f"{odds_2:.2f} ({res['eff_2']:.2f})", f"{res['blend_loss']*100:.1f}%", f"[{res['ci_loss_low']*100:.1f}% - {res['ci_loss_high']*100:.1f}%]", c_ev(res['ev_loss']), f"${res['stake_loss']:.2f} ({res['pct_loss']*100:.2f}%)")
    t1.add_row("1X / AH +0.5 Local", f"{1.0/(1.0/odds_1+1.0/odds_X):.2f} ({res['eff_1X']:.2f})", f"{(res['blend_win']+res['blend_draw'])*100:.1f}%", "-", c_ev(res['ev_1X']), f"${res['stake_1X']:.2f} ({res['pct_1X']*100:.2f}%)")
    t1.add_row("X2 / AH +0.5 Visita", f"{1.0/(1.0/odds_X+1.0/odds_2):.2f} ({res['eff_X2']:.2f})", f"{(res['blend_draw']+res['blend_loss'])*100:.1f}%", "-", c_ev(res['ev_X2']), f"${res['stake_X2']:.2f} ({res['pct_X2']*100:.2f}%)")
    console.print(t1)

    t2 = Table(title="2. Mercados Derivados (Goles O/U, BTTS, Córneres & Tarjetas)", show_header=True, header_style="bold blue")
    t2.add_column("Mercado Derivado / Prop", style="cyan")
    t2.add_column("Odds Entrada")
    t2.add_column("Prob. Modelo", justify="right")
    t2.add_column("Net EV", justify="right")
    t2.add_column("Stake Kelly Multi ($ / %)", justify="right")

    odd_ou25 = extra_odds.get('ou25_over', 1.0 / max(res['prob_over25_goals'], 0.05))
    odd_btts = extra_odds.get('btts_yes', 1.0 / max(res['prob_btts_yes'], 0.05))
    odd_corn = extra_odds.get('corners_over95', 1.0 / max(res['prob_over95_corners'], 0.05))
    odd_card = extra_odds.get('cards_over45', 1.0 / max(res['prob_over45_cards'], 0.05))

    t2.add_row("Over 2.5 Goles Totales", f"{odd_ou25:.2f}", f"{res['prob_over25_goals']*100:.1f}%", c_ev(res['ev_ou25']), f"${res['stake_ou25']:.2f} ({res['pct_ou25']*100:.2f}%)")
    t2.add_row("Under 2.5 Goles Totales", f"{1.0/max(res['prob_under25_goals'], 0.01):.2f}", f"{res['prob_under25_goals']*100:.1f}%", "[dim]Model Fair[/dim]", "$0.00 (0.00%)")
    t2.add_row("Both Teams to Score (BTTS Yes)", f"{odd_btts:.2f}", f"{res['prob_btts_yes']*100:.1f}%", c_ev(res['ev_btts']), f"${res['stake_btts']:.2f} ({res['pct_btts']*100:.2f}%)")
    t2.add_row("Over 9.5 Córneres Totales", f"{odd_corn:.2f}", f"{res['prob_over95_corners']*100:.1f}%", c_ev(res['ev_corners']), f"${res['stake_corners']:.2f} ({res['pct_corners']*100:.2f}%)")
    t2.add_row("Over 4.5 Tarjetas Totales", f"{odd_card:.2f}", f"{res['prob_over45_cards']*100:.1f}%", c_ev(res['ev_cards']), f"${res['stake_cards']:.2f} ({res['pct_cards']*100:.2f}%)")
    console.print(t2)

    best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']

    console.print("\n[bold]=== RECOMENDACIÓN DE STAKING MULTI-MERCADO INSTITUCIONAL ===[/bold]")
    if best_ev > res['league_ev_thresh']:
        if best_stake <= 0 or best_pct < 0.0001:
            console.print(f"[yellow]El mejor EV ({best_bet_name} +{best_ev*100:.2f}%) supera el umbral de la liga ({res['league_ev_thresh']*100:.2f}%), pero el stake Kelly Multi-Activo es minúsculo (< 0.01%).[/yellow] -> [bold]PASS / NO BET[/bold]")
        else:
            rec = Panel(
                f"[bold green]MEJOR OPORTUNIDAD VALUE DETECTADA[/bold green]\n"
                f"Mercado Seleccionado: [bold]{best_bet_name}[/bold] @ {best_raw_odd:.2f} (Efectiva: {best_eff_odd:.2f})\n"
                f"Net EV Proyectado: [bold]+{best_ev*100:.2f}%[/bold] (Umbral Liga {comp}: {res['league_ev_thresh']*100:.2f}%)\n"
                f"Stake Kelly Multi-Activo: [bold]${best_stake:.2f}[/bold] ({best_pct*100:.2f}% del bankroll)",
                title="SISTEMA DE STAKING INSTITUCIONAL MULTI-MERCADO", border_style="green"
            )
            console.print(rec)
            
            # Exportar archivo de señales
            exp_file = export_trade_signals(res)
            console.print(f"[dim]Señal de trading exportada a: {exp_file}[/dim]")
    elif best_ev > 0:
        console.print(Panel(f"[yellow]EDGE INSUFICIENTE EN TODOS LOS MERCADOS.[/yellow]\nEl mejor EV proyectado fue en '{best_bet_name}' con +{best_ev*100:.2f}%. Umbral liga {comp}: {res['league_ev_thresh']*100:.2f}%. -> [bold]PASS / NO BET[/bold]", title="DECISIÓN SISTEMA", border_style="yellow"))
    else:
        console.print(Panel("[bold red]NO HAY VALUE EN NINGÚN MERCADO EN ESTE PARTIDO.[/bold red]\nLas cuotas del mercado son más eficientes que las proyecciones cuantitativas.", title="DECISIÓN SISTEMA", border_style="red"))

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
