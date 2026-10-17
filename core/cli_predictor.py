import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import entropy, poisson, nbinom
import questionary
from questionary import Style, Choice
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add core path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from models.train_quant_advanced import predict_from_model, KalmanDixonColesQuantModel
from market_models.train_gbm_model import compute_gbm_features
from market_models.train_market import compute_shin_probs, compute_power_probs
from models.train_nn import SklearnPyTorchWrapper10, PyTorchGatedResNet10, GatedContext, ResidualBlock
from models.train_draws import HybridDrawsEnsemble
from models.train_cards_model import build_perspective_features, calc_cards_probabilities_vectorized
from ingestion.live_odds_feed import LiveOddsFeedFetcher
from ingestion.lineup_fetcher import LineupImpactFetcher
from team_mapping import normalize_team_name
from markowitz_optimizer import MarkowitzPortfolioOptimizer

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
CARDS_HOME_PATH = os.path.join(MODEL_DIR, 'cards_home_xgboost_model.pkl')
CARDS_AWAY_PATH = os.path.join(MODEL_DIR, 'cards_away_xgboost_model.pkl')
CARDS_PATH = os.path.join(MODEL_DIR, 'cards_total_xgboost_model.pkl')

GBM_DATASET_PATH = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_with_gbm_features.parquet'))
REF_DATASET_PATH = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_with_referees.parquet'))
DATASET_PATH = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_with_odds.parquet'))
FALLBACK_DATASET = os.path.abspath(os.path.join(script_dir, '../data/processed/matches_dataset.parquet'))
OPTIMIZED_PARAMS_FILE = os.path.abspath(os.path.join(script_dir, '../data/processed/models_best_parameters/optimal_bankroll_params.json'))

# --- ESTILO DE COLORES PERSONALIZADO PARA QUESTIONARY CLI ---
CUSTOM_QUESTIONARY_STYLE = Style([
    ('qmark', 'fg:#00d2ff bold'),       # Signo de interrogación en cian eléctrico
    ('question', 'fg:#ffffff bold'),    # Texto de la pregunta en blanco brillante
    ('answer', 'fg:#00ff87 bold'),      # Respuesta seleccionada en verde neón
    ('pointer', 'fg:#00d2ff bold'),     # Puntero de selección en cian
    ('highlighted', 'fg:#00d2ff bold'), # Opción resaltada en cian
    ('selected', 'fg:#00ff87 bold'),    # Opción elegida
    ('separator', 'fg:#6c757d'),        # Separadores
    ('instruction', 'fg:#6c757d'),      # Instrucciones
    ('text', 'fg:#f8f9fa'),             # Texto plano
    ('choice', 'fg:#38bdf8'),           # Texto de opciones en azul cielo
])

# --- DICCIONARIO DE NOMBRES Y BANDERAS OFICIALES DE LIGAS ---
COMPETITION_DISPLAY_NAMES = {
    # Inglaterra
    'Premier League': '🇬🇧 Premier League (Inglaterra)',
    'E0': '🇬🇧 Premier League (Inglaterra)',
    'Championship': '🇬🇧 EFL Championship (Inglaterra)',
    'E1': '🇬🇧 EFL Championship (Inglaterra)',
    'League One': '🇬🇧 EFL League One (Inglaterra)',
    'E2': '🇬🇧 EFL League One (Inglaterra)',
    
    # España
    'La Liga': '🇪🇸 La Liga EA Sports (España)',
    'SP1': '🇪🇸 La Liga EA Sports (España)',
    'La Liga 2': '🇪🇸 La Liga Hypermotion (España)',
    'SP2': '🇪🇸 La Liga Hypermotion (España)',
    
    # Alemania
    '1. Bundesliga': '🇩🇪 1. Bundesliga (Alemania)',
    'D1': '🇩🇪 1. Bundesliga (Alemania)',
    '2. Bundesliga': '🇩🇪 2. Bundesliga (Alemania)',
    'D2': '🇩🇪 2. Bundesliga (Alemania)',
    
    # Italia
    'Serie A': '🇮🇹 Serie A Enilive (Italia)',
    'I1': '🇮🇹 Serie A Enilive (Italia)',
    'Serie B': '🇮🇹 Serie BKT (Italia)',
    'I2': '🇮🇹 Serie BKT (Italia)',
    
    # Francia
    'Ligue 1': '🇫🇷 Ligue 1 McDonald\'s (Francia)',
    'F1': '🇫🇷 Ligue 1 McDonald\'s (Francia)',
    'Ligue 2': '🇫🇷 Ligue 2 (Francia)',
    'F2': '🇫🇷 Ligue 2 (Francia)',
    
    # Japón
    'J1': '🇯🇵 Meiji Yasuda J1 League (Japón)',
    'J-League 1': '🇯🇵 Meiji Yasuda J1 League (Japón)',
    'JPN': '🇯🇵 Meiji Yasuda J1 League (Japón)',
    
    # Países Bajos, Bélgica, Portugal, Turquía, Grecia, Escocia
    'Eredivisie': '🇳🇱 Eredivisie (Países Bajos)',
    'N1': '🇳🇱 Eredivisie (Países Bajos)',
    'Jupiler Pro League': '🇧🇪 Jupiler Pro League (Bélgica)',
    'B1': '🇧🇪 Jupiler Pro League (Bélgica)',
    'Primeira Liga': '🇵🇹 Liga Portugal Betclic (Portugal)',
    'P1': '🇵🇹 Liga Portugal Betclic (Portugal)',
    'Süper Lig': '🇹🇷 Trendyol Süper Lig (Turquía)',
    'T1': '🇹🇷 Trendyol Süper Lig (Turquía)',
    'Super League': '🇬🇷 Stoiximan Super League (Grecia)',
    'G1': '🇬🇷 Stoiximan Super League (Grecia)',
    'Scottish Premiership': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership (Escocia)',
    'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership (Escocia)',
    
    # Suecia, Noruega, Dinamarca, Suiza, Austria
    'Allsvenskan': '🇸🇪 Allsvenskan (Suecia)',
    'SWE': '🇸🇪 Allsvenskan (Suecia)',
    'Eliteserien': '🇳🇴 Eliteserien (Noruega)',
    'NOR': '🇳🇴 Eliteserien (Noruega)',
    'Superligaen': '🇩🇰 3F Superliga (Dinamarca)',
    'DNK': '🇩🇰 3F Superliga (Dinamarca)',
    'Swiss Super League': '🇨🇭 Credit Suisse Super League (Suiza)',
    'SWZ': '🇨🇭 Credit Suisse Super League (Suiza)',
    'Austrian Bundesliga': '🇦🇹 Admiral Bundesliga (Austria)',
    'AUT': '🇦🇹 Admiral Bundesliga (Austria)',
    
    # América & Internacional
    'Major League Soccer': '🇺🇸 Major League Soccer (EEUU)',
    'MLS': '🇺🇸 Major League Soccer (EEUU)',
    'Liga MX': '🇲🇽 Liga MX (México)',
    'MEX': '🇲🇽 Liga MX (México)',
    'MEX1': '🇲🇽 Liga MX (México)',
    'Champions League': '🇪🇺 UEFA Champions League',
    'CL': '🇪🇺 UEFA Champions League',
    'Europa League': '🇪🇺 UEFA Europa League',
    'EL': '🇪🇺 UEFA Europa League',
    'FIFA World Cup': '🏆 FIFA World Cup (Mundial)',
    'WC': '🏆 FIFA World Cup (Mundial)'
}

def get_formatted_comp_name(comp):
    return COMPETITION_DISPLAY_NAMES.get(comp, f"🏳️ {comp}")

# --- DICCIONARIO DE NOMBRES OFICIALES DE CASAS DE APUESTAS ---
OFFICIAL_TEAM_DISPLAY_NAMES = {
    # España (La Liga & La Liga 2)
    'Bilbao': 'Athletic Club',
    'Athletic Bilbao': 'Athletic Club',
    'Ath Bilbao': 'Athletic Club',
    'Ath Madrid': 'Atlético de Madrid',
    'Atletico Madrid': 'Atlético de Madrid',
    'Barcelona': 'FC Barcelona',
    'Barca': 'FC Barcelona',
    'Real Madrid': 'Real Madrid CF',
    'Betis': 'Real Betis Balompié',
    'Real Betis': 'Real Betis Balompié',
    'Sociedad': 'Real Sociedad',
    'Celta': 'RC Celta de Vigo',
    'Celta Vigo': 'RC Celta de Vigo',
    'Sevilla': 'Sevilla FC',
    'Valencia': 'Valencia CF',
    'Villarreal': 'Villarreal CF',
    'Espanyol': 'RC D Espanyol',
    'Mallorca': 'RCD Mallorca',
    'Osasuna': 'CA Osasuna',
    'Rayo Vallecano': 'Rayo Vallecano',
    'Rayo': 'Rayo Vallecano',
    'Getafe': 'Getafe CF',
    'Girona': 'Girona FC',
    'Alaves': 'Deportivo Alavés',
    'Deportivo Alaves': 'Deportivo Alavés',
    'Las Palmas': 'UD Las Palmas',
    'Leganes': 'CD Leganés',
    'Valladolid': 'Real Valladolid',
    
    # Inglaterra (Premier League & Championship)
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Spurs': 'Tottenham Hotspur',
    'Tottenham': 'Tottenham Hotspur',
    'Wolves': 'Wolverhampton Wanderers',
    'Wolverhampton': 'Wolverhampton Wanderers',
    'Newcastle': 'Newcastle United',
    'West Ham': 'West Ham United',
    'Leicester': 'Leicester City',
    'Ipswich': 'Ipswich Town',
    'Southampton': 'Southampton FC',
    'Bournemouth': 'AFC Bournemouth',
    'Nottingham': 'Nottingham Forest',
    
    # Italia (Serie A)
    'Inter': 'Inter Milan',
    'Internazionale': 'Inter Milan',
    'Milan': 'AC Milan',
    'Juve': 'Juventus FC',
    'Juventus': 'Juventus FC',
    'Fiorentina': 'ACF Fiorentina',
    'Roma': 'AS Roma',
    'Lazio': 'SS Lazio',
    'Napoli': 'SSC Napoli',
    'Atalanta': 'Atalanta BC',
    
    # Alemania (Bundesliga)
    'Bayern Munich': 'Bayern München',
    'Bayern': 'Bayern München',
    'Dortmund': 'Borussia Dortmund',
    'B. Dortmund': 'Borussia Dortmund',
    'Leverkusen': 'Bayer 04 Leverkusen',
    'Bayer Leverkusen': 'Bayer 04 Leverkusen',
    'RB Leipzig': 'RB Leipzig',
    'Gladbach': 'Borussia Mönchengladbach',
    'Mönchengladbach': 'Borussia Mönchengladbach',
    'Eintracht Frankfurt': 'Eintracht Frankfurt',
    
    # Francia (Ligue 1)
    'PSG': 'Paris Saint-Germain',
    'Paris SG': 'Paris Saint-Germain',
    'Monaco': 'AS Monaco',
    'Marseille': 'Olympique de Marseille',
    'Lyon': 'Olympique Lyonnais',
    'Lille': 'LOSC Lille'
}

def get_official_team_name(name):
    if not name:
        return ""
    clean = str(name).strip()
    return OFFICIAL_TEAM_DISPLAY_NAMES.get(clean, clean)

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

from league_mapping import CANONICAL_LEAGUES, COMPETITION_MAPPING, normalize_league, get_param_by_comp

def calculate_dynamic_alpha(p_model, p_market, alpha_low, alpha_med, alpha_high):
    """ Función continua (smooth decay) para blending de probabilidades basada en divergencia relativa """
    rel_divergence = abs(p_model - p_market) / max(p_market, 0.05)
    alpha_max = max(alpha_low, alpha_med, alpha_high)
    alpha_min = min(alpha_low, alpha_med, alpha_high)
    decay = np.exp(-6.0 * (rel_divergence ** 2))
    return alpha_min + (alpha_max - alpha_min) * decay

def calculate_market_slippage(odds, stake, liquidity_cap):
    if stake <= 0 or liquidity_cap <= 0:
        return odds
    ratio = min(stake / liquidity_cap, 1.0)
    impact = MARKET_IMPACT_GAMMA * np.sqrt(ratio)
    effective_odds = 1.0 + (odds - 1.0) * (1.0 - impact)
    return max(effective_odds, 1.001)

def calculate_bayesian_uncertainty(probs_list):
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
    Asignación de portafolio multi-activo basada en la Teoría de Portafolio de Harry Markowitz (1952) (Mean-Variance Optimization).
    """
    n = len(ev_vec)
    if n == 0:
        return [], []
    
    scale = float(penalty * (league_kelly / 0.015))
    penalties = [scale] * n
    liquidity_caps = [max_liquidity] * n

    optimizer = MarkowitzPortfolioOptimizer(
        risk_aversion=2.0,
        max_stake_pct=max_stake_pct,
        max_daily_portfolio_pct=0.15
    )

    res = optimizer.optimize_mean_variance(
        ev_vec=ev_vec,
        odds_vec=odds_vec,
        prob_vec=prob_vec,
        corr_matrix=corr_matrix,
        bankroll=bankroll,
        uncertainty_penalties=penalties,
        liquidity_caps=liquidity_caps
    )

    stakes = [float(s) if ev_vec[i] > 0.0 else 0.0 for i, s in enumerate(res['stakes'])]
    pcts = [float(p) if ev_vec[i] > 0.0 else 0.0 for i, p in enumerate(res['weights'])]
    return stakes, pcts

def compute_poisson_market_correlations(poisson_matrix):
    rows, cols = poisson_matrix.shape
    weights = poisson_matrix.flatten()
    
    ind_win = np.fromfunction(lambda i, j: i > j, (rows, cols)).flatten().astype(float)
    ind_draw = np.fromfunction(lambda i, j: i == j, (rows, cols)).flatten().astype(float)
    ind_loss = np.fromfunction(lambda i, j: i < j, (rows, cols)).flatten().astype(float)
    ind_1X = np.fromfunction(lambda i, j: i >= j, (rows, cols)).flatten().astype(float)
    ind_X2 = np.fromfunction(lambda i, j: i <= j, (rows, cols)).flatten().astype(float)
    ind_ou25 = np.fromfunction(lambda i, j: (i + j) > 2, (rows, cols)).flatten().astype(float)
    ind_ou25_un = np.fromfunction(lambda i, j: (i + j) <= 2, (rows, cols)).flatten().astype(float)
    ind_btts = np.fromfunction(lambda i, j: (i >= 1) & (j >= 1), (rows, cols)).flatten().astype(float)
    ind_corn = np.ones(rows * cols) * 0.5
    ind_corn_un = np.ones(rows * cols) * 0.5
    ind_card = np.ones(rows * cols) * 0.5

    matrix_ind = np.vstack([
        ind_win, ind_draw, ind_loss, ind_1X, ind_X2,
        ind_ou25, ind_ou25_un, ind_btts,
        ind_corn, ind_corn_un, ind_card
    ])
    
    mean_vec = matrix_ind @ weights
    dev = matrix_ind - mean_vec[:, None]
    weighted_cov = (dev * weights[None, :]) @ dev.T
    
    std_vec = np.sqrt(np.diag(weighted_cov))
    std_vec[std_vec == 0] = 1.0
    corr_matrix = weighted_cov / np.outer(std_vec, std_vec)
    np.fill_diagonal(corr_matrix, 1.0)
    # Perfect negative correlation between complementary pairs
    corr_matrix[5, 6] = -0.99
    corr_matrix[6, 5] = -0.99
    corr_matrix[8, 9] = -0.99
    corr_matrix[9, 8] = -0.99
    return np.clip(corr_matrix, -1.0, 1.0)

def export_trade_signals(res, filepath='trade_signals.json'):
    data = {
        'timestamp': datetime.now().isoformat(),
        'match': f"{res['home_official']} vs {res['away_official']}",
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
    for candidate_path in [GBM_DATASET_PATH, REF_DATASET_PATH, DATASET_PATH, FALLBACK_DATASET]:
        if os.path.exists(candidate_path):
            return pd.read_parquet(candidate_path, engine='fastparquet')
    console.print(f"[red]Error: Dataset no encontrado en {DATASET_PATH}[/red]")
    return None

def load_all_models():
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
    if os.path.exists(CARDS_HOME_PATH) and os.path.exists(CARDS_AWAY_PATH):
        models_dict['cards'] = {
            'home': joblib.load(CARDS_HOME_PATH),
            'away': joblib.load(CARDS_AWAY_PATH)
        }
    elif os.path.exists(CARDS_PATH):
        models_dict['cards'] = joblib.load(CARDS_PATH)
    else:
        models_dict['cards'] = None

    return models_dict

def get_latest_team_row(df, team_name, as_of_date=None):
    if not team_name:
        return None
    norm = normalize_team_name(team_name)
    
    def _filter_date(sub_df):
        if as_of_date is not None and not sub_df.empty and 'match_date' in sub_df.columns:
            as_of_val = as_of_date
            if pd.api.types.is_datetime64_any_dtype(sub_df['match_date']):
                as_of_val = pd.to_datetime(as_of_date)
            elif not isinstance(as_of_date, str):
                as_of_val = str(as_of_date)
            dated_df = sub_df[sub_df['match_date'] < as_of_val]
            if not dated_df.empty:
                return dated_df
        return sub_df
    
    # 1. Búsqueda exacta por nombre o normalizado
    team_df = df[(df['team'] == team_name) | (df['team'] == norm)]
    if not team_df.empty:
        team_df = _filter_date(team_df).sort_values('match_date')
        if not team_df.empty:
            return team_df.iloc[-1]
        
    opp_df = df[(df['opponent'] == team_name) | (df['opponent'] == norm)]
    if not opp_df.empty:
        opp_df = _filter_date(opp_df).sort_values('match_date')
        if not opp_df.empty:
            return opp_df.iloc[-1]
        
    # 2. Búsqueda flexible por alias / minúsculas
    raw_lower = str(team_name).lower().replace(' ', '').replace('cf', '').replace('fc', '')
    norm_lower = str(norm).lower().replace(' ', '').replace('cf', '').replace('fc', '')
    
    all_teams = set(df['team'].dropna().unique()).union(set(df['opponent'].dropna().unique()))
    matched_name = None
    for t in all_teams:
        t_norm = normalize_team_name(t)
        t_lower = str(t).lower().replace(' ', '').replace('cf', '').replace('fc', '')
        t_norm_lower = str(t_norm).lower().replace(' ', '').replace('cf', '').replace('fc', '')
        if raw_lower in [t_lower, t_norm_lower] or norm_lower in [t_lower, t_norm_lower]:
            matched_name = t
            break
            
    if not matched_name:
        for t in all_teams:
            t_lower = str(t).lower()
            if norm_lower in t_lower or t_lower in norm_lower:
                matched_name = t
                break

    if matched_name:
        team_df = df[(df['team'] == matched_name)]
        if not team_df.empty:
            team_df = _filter_date(team_df).sort_values('match_date')
            if not team_df.empty:
                return team_df.iloc[-1]
        opp_df = df[(df['opponent'] == matched_name)]
        if not opp_df.empty:
            opp_df = _filter_date(opp_df).sort_values('match_date')
            if not opp_df.empty:
                return opp_df.iloc[-1]

    return None

def compute_meta_features_live(df_base, open_odds_win, open_odds_draw, open_odds_loss, comp_id=0):
    df_base = df_base.fillna(0.0)
    meta = pd.DataFrame(index=df_base.index)
    cols_loss = [c for c in df_base.columns if 'loss' in c.lower()]
    cols_draw = [c for c in df_base.columns if 'draw' in c.lower()]
    cols_win = [c for c in df_base.columns if 'win' in c.lower()]
    
    meta['meta_std_loss'] = df_base[cols_loss].std(axis=1).fillna(0.0) if cols_loss else 0.0
    meta['meta_std_draw'] = df_base[cols_draw].std(axis=1).fillna(0.0) if cols_draw else 0.0
    meta['meta_std_win'] = df_base[cols_win].std(axis=1).fillna(0.0) if cols_win else 0.0
    
    mean_loss = df_base[cols_loss].mean(axis=1) if cols_loss else 0.0
    mean_draw = df_base[cols_draw].mean(axis=1) if cols_draw else 0.0
    mean_win = df_base[cols_win].mean(axis=1) if cols_win else 1.0
    
    sums = mean_loss + mean_draw + mean_win
    mean_loss = mean_loss / np.where(sums > 0, sums, 1.0)
    mean_draw = mean_draw / np.where(sums > 0, sums, 1.0)
    mean_win = mean_win / np.where(sums > 0, sums, 1.0)
    
    mean_probs = pd.DataFrame({'loss': mean_loss, 'draw': mean_draw, 'win': mean_win}).fillna(1.0 / 3.0)
    meta['meta_entropy'] = mean_probs.apply(lambda row: entropy([max(row['loss'], 1e-9), max(row['draw'], 1e-9), max(row['win'], 1e-9)]), axis=1)
    
    ow = float(open_odds_win) if pd.notna(open_odds_win) and open_odds_win > 1.01 else 2.50
    od = float(open_odds_draw) if pd.notna(open_odds_draw) and open_odds_draw > 1.01 else 3.20
    ol = float(open_odds_loss) if pd.notna(open_odds_loss) and open_odds_loss > 1.01 else 2.80
    meta['implied_open_loss'] = 1.0 / max(ol, 1.01)
    meta['implied_open_draw'] = 1.0 / max(od, 1.01)
    meta['implied_open_win'] = 1.0 / max(ow, 1.01)
    meta['meta_competition_id'] = comp_id
    meta['competition_id'] = comp_id
    
    return pd.concat([df_base, meta], axis=1).fillna(0.0)

def predict_corners_live(corners_data, home_row, away_row):
    """
    Calcula los córneres esperados con perspectiva dual (Home + Away)
    y la probabilidad calibrada de Over/Under 9.5 mediante Binomial Negativa / Poisson + Isotonic Regression.
    Incorpora imputación dinámica de estadísticas de conteo para ligas donde los córneres crudos no son tabulados.
    """
    if not corners_data or 'model_corners' not in corners_data:
        return 5.2, 4.4, 9.6, 0.51, 0.49
    
    if home_row is None:
        home_row = pd.Series({'team_elo': 1500.0})
    if away_row is None:
        away_row = pd.Series({'team_elo': 1500.0})

    feature_cols = corners_data.get('features', [])
    model_corners = corners_data['model_corners']
    c_disp = float(corners_data.get('c_dispersion', 1.15))
    calibrators = corners_data.get('calibrators', {})

    h_elo = float(home_row.get('team_elo', 1500.0) or 1500.0)
    a_elo = float(away_row.get('team_elo', 1500.0) or 1500.0)
    elo_diff = h_elo - a_elo
    h_att = float(home_row.get('team_att_rating', 1.0) or 1.0)
    a_att = float(away_row.get('team_att_rating', 1.0) or 1.0)
    h_def = float(home_row.get('team_def_rating', 1.0) or 1.0)
    a_def = float(away_row.get('team_def_rating', 1.0) or 1.0)

    h_factor = np.clip(1.0 + (elo_diff / 800.0) + (h_att - a_def) * 0.15, 0.70, 1.40)
    a_factor = np.clip(1.0 - (elo_diff / 800.0) + (a_att - h_def) * 0.15, 0.70, 1.40)

    # Imputación dinámica si las estadísticas de evento están en cero en la base de datos (ej. Liga MX)
    h_corn_val = float(home_row.get('corners_ema3', 0.0) or 0.0)
    if h_corn_val <= 0.5:
        h_corn_3 = 5.20 * h_factor
        h_corn_5 = 5.20 * h_factor
        h_shots_3 = 13.0 * h_factor
        h_shots_5 = 13.0 * h_factor
        h_cross_3 = 16.5 * h_factor
        h_cross_5 = 16.5 * h_factor
        h_poss = float(np.clip(0.52 + (elo_diff / 2000.0), 0.38, 0.65))
    else:
        h_corn_3 = h_corn_val
        h_corn_5 = float(home_row.get('corners_ema5', h_corn_val) or h_corn_val)
        h_shots_3 = float(home_row.get('shots_total_ema3', 12.5) or 12.5)
        h_shots_5 = float(home_row.get('shots_total_ema5', 12.5) or 12.5)
        h_cross_3 = float(home_row.get('crosses_ema3', 15.5) or 15.5)
        h_cross_5 = float(home_row.get('crosses_ema5', 15.5) or 15.5)
        h_poss = float(home_row.get('possession_pct_ema3', 0.50) or 0.50)

    a_corn_val = float(away_row.get('corners_ema3', 0.0) or 0.0)
    if a_corn_val <= 0.5:
        a_corn_3 = 4.40 * a_factor
        a_corn_5 = 4.40 * a_factor
        a_shots_3 = 10.5 * a_factor
        a_shots_5 = 10.5 * a_factor
        a_cross_3 = 13.5 * a_factor
        a_cross_5 = 13.5 * a_factor
        a_poss = 1.0 - h_poss
    else:
        a_corn_3 = a_corn_val
        a_corn_5 = float(away_row.get('corners_ema5', a_corn_val) or a_corn_val)
        a_shots_3 = float(away_row.get('shots_total_ema3', 10.5) or 10.5)
        a_shots_5 = float(away_row.get('shots_total_ema5', 10.5) or 10.5)
        a_cross_3 = float(away_row.get('crosses_ema3', 13.5) or 13.5)
        a_cross_5 = float(away_row.get('crosses_ema5', 13.5) or 13.5)
        a_poss = float(away_row.get('possession_pct_ema3', 0.50) or 0.50)

    # 1. Perspectiva Local (is_home = 1)
    df_h = pd.DataFrame(index=[0])
    df_h['corners_ema3'] = h_corn_3
    df_h['corners_ema5'] = h_corn_5
    df_h['shots_total_ema3'] = h_shots_3
    df_h['shots_total_ema5'] = h_shots_5
    df_h['crosses_ema3'] = h_cross_3
    df_h['crosses_ema5'] = h_cross_5
    df_h['possession_pct_ema3'] = h_poss
    df_h['possession_pct_ema5'] = h_poss
    df_h['opp_corners_ema3'] = a_corn_3
    df_h['opp_corners_ema5'] = a_corn_5
    df_h['opp_shots_total_ema3'] = a_shots_3
    df_h['opp_shots_total_ema5'] = a_shots_5
    df_h['opp_crosses_ema3'] = a_cross_3
    df_h['opp_crosses_ema5'] = a_cross_5
    df_h['opp_possession_pct_ema3'] = a_poss
    df_h['opp_possession_pct_ema5'] = a_poss
    df_h['team_att_rating'] = h_att
    df_h['team_def_rating'] = h_def
    df_h['opp_att_rating'] = a_att
    df_h['opp_def_rating'] = a_def
    df_h['elo_diff'] = elo_diff
    df_h['relative_attack_strength'] = float(home_row.get('relative_attack_strength', 0.0) or 0.0)
    df_h['xg_momentum_macd'] = float(home_row.get('xg_momentum_macd', 0.0) or 0.0)
    df_h['is_home'] = 1

    # 2. Perspectiva Visitante (is_home = 0)
    df_a = pd.DataFrame(index=[0])
    df_a['corners_ema3'] = a_corn_3
    df_a['corners_ema5'] = a_corn_5
    df_a['shots_total_ema3'] = a_shots_3
    df_a['shots_total_ema5'] = a_shots_5
    df_a['crosses_ema3'] = a_cross_3
    df_a['crosses_ema5'] = a_cross_5
    df_a['possession_pct_ema3'] = a_poss
    df_a['possession_pct_ema5'] = a_poss
    df_a['opp_corners_ema3'] = h_corn_3
    df_a['opp_corners_ema5'] = h_corn_5
    df_a['opp_shots_total_ema3'] = h_shots_3
    df_a['opp_shots_total_ema5'] = h_shots_5
    df_a['opp_crosses_ema3'] = h_cross_3
    df_a['opp_crosses_ema5'] = h_cross_5
    df_a['opp_possession_pct_ema3'] = h_poss
    df_a['opp_possession_pct_ema5'] = h_poss
    df_a['team_att_rating'] = a_att
    df_a['team_def_rating'] = a_def
    df_a['opp_att_rating'] = h_att
    df_a['opp_def_rating'] = h_def
    df_a['elo_diff'] = -elo_diff
    df_a['relative_attack_strength'] = -float(home_row.get('relative_attack_strength', 0.0) or 0.0)
    df_a['xg_momentum_macd'] = -float(home_row.get('xg_momentum_macd', 0.0) or 0.0)
    df_a['is_home'] = 0

    for f in feature_cols:
        if f not in df_h.columns: df_h[f] = 0.0
        if f not in df_a.columns: df_a[f] = 0.0

    try:
        exp_h = float(model_corners.predict(df_h[feature_cols].fillna(0.0))[0])
        exp_a = float(model_corners.predict(df_a[feature_cols].fillna(0.0))[0])
    except Exception:
        exp_h, exp_a = 5.20 * h_factor, 4.40 * a_factor

    exp_h = float(np.clip(exp_h, 2.5, 9.5))
    exp_a = float(np.clip(exp_a, 2.0, 8.5))
    exp_tot = exp_h + exp_a

    # Cálculo estocástico de Over 9.5 y Under 9.5
    k = 9
    if c_disp <= 1.02:
        prob_over95_raw = float(1.0 - poisson.cdf(k, exp_tot))
    else:
        p_nb = 1.0 / c_disp
        n_nb = max(exp_tot, 1e-4) * p_nb / (1.0 - p_nb)
        n_nb = max(n_nb, 1e-4)
        prob_over95_raw = float(1.0 - nbinom.cdf(k, n_nb, p_nb))

    prob_over95_raw = float(np.clip(prob_over95_raw, 0.01, 0.99))

    calib = calibrators.get(9.5)
    if calib is not None:
        try:
            prob_over95_calib = float(calib.predict(np.array([prob_over95_raw]))[0])
            prob_over95 = float(np.clip(prob_over95_calib, 0.01, 0.99))
        except Exception:
            prob_over95 = prob_over95_raw
    else:
        prob_over95 = prob_over95_raw

    prob_under95 = float(1.0 - prob_over95)
    return exp_h, exp_a, exp_tot, prob_over95, prob_under95

def build_live_match_features(
    df, home_team, away_team, comp, odds_1, odds_X, odds_2,
    open_odds_win=None, open_odds_draw=None, open_odds_loss=None,
    as_of_date=None
):
    home_row = get_latest_team_row(df, home_team, as_of_date=as_of_date)
    away_row = get_latest_team_row(df, away_team, as_of_date=as_of_date)
    
    if home_row is None:
        home_row = pd.Series({'team_elo': 1500.0, 'team_squad_value': 0.0})
    if away_row is None:
        away_row = pd.Series({'team_elo': 1500.0, 'team_squad_value': 0.0})

    input_data = {}
    
    for col in home_row.index:
        if isinstance(home_row[col], (int, float, np.number)) and not pd.isna(home_row[col]):
            input_data[col] = float(home_row[col])

    input_data['is_home'] = 1
    input_data['team'] = home_team
    input_data['opponent'] = away_team
    input_data['competition'] = comp
    
    comp_categories = pd.factorize(df['competition'])[1]
    input_data['competition_id'] = comp_categories.get_loc(comp) if comp in comp_categories else 0

    # Mapear atributos del rival actual (away_row) en lugar de arrastrar el rival previo de home_row
    input_data['team_elo'] = float(home_row.get('team_elo', 1500.0) or 1500.0)
    input_data['opp_elo'] = float(away_row.get('team_elo', 1500.0) or 1500.0)
    input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']

    input_data['team_squad_value'] = float(home_row.get('team_squad_value', 0.0) or 0.0)
    input_data['opp_squad_value'] = float(away_row.get('team_squad_value', 0.0) or 0.0)
    input_data['squad_value_diff'] = input_data['team_squad_value'] - input_data['opp_squad_value']

    input_data['team_att_rating'] = float(home_row.get('team_att_rating', 1.0) or 1.0)
    input_data['team_def_rating'] = float(home_row.get('team_def_rating', 1.0) or 1.0)
    input_data['opp_att_rating'] = float(away_row.get('team_att_rating', 1.0) or 1.0)
    input_data['opp_def_rating'] = float(away_row.get('team_def_rating', 1.0) or 1.0)

    input_data['rest_days'] = float(home_row.get('rest_days', 4.0) or 4.0)
    input_data['opp_rest_days'] = float(away_row.get('rest_days', 4.0) or 4.0)
    input_data['rest_diff'] = input_data['rest_days'] - input_data['opp_rest_days']

    input_data['win_streak_3'] = float(home_row.get('win_streak_3', 0.0) or 0.0)
    input_data['loss_streak_3'] = float(home_row.get('loss_streak_3', 0.0) or 0.0)
    input_data['xg_momentum_macd'] = float(home_row.get('xg_momentum_macd', 0.0) or 0.0)
    input_data['opp_win_streak_3'] = float(away_row.get('win_streak_3', 0.0) or 0.0)
    input_data['opp_loss_streak_3'] = float(away_row.get('loss_streak_3', 0.0) or 0.0)
    input_data['opp_xg_momentum_macd'] = float(away_row.get('xg_momentum_macd', 0.0) or 0.0)

    input_data['fatigue_index'] = float(home_row.get('fatigue_index', 0.0) or 0.0)
    input_data['opp_fatigue_index'] = float(away_row.get('fatigue_index', 0.0) or 0.0)
    input_data['fatigue_diff'] = input_data['fatigue_index'] - input_data['opp_fatigue_index']

    input_data['xg_volatility_5'] = float(home_row.get('xg_volatility_5', 0.5) or 0.5)
    input_data['opp_xg_volatility_5'] = float(away_row.get('xg_volatility_5', 0.5) or 0.5)
    input_data['volatility_diff'] = input_data['xg_volatility_5'] - input_data['opp_xg_volatility_5']

    # Mapear todas las EMAs disponibles del rival actual (away_row -> opp_*_ema*)
    for col in away_row.index:
        if (col.endswith('_ema3') or col.endswith('_ema5')) and not col.startswith('opp_'):
            val = away_row[col]
            if isinstance(val, (int, float, np.number)) and not pd.isna(val):
                input_data[f'opp_{col}'] = float(val)

    input_data['relative_attack_strength'] = float(
        input_data.get('xg_created_ema5', 1.2) - input_data.get('opp_xg_conceded_ema5', 1.2)
    )

    # Feature Engineering de Contexto (Paridad 1:1 con train_context.py)
    home_adv = 100.0 if input_data.get('is_home', 1) == 1 else -100.0
    input_data['elo_prob_win'] = float(1.0 / (1.0 + 10.0 ** (-(input_data['elo_diff'] + home_adv) / 400.0)))
    input_data['log_squad_ratio'] = float(
        np.log1p(max(input_data['team_squad_value'], 0.0)) - np.log1p(max(input_data['opp_squad_value'], 0.0))
    )
    rest_team = max(input_data['rest_days'], 0.0)
    rest_opp = max(input_data['opp_rest_days'], 0.0)
    input_data['recovery_index_team'] = float(1.0 - np.exp(-rest_team / 3.0))
    input_data['recovery_index_opp'] = float(1.0 - np.exp(-rest_opp / 3.0))
    input_data['recovery_diff'] = float(input_data['recovery_index_team'] - input_data['recovery_index_opp'])
    vol = max(input_data['xg_volatility_5'], 0.05)
    input_data['elo_sharpe_ratio'] = float(input_data['elo_diff'] / (vol + 1e-4))
    input_data['elo_momentum_interaction'] = float((input_data['elo_prob_win'] - 0.5) * input_data['xg_momentum_macd'])

    # Feature Engineering de Empates (Paridad 1:1 con train_draws.py)
    lh_draw = max(float(input_data.get('xg_created_ema5', 1.2)), 0.05)
    la_draw = max(float(input_data.get('opp_xg_created_ema5', 1.2)), 0.05)
    input_data['poisson_draw_prob'] = float(sum(poisson.pmf(k, lh_draw) * poisson.pmf(k, la_draw) for k in range(10)))
    input_data['expected_match_xg'] = float(lh_draw + la_draw)
    input_data['abs_elo_diff'] = float(abs(input_data['elo_diff']))

    comp_draw_rate = 0.26
    if 'competition' in df.columns and 'outcome' in df.columns:
        sub_comp = df[df['competition'] == comp]
        if as_of_date is not None and 'match_date' in sub_comp.columns:
            sub_comp = sub_comp[sub_comp['match_date'] < as_of_date]
        if len(sub_comp) > 0:
            draws_cnt = float((sub_comp['outcome'] == 0).sum())
            comp_draw_rate = float((draws_cnt + 10.0 * 0.26) / (len(sub_comp) + 10.0))
    input_data['competition_draw_rate'] = comp_draw_rate

    # Feature Engineering de Mercado y GBM (Paridad 1:1 con train_market.py y train_gbm_model.py)
    open_1 = float(open_odds_win if open_odds_win is not None else odds_1)
    open_X = float(open_odds_draw if open_odds_draw is not None else odds_X)
    open_2 = float(open_odds_loss if open_odds_loss is not None else odds_2)

    input_data['open_odds_win'] = open_1
    input_data['open_odds_draw'] = open_X
    input_data['open_odds_loss'] = open_2
    input_data['odds_win'] = float(odds_1)
    input_data['odds_draw'] = float(odds_X)
    input_data['odds_loss'] = float(odds_2)

    input_data['implied_prob_draw'] = float(1.0 / max(odds_X, 1.01))
    input_data['open_implied_prob_draw'] = float(1.0 / max(open_X, 1.01))
    input_data['odds_match_balance'] = float(abs((1.0 / max(odds_1, 1.01)) - (1.0 / max(odds_2, 1.01))))

    sum_implied_o = (1.0 / max(open_1, 1.01)) + (1.0 / max(open_X, 1.01)) + (1.0 / max(open_2, 1.01))
    input_data['vig_open'] = float(sum_implied_o - 1.0)
    input_data['open_prob_win'] = float((1.0 / max(open_1, 1.01)) / sum_implied_o)
    input_data['open_prob_draw'] = float((1.0 / max(open_X, 1.01)) / sum_implied_o)
    input_data['open_prob_loss'] = float((1.0 / max(open_2, 1.01)) / sum_implied_o)

    input_data['prob_win_implied'] = input_data['open_prob_win']
    input_data['prob_draw_implied'] = input_data['open_prob_draw']
    input_data['prob_loss_implied'] = input_data['open_prob_loss']

    sw, sd, sl, sz = compute_shin_probs([open_1], [open_X], [open_2])
    input_data['shin_prob_win'] = float(sw[0])
    input_data['shin_prob_draw'] = float(sd[0])
    input_data['shin_prob_loss'] = float(sl[0])
    input_data['shin_z_param'] = float(sz.item() if hasattr(sz, 'item') else sz)

    pw_pow, pd_pow, pl_pow = compute_power_probs([open_1], [open_X], [open_2])
    input_data['power_prob_win'] = float(pw_pow[0])
    input_data['power_prob_draw'] = float(pd_pow[0])
    input_data['power_prob_loss'] = float(pl_pow[0])

    input_data['shin_base_prob_win'] = float(sw[0])
    input_data['shin_base_prob_draw'] = float(sd[0])
    input_data['shin_base_prob_loss'] = float(sl[0])
    input_data['shin_insider_open'] = float(input_data['shin_z_param'])

    p_open = np.array([input_data['open_prob_win'], input_data['open_prob_draw'], input_data['open_prob_loss']])
    input_data['open_market_entropy'] = float(-np.sum(p_open * np.log(np.maximum(p_open, 1e-9))))
    input_data['market_entropy'] = input_data['open_market_entropy']
    sorted_p = np.sort(p_open)
    input_data['fav_prob_gap'] = float(sorted_p[2] - sorted_p[1])

    # Si las cuotas en vivo difieren de la apertura, medir deriva en vivo; si no, usar deriva histórica (t-1) del equipo
    has_live_movement = (abs(odds_1 - open_1) > 1e-4 or abs(odds_X - open_X) > 1e-4 or abs(odds_2 - open_2) > 1e-4)
    if has_live_movement:
        sum_implied_c = (1.0 / max(odds_1, 1.01)) + (1.0 / max(odds_X, 1.01)) + (1.0 / max(odds_2, 1.01))
        close_prob_w = (1.0 / max(odds_1, 1.01)) / sum_implied_c
        close_prob_d = (1.0 / max(odds_X, 1.01)) / sum_implied_c
        close_prob_l = (1.0 / max(odds_2, 1.01)) / sum_implied_c
        input_data['vig_close'] = float(sum_implied_c - 1.0)
        input_data['overround_drop'] = float(input_data['vig_open'] - input_data['vig_close'])
        input_data['steam_win'] = float(close_prob_w - input_data['open_prob_win'])
        input_data['steam_draw'] = float(close_prob_d - input_data['open_prob_draw'])
        input_data['steam_loss'] = float(close_prob_l - input_data['open_prob_loss'])
        input_data['log_steam_win'] = float(np.log(max(close_prob_w, 1e-4) / max(input_data['open_prob_win'], 1e-4)))
        input_data['log_steam_draw'] = float(np.log(max(close_prob_d, 1e-4) / max(input_data['open_prob_draw'], 1e-4)))
        input_data['log_steam_loss'] = float(np.log(max(close_prob_l, 1e-4) / max(input_data['open_prob_loss'], 1e-4)))
        input_data['rel_steam_win'] = float(input_data['steam_win'] / max(input_data['open_prob_win'], 1e-4))
        input_data['rel_steam_draw'] = float(input_data['steam_draw'] / max(input_data['open_prob_draw'], 1e-4))
        input_data['rel_steam_loss'] = float(input_data['steam_loss'] / max(input_data['open_prob_loss'], 1e-4))
        p_close = np.array([close_prob_w, close_prob_d, close_prob_l])
        close_entropy = float(-np.sum(p_close * np.log(np.maximum(p_close, 1e-9))))
        input_data['entropy_change'] = float(close_entropy - input_data['open_market_entropy'])
    else:
        # Deriva histórica (t-1) desde home_row (sin contaminar ni sobreescribir con 0.0 las columnas gbm_* precalculadas)
        prev_ow = float(home_row.get('open_odds_win', open_1) or open_1)
        prev_od = float(home_row.get('open_odds_draw', open_X) or open_X)
        prev_ol = float(home_row.get('open_odds_loss', open_2) or open_2)
        prev_cw = float(home_row.get('closing_odds_win', prev_ow) or prev_ow)
        prev_cd = float(home_row.get('closing_odds_draw', prev_od) or prev_od)
        prev_cl = float(home_row.get('closing_odds_loss', prev_ol) or prev_ol)
        s_o = (1.0 / max(prev_ow, 1.01)) + (1.0 / max(prev_od, 1.01)) + (1.0 / max(prev_ol, 1.01))
        s_c = (1.0 / max(prev_cw, 1.01)) + (1.0 / max(prev_cd, 1.01)) + (1.0 / max(prev_cl, 1.01))
        po_w, po_d, po_l = (1.0 / max(prev_ow, 1.01)) / s_o, (1.0 / max(prev_od, 1.01)) / s_o, (1.0 / max(prev_ol, 1.01)) / s_o
        pc_w, pc_d, pc_l = (1.0 / max(prev_cw, 1.01)) / s_c, (1.0 / max(prev_cd, 1.01)) / s_c, (1.0 / max(prev_cl, 1.01)) / s_c
        input_data['vig_close'] = float(s_c - 1.0)
        input_data['overround_drop'] = float((s_o - 1.0) - (s_c - 1.0))
        input_data['steam_win'] = float(pc_w - po_w)
        input_data['steam_draw'] = float(pc_d - po_d)
        input_data['steam_loss'] = float(pc_l - po_l)
        input_data['log_steam_win'] = float(np.log(max(pc_w, 1e-4) / max(po_w, 1e-4)))
        input_data['log_steam_draw'] = float(np.log(max(pc_d, 1e-4) / max(po_d, 1e-4)))
        input_data['log_steam_loss'] = float(np.log(max(pc_l, 1e-4) / max(po_l, 1e-4)))
        input_data['rel_steam_win'] = float(input_data['steam_win'] / max(po_w, 1e-4))
        input_data['rel_steam_draw'] = float(input_data['steam_draw'] / max(po_d, 1e-4))
        input_data['rel_steam_loss'] = float(input_data['steam_loss'] / max(po_l, 1e-4))
        po_arr = np.array([po_w, po_d, po_l])
        pc_arr = np.array([pc_w, pc_d, pc_l])
        input_data['entropy_change'] = float(
            (-np.sum(pc_arr * np.log(np.maximum(pc_arr, 1e-9)))) - (-np.sum(po_arr * np.log(np.maximum(po_arr, 1e-9))))
        )

    # Preservar variables GBM históricas desde home_row si existen; usar defaults sólo si no están presentes
    input_data.setdefault('shin_insider_close', float(input_data['shin_z_param']))
    input_data.setdefault('shin_insider_drift', 0.0)
    for out_k in ['win', 'draw', 'loss']:
        input_data.setdefault(f'gbm_mu_{out_k}', float(input_data[f'log_steam_{out_k}']))
        input_data.setdefault(f'gbm_sigma_{out_k}', 0.05)
        input_data.setdefault(f'gbm_mean_{out_k}', float(input_data[f'log_steam_{out_k}']))
        input_data.setdefault(f'gbm_ewma_mu_{out_k}', float(input_data[f'log_steam_{out_k}']))
        input_data.setdefault(f'gbm_ewma_std_{out_k}', 0.05)
        input_data.setdefault(f'gbm_z_{out_k}', 0.0)

    return input_data

def predict_match(
    home_team,
    away_team,
    comp,
    odds_1,
    odds_X,
    odds_2,
    open_odds_win=None,
    open_odds_draw=None,
    open_odds_loss=None,
    extra_odds=None,
    bankroll=1000.0,
    injuries_home=0,
    injuries_away=0,
    df=None,
    models_dict=None,
    as_of_date=None
):
    if df is None:
        df = load_data()
        if df is None: return None

    if models_dict is None:
        models_dict = load_all_models()
        if models_dict is None: return None

    extra_odds = extra_odds or {}
    
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
    league_alpha_low = get_param_by_comp(alpha_div_low_dict, comp, 0.35)
    league_alpha_med = get_param_by_comp(alpha_div_med_dict, comp, 0.25)
    league_alpha_high = get_param_by_comp(alpha_div_high_dict, comp, 0.15)

    input_data = build_live_match_features(
        df, home_team, away_team, comp, odds_1, odds_X, odds_2,
        open_odds_win=open_odds_win, open_odds_draw=open_odds_draw, open_odds_loss=open_odds_loss,
        as_of_date=as_of_date
    )
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

    all_req_features = set()
    for m_key in ['context', 'nn', 'draws', 'gbm', 'market', 'fundamental', 'final']:
        if m_key in models_dict and isinstance(models_dict[m_key], dict) and 'features' in models_dict[m_key]:
            all_req_features.update(models_dict[m_key]['features'])
    for f in all_req_features:
        if f not in input_data:
            input_data[f] = 0.0

    df_input_full = pd.DataFrame([input_data])

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

    w_q, d_q, l_q, mu_h_np, mu_a_np = predict_from_model(quant_dict['model'], quant_dict['team_mapping'], df_input_full)
    q_win, q_draw, q_loss = w_q[0], d_q[0], l_q[0]
    xg_s, xg_c = float(mu_h_np[0]), float(mu_a_np[0])

    lineup_fetcher = LineupImpactFetcher()
    xg_s, xg_c = lineup_fetcher.calculate_lineup_xg_impact(xg_s, xg_c, missing_home_starters=injuries_home, missing_away_starters=injuries_away)

    df_ctx = df_input_full[context_data['features']].copy()
    if 'competition_id' in df_ctx.columns: df_ctx['competition_id'] = df_ctx['competition_id'].astype('category')
    ctx_probs = context_data['model'].predict_proba(df_ctx)[0]
    
    nn_probs = nn_data['model'].predict_proba(df_input_full[nn_data['features']])[0]
    draws_prob = draws_data['model'].predict_proba(df_input_full[draws_data['features']])[0][1]
    mkt_probs = market_data['model'].predict_proba(df_input_full[market_data['features']])[0]
    gbm_probs = gbm_data['model'].predict_proba(df_input_full[gbm_data['features']])[0]

    df_fund_input = pd.DataFrame({
        'predicted_xg_scored_quant': [xg_s], 'predicted_xg_conceded_quant': [xg_c],
        'quant_win_prob': [q_win], 'quant_draw_prob': [q_draw], 'quant_loss_prob': [q_loss],
        'prob_loss_ctx': [ctx_probs[0]], 'prob_draw_ctx': [ctx_probs[1]], 'prob_win_ctx': [ctx_probs[2]],
        'prob_loss_nn': [nn_probs[0]], 'prob_draw_nn': [nn_probs[1]], 'prob_win_nn': [nn_probs[2]],
        'prob_is_draw': [draws_prob]
    })[fund_data['features']]
    fund_probs = fund_data['model'].predict_proba(df_fund_input)[0]

    df_fund_out = pd.DataFrame({'fund_prob_loss': [fund_probs[0]], 'fund_prob_draw': [fund_probs[1]], 'fund_prob_win': [fund_probs[2]]})
    df_mkt = pd.DataFrame({'prob_loss_mkt': [mkt_probs[0]], 'prob_draw_mkt': [mkt_probs[1]], 'prob_win_mkt': [mkt_probs[2]],
                           'prob_loss_gbm': [gbm_probs[0]], 'prob_draw_gbm': [gbm_probs[1]], 'prob_win_gbm': [gbm_probs[2]]})
    
    df_final = compute_meta_features_live(
        pd.concat([df_fund_out, df_mkt], axis=1),
        input_data['open_odds_win'], input_data['open_odds_draw'], input_data['open_odds_loss'],
        input_data['competition_id']
    )
    
    for feat in final_data['features']:
        if feat not in df_final.columns:
            df_final[feat] = 0.0

    df_final = df_final[final_data['features']]
    if 'competition_id' in df_final.columns: df_final['competition_id'] = df_final['competition_id'].astype('category')

    final_probs = final_data['model'].predict_proba(df_final)[0]
    final_probs /= np.sum(final_probs)
    prob_loss, prob_draw, prob_win = final_probs[0], final_probs[1], final_probs[2]

    all_win_probs = [q_win, ctx_probs[2], nn_probs[2], mkt_probs[2], gbm_probs[2], fund_probs[2], prob_win]
    all_draw_probs = [q_draw, ctx_probs[1], nn_probs[1], mkt_probs[1], gbm_probs[1], fund_probs[1], prob_draw]
    all_loss_probs = [q_loss, ctx_probs[0], nn_probs[0], mkt_probs[0], gbm_probs[0], fund_probs[0], prob_loss]

    _, se_win, ci_win_low, ci_win_high, pen_win = calculate_bayesian_uncertainty(all_win_probs)
    _, se_draw, ci_draw_low, ci_draw_high, pen_draw = calculate_bayesian_uncertainty(all_draw_probs)
    _, se_loss, ci_loss_low, ci_loss_high, pen_loss = calculate_bayesian_uncertainty(all_loss_probs)

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

    corners_data = models_dict.get('corners')
    home_row_corners = get_latest_team_row(df, home_team, as_of_date=as_of_date)
    away_row_corners = get_latest_team_row(df, away_team, as_of_date=as_of_date)
    exp_corners_home, exp_corners_away, exp_corners, prob_over95_corners, prob_under95_corners = predict_corners_live(
        corners_data, home_row_corners, away_row_corners
    )

    cards_data = models_dict.get('cards')
    exp_cards = 4.2
    prob_over45_cards = float(1.0 - sum(poisson.pmf(k, exp_cards) for k in range(5)))
    if isinstance(cards_data, dict) and 'home' in cards_data and 'away' in cards_data:
        try:
            df_c_h = build_perspective_features(df_input_full, perspective='home')
            df_c_a = build_perspective_features(df_input_full, perspective='away')
            req_h = list(getattr(cards_data['home'], 'feature_names_in_', df_c_h.columns))
            req_a = list(getattr(cards_data['away'], 'feature_names_in_', df_c_a.columns))
            for col in req_h:
                if col not in df_c_h.columns: df_c_h[col] = 0.0
            for col in req_a:
                if col not in df_c_a.columns: df_c_a[col] = 0.0
            lam_c_h = float(np.clip(cards_data['home'].predict(df_c_h[req_h])[0], 0.3, 6.5))
            lam_c_a = float(np.clip(cards_data['away'].predict(df_c_a[req_a])[0], 0.3, 6.5))
            exp_cards = float(lam_c_h + lam_c_a)
            c_probs = calc_cards_probabilities_vectorized(np.array([lam_c_h]), np.array([lam_c_a]))
            prob_over45_cards = float(c_probs['prob_over_4_5_cards'][0])
        except Exception:
            pass
    elif cards_data and hasattr(cards_data, 'predict'):
        try:
            exp_cards = float(cards_data.predict(df_input_full)[0])
            prob_over45_cards = float(1.0 - sum(poisson.pmf(k, exp_cards) for k in range(5)))
        except Exception:
            pass

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

    combined_odds_1X = 1.0 / (impl_win + impl_draw)
    net_combined_1X = 1.0 + (combined_odds_1X - 1.0) * (1.0 - TAX_RETENTION_RATE)
    blend_1X = blend_win + blend_draw
    ev_1X = (blend_1X * net_combined_1X) - 1.0 - EXPECTED_CLV_DROP

    combined_odds_X2 = 1.0 / (impl_draw + impl_loss)
    net_combined_X2 = 1.0 + (combined_odds_X2 - 1.0) * (1.0 - TAX_RETENTION_RATE)
    blend_X2 = blend_draw + blend_loss
    ev_X2 = (blend_X2 * net_combined_X2) - 1.0 - EXPECTED_CLV_DROP

    # Mercados secundarios: sólo elegibles para asignación de capital si se proveen cuotas reales en extra_odds
    xg_tot = float(xg_s + xg_c)
    p_ov25_base = float(1.0 - poisson.cdf(2, xg_tot))
    default_odd_ov25 = float(np.clip(1.0 / (max(p_ov25_base, 0.15) * 1.03), 1.30, 3.50))
    default_odd_un25 = float(np.clip(1.0 / (max(1.0 - p_ov25_base, 0.15) * 1.03), 1.30, 3.50))

    odd_ou25 = float(extra_odds.get('ou25_over', default_odd_ov25))
    odd_ou25_under = float(extra_odds.get('ou25_under', default_odd_un25))
    ev_ou25 = ((prob_over25_goals * (1.0 + (odd_ou25 - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'ou25_over' in extra_odds else -1.0
    ev_ou25_under = ((prob_under25_goals * (1.0 + (odd_ou25_under - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'ou25_under' in extra_odds else -1.0

    default_odd_btts = float(np.clip(1.0 / (max(prob_btts_yes, 0.20) * 1.03), 1.40, 2.80))
    odd_btts = float(extra_odds.get('btts_yes', default_odd_btts))
    ev_btts = ((prob_btts_yes * (1.0 + (odd_btts - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'btts_yes' in extra_odds else -1.0

    default_odd_corn_ov = float(np.clip(1.0 / (max(prob_over95_corners, 0.20) * 1.03), 1.40, 2.80))
    default_odd_corn_un = float(np.clip(1.0 / (max(prob_under95_corners, 0.20) * 1.03), 1.40, 2.80))
    odd_corners = float(extra_odds.get('corners_over95', default_odd_corn_ov))
    odd_corners_under = float(extra_odds.get('corners_under95', default_odd_corn_un))
    ev_corners = ((prob_over95_corners * (1.0 + (odd_corners - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'corners_over95' in extra_odds else -1.0
    ev_corners_under = ((prob_under95_corners * (1.0 + (odd_corners_under - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'corners_under95' in extra_odds else -1.0

    default_odd_card = float(np.clip(1.0 / (max(prob_over45_cards, 0.20) * 1.03), 1.40, 2.80))
    odd_cards = float(extra_odds.get('cards_over45', default_odd_card))
    ev_cards = ((prob_over45_cards * (1.0 + (odd_cards - 1.0) * (1 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP) if 'cards_over45' in extra_odds else -1.0

    max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

    corr_matrix = compute_poisson_market_correlations(poisson_matrix)
    ev_vector = [ev_win, ev_draw, ev_loss, ev_1X, ev_X2, ev_ou25, ev_ou25_under, ev_btts, ev_corners, ev_corners_under, ev_cards]
    odds_vector = [odds_1, odds_X, odds_2, combined_odds_1X, combined_odds_X2, odd_ou25, odd_ou25_under, odd_btts, odd_corners, odd_corners_under, odd_cards]
    prob_vector = [blend_win, blend_draw, blend_loss, blend_1X, blend_X2, prob_over25_goals, prob_under25_goals, prob_btts_yes, prob_over95_corners, prob_under95_corners, prob_over45_cards]

    avg_penalty = np.mean([pen_win, pen_draw, pen_loss])
    multi_stakes, multi_pcts = solve_multi_asset_kelly(
        ev_vector, odds_vector, prob_vector, corr_matrix, bankroll, MAX_STAKE_PCT,
        penalty=avg_penalty, max_liquidity=max_liquidity, league_kelly=league_kelly
    )

    s_win, s_draw, s_loss, s_1X, s_X2, s_ou25, s_un25, s_btts, s_corn, s_corn_un, s_cards = multi_stakes
    p_win_stk, p_draw_stk, p_loss_stk, p_1X_stk, p_X2_stk, p_ou25_stk, p_un25_stk, p_btts_stk, p_corn_stk, p_corn_un_stk, p_cards_stk = multi_pcts

    eff_1 = calculate_market_slippage(odds_1, s_win, max_liquidity)
    eff_X = calculate_market_slippage(odds_X, s_draw, max_liquidity)
    eff_2 = calculate_market_slippage(odds_2, s_loss, max_liquidity)
    eff_1X = calculate_market_slippage(combined_odds_1X, s_1X, max_liquidity)
    eff_X2 = calculate_market_slippage(combined_odds_X2, s_X2, max_liquidity)

    home_official = get_official_team_name(home_team)
    away_official = get_official_team_name(away_team)

    all_bets = [
        (f"1 ({home_official})", ev_win, s_win, p_win_stk, odds_1, eff_1, blend_win),
        ("X (Empate)", ev_draw, s_draw, p_draw_stk, odds_X, eff_X, blend_draw),
        (f"2 ({away_official})", ev_loss, s_loss, p_loss_stk, odds_2, eff_2, blend_loss),
        ("Doble Oportunidad 1X", ev_1X, s_1X, p_1X_stk, combined_odds_1X, eff_1X, blend_1X),
        ("Doble Oportunidad X2", ev_X2, s_X2, p_X2_stk, combined_odds_X2, eff_X2, blend_X2),
        ("Over 2.5 Goles", ev_ou25, s_ou25, p_ou25_stk, odd_ou25, odd_ou25, prob_over25_goals),
        ("Under 2.5 Goles", ev_ou25_under, s_un25, p_un25_stk, odd_ou25_under, odd_ou25_under, prob_under25_goals),
        ("BTTS (Ambos Anotan)", ev_btts, s_btts, p_btts_stk, odd_btts, odd_btts, prob_btts_yes),
        ("Over 9.5 Córneres", ev_corners, s_corn, p_corn_stk, odd_corners, odd_corners, prob_over95_corners),
        ("Under 9.5 Córneres", ev_corners_under, s_corn_un, p_corn_un_stk, odd_corners_under, odd_corners_under, prob_under95_corners),
        ("Over 4.5 Tarjetas", ev_cards, s_cards, p_cards_stk, odd_cards, odd_cards, prob_over45_cards)
    ]
    
    # Criterio de Selección de Apostado Cuantitativo: Ranking uniforme por EV neto esperado
    sorted_bets = sorted(all_bets, key=lambda x: x[1], reverse=True)
    best_bet = sorted_bets[0]

    return {
        'home_team': home_team, 'away_team': away_team, 'competition': comp,
        'home_official': home_official, 'away_official': away_official,
        'xg_scored': xg_s, 'xg_conceded': xg_c,
        'exp_corners': exp_corners, 'exp_corners_home': exp_corners_home, 'exp_corners_away': exp_corners_away,
        'exp_cards': exp_cards,
        'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
        'ci_win_low': ci_win_low, 'ci_win_high': ci_win_high, 'se_win': se_win,
        'ci_draw_low': ci_draw_low, 'ci_draw_high': ci_draw_high, 'se_draw': se_draw,
        'ci_loss_low': ci_loss_low, 'ci_loss_high': ci_loss_high, 'se_loss': se_loss,
        'blend_win': blend_win, 'blend_draw': blend_draw, 'blend_loss': blend_loss,
        'prob_over25_goals': prob_over25_goals, 'prob_under25_goals': prob_under25_goals,
        'prob_btts_yes': prob_btts_yes, 'prob_btts_no': prob_btts_no,
        'prob_over95_corners': prob_over95_corners, 'prob_under95_corners': prob_under95_corners,
        'prob_over45_cards': prob_over45_cards,
        'ev_win': ev_win, 'ev_draw': ev_draw, 'ev_loss': ev_loss,
        'ev_1X': ev_1X, 'ev_X2': ev_X2,
        'ev_ou25': ev_ou25, 'ev_ou25_under': ev_ou25_under,
        'ev_btts': ev_btts,
        'odd_ou25': odd_ou25, 'odd_ou25_under': odd_ou25_under,
        'odd_btts': odd_btts, 'odd_corners': odd_corners, 'odd_corners_under': odd_corners_under, 'odd_cards': odd_cards,
        'ev_corners': ev_corners, 'ev_corners_under': ev_corners_under,
        'ev_cards': ev_cards,
        'stake_win': s_win, 'stake_draw': s_draw, 'stake_loss': s_loss,
        'stake_1X': s_1X, 'stake_X2': s_X2,
        'stake_ou25': s_ou25, 'stake_ou25_under': s_un25,
        'stake_btts': s_btts,
        'stake_corners': s_corn, 'stake_corners_under': s_corn_un,
        'stake_cards': s_cards,
        'pct_win': p_win_stk, 'pct_draw': p_draw_stk, 'pct_loss': p_loss_stk,
        'pct_1X': p_1X_stk, 'pct_X2': p_X2_stk,
        'pct_ou25': p_ou25_stk, 'pct_ou25_under': p_un25_stk,
        'pct_btts': p_btts_stk,
        'pct_corners': p_corn_stk, 'pct_corners_under': p_corn_un_stk,
        'pct_cards': p_cards_stk,
        'eff_1': eff_1, 'eff_X': eff_X, 'eff_2': eff_2, 'eff_1X': eff_1X, 'eff_X2': eff_X2,
        'league_ev_thresh': league_ev_thresh, 'league_kelly': league_kelly,
        'bayesian_penalty': avg_penalty, 'all_bets': all_bets, 'best_bet': best_bet
    }

def format_match_datetime(date_str):
    if not date_str: return ""
    try:
        if 'T' in str(date_str):
            dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
            return dt.strftime('%d/%m/%Y a las %H:%M UTC')
        return str(date_str)
    except Exception:
        return str(date_str)

def main():
    console.print(Panel.fit(
        "[bold bright_cyan]⚡ PROPHETIA2 QUANT[/bold bright_cyan] • [bold bright_yellow]INSTITUTIONAL VALUE ENGINE 10/10[/bold bright_yellow]\n"
        "[dim white]Live Pinnacle Odds Feed • 95% Bayesian Credible Intervals • Multi-Asset Kelly Portfolio[/dim white]",
        border_style="bright_blue"
    ))
    
    models_dict = load_all_models()
    if models_dict is None: return

    df = load_data()
    if df is None: return

    competitions = df['competition'].dropna().unique().tolist()
    comp_choices = [
        Choice(title=get_formatted_comp_name(c), value=c)
        for c in sorted(competitions)
    ]

    comp = questionary.select(
        "🏆 Selecciona la Competición / Liga:",
        choices=comp_choices,
        style=CUSTOM_QUESTIONARY_STYLE
    ).ask()

    if not comp: return

    # --- INGESTA OBLIGATORIA DE CUOTAS EN VIVO (Pinnacle / The-Odds-API / Football-Data / ESPN API) ---
    console.print(f"[cyan]📥 Descargando partidos y cuotas en tiempo real para {get_formatted_comp_name(comp)}...[/cyan]")
    live_fetcher = LiveOddsFeedFetcher()
    live_matches = live_fetcher.get_live_league_fixtures(comp)

    selected_match = None
    home_team, away_team = None, None
    odds_1, odds_X, odds_2 = None, None, None
    open_odds_win, open_odds_draw, open_odds_loss = None, None, None
    selected_match_date = ""
    extra_odds = {}
    
    if live_matches:
        match_choices = []
        for i, m in enumerate(live_matches):
            dt_str = format_match_datetime(m.get('date', ''))
            date_part = f"📅 {dt_str} | " if dt_str else ""
            source_str = m.get('source', 'Live Feed')
            if m.get('has_live_odds', True):
                op_w = m.get('open_odds_win', m['odds_1'])
                op_x = m.get('open_odds_draw', m['odds_X'])
                op_l = m.get('open_odds_loss', m['odds_2'])
                title_str = (
                    f"{date_part}{get_official_team_name(m['home_team'])} vs {get_official_team_name(m['away_team'])} | "
                    f"Cuotas: [1] {m['odds_1']:.2f} | [X] {m['odds_X']:.2f} | [2] {m['odds_2']:.2f} "
                    f"(Apertura: {op_w:.2f}/{op_x:.2f}/{op_l:.2f}) [{source_str}]"
                )
            else:
                title_str = (
                    f"{date_part}{get_official_team_name(m['home_team'])} vs {get_official_team_name(m['away_team'])} | "
                    f"[{source_str}]"
                )
            match_choices.append(Choice(title=title_str, value=i))
            
        match_choices.append(Choice(title="--> ⚙️ Ingresar partido y cuotas manualmente", value=-1))
        
        selected_idx = questionary.select(
            "⚽ Selecciona un Partido Próximo en Vivo:",
            choices=match_choices,
            style=CUSTOM_QUESTIONARY_STYLE
        ).ask()
        
        if selected_idx is not None and selected_idx >= 0:
            selected_match = live_matches[selected_idx]
            home_team = selected_match['home_team']
            away_team = selected_match['away_team']
            odds_1 = selected_match['odds_1']
            odds_X = selected_match['odds_X']
            odds_2 = selected_match['odds_2']
            open_odds_win = selected_match.get('open_odds_win', odds_1)
            open_odds_draw = selected_match.get('open_odds_draw', odds_X)
            open_odds_loss = selected_match.get('open_odds_loss', odds_2)
            selected_match_date = selected_match.get('date', '')
            extra_odds = selected_match.get('extra_odds', {})

            if not selected_match.get('has_live_odds', True):
                console.print("\n[bold yellow]⚠️ Atención: Las casas de apuestas aún no han publicado cuotas oficiales para este partido de fecha futura.[/bold yellow]")
                change_odds = questionary.confirm(
                    "¿Deseas ingresar las cuotas de tu casa de apuestas manualmente para este partido?",
                    default=False,
                    style=CUSTOM_QUESTIONARY_STYLE
                ).ask()
                
                if change_odds:
                    try:
                        odds_1 = float(questionary.text("Cuota Local [1]:", default="2.10", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        odds_X = float(questionary.text("Cuota Empate [X]:", default="3.30", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        odds_2 = float(questionary.text("Cuota Visitante [2]:", default="3.40", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        open_odds_win, open_odds_draw, open_odds_loss = odds_1, odds_X, odds_2
                    except (ValueError, TypeError):
                        console.print("[yellow]Usando cuotas baseline estimadas.[/yellow]")
                else:
                    console.print("[dim white]Continuando con estimación baseline del modelo.[/dim white]")
            else:
                console.print(f"[bold green]✔ Cuotas en vivo cargadas correctamente para [bold bright_cyan]{get_official_team_name(home_team)}[/bold bright_cyan] vs [bold bright_magenta]{get_official_team_name(away_team)}[/bold bright_magenta]![/bold green]")

    if not home_team or not away_team:
        teams_in_comp = df[df['competition'] == comp]['team'].dropna().unique().tolist()
        team_choices = [
            Choice(title=get_official_team_name(t), value=t)
            for t in sorted(teams_in_comp)
        ]
        
        home_team = questionary.select("🏠 Equipo Local:", choices=team_choices, style=CUSTOM_QUESTIONARY_STYLE).ask()
        away_team = questionary.select("✈️ Equipo Visitante:", choices=team_choices, style=CUSTOM_QUESTIONARY_STYLE).ask()
        
        if not home_team or not away_team:
            console.print("[red]Debes seleccionar ambos equipos.[/red]")
            return
            
        try:
            odds_1 = float(questionary.text("Cuota Local [1]:", style=CUSTOM_QUESTIONARY_STYLE).ask())
            odds_X = float(questionary.text("Cuota Empate [X]:", style=CUSTOM_QUESTIONARY_STYLE).ask())
            odds_2 = float(questionary.text("Cuota Visitante [2]:", style=CUSTOM_QUESTIONARY_STYLE).ask())
        except (ValueError, TypeError):
            console.print("[red]Cuotas inválidas.[/red]")
            return

    try:
        injuries_home = int(questionary.text("📋 Titulares clave ausentes en Local [0-5]:", default="0", style=CUSTOM_QUESTIONARY_STYLE).ask())
        injuries_away = int(questionary.text("📋 Titulares clave ausentes en Visitante [0-5]:", default="0", style=CUSTOM_QUESTIONARY_STYLE).ask())
        bankroll = float(questionary.text("💰 Bankroll actual ($):", default="1000", style=CUSTOM_QUESTIONARY_STYLE).ask())
    except (ValueError, TypeError):
        injuries_home, injuries_away, bankroll = 0, 0, 1000.0

    try:
        custom_props = questionary.confirm(
            "¿Deseas personalizar las cuotas de Mercados Derivados (Over/Under, BTTS, Córneres, Tarjetas)?",
            default=False,
            style=CUSTOM_QUESTIONARY_STYLE
        ).ask()
        if custom_props:
            ou25_in = questionary.text("Cuota Over 2.5 Goles:", default=str(extra_odds.get('ou25_over', '1.95')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            un25_in = questionary.text("Cuota Under 2.5 Goles:", default=str(extra_odds.get('ou25_under', '1.85')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            btts_in = questionary.text("Cuota BTTS Ambos Anotan:", default=str(extra_odds.get('btts_yes', '1.85')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            corn_in = questionary.text("Cuota Over 9.5 Córneres:", default=str(extra_odds.get('corners_over95', '1.85')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            corn_un_in = questionary.text("Cuota Under 9.5 Córneres:", default=str(extra_odds.get('corners_under95', '1.85')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            card_in = questionary.text("Cuota Over 4.5 Tarjetas:", default=str(extra_odds.get('cards_over45', '1.85')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            extra_odds['ou25_over'] = float(ou25_in)
            extra_odds['ou25_under'] = float(un25_in)
            extra_odds['btts_yes'] = float(btts_in)
            extra_odds['corners_over95'] = float(corn_in)
            extra_odds['corners_under95'] = float(corn_un_in)
            extra_odds['cards_over45'] = float(card_in)
    except Exception:
        pass

    res = predict_match(
        home_team, away_team, comp, odds_1, odds_X, odds_2,
        open_odds_win=open_odds_win, open_odds_draw=open_odds_draw, open_odds_loss=open_odds_loss,
        extra_odds=extra_odds, bankroll=bankroll,
        injuries_home=injuries_home, injuries_away=injuries_away,
        df=df, models_dict=models_dict
    )

    if res is None:
        console.print("[red]Error ejecutando predicción.[/red]")
        return

    home_off = res['home_official']
    away_off = res['away_official']

    console.print(f"[dim white]Parámetros Liga {comp}: EV Umbral = {res['league_ev_thresh']*100:.2f}% | Kelly Frac = {res['league_kelly']:.4f} | Penalizador Incertidumbre = {res['bayesian_penalty']:.3f}[/dim white]")
    
    # Header del Partido
    dt_formatted = format_match_datetime(selected_match_date)
    dt_header = f"[bold yellow]📅 Fecha & Hora del Partido:[/bold yellow] [bold bright_cyan]{dt_formatted}[/bold bright_cyan]\n" if dt_formatted else ""

    op_win_disp = open_odds_win if open_odds_win else odds_1
    op_draw_disp = open_odds_draw if open_odds_draw else odds_X
    op_loss_disp = open_odds_loss if open_odds_loss else odds_2

    has_live_odds = selected_match.get('has_live_odds', True) if selected_match else True
    if has_live_odds:
        odds_footer = f"[dim white]Cuotas Mercado Live: [1] {odds_1:.2f} | [X] {odds_X:.2f} | [2] {odds_2:.2f} (Apertura: [1] {op_win_disp:.2f} | [X] {op_draw_disp:.2f} | [2] {op_loss_disp:.2f})[/dim white]"
    else:
        odds_footer = f"[dim yellow]⚠️ Cuotas en vivo no publicadas en casas de apuestas aún. (Cuotas evaluadas: [1] {odds_1:.2f} | [X] {odds_X:.2f} | [2] {odds_2:.2f})[/dim yellow]"

    console.print(Panel(
        f"{dt_header}"
        f"[bold bright_cyan]🏠 LOCAL: {home_off}[/bold bright_cyan]\n"
        f"[bold bright_magenta]✈️ VISITANTE: {away_off}[/bold bright_magenta]\n\n"
        f"[bold cyan]Marcador Esperado Quant (xG):[/bold cyan] [bold bright_cyan]{home_off}[/bold bright_cyan] [bold bright_yellow]{res['xg_scored']:.2f} - {res['xg_conceded']:.2f}[/bold bright_yellow] [bold bright_magenta]{away_off}[/bold bright_magenta]\n"
        f"[bold cyan]Córneres Esperados:[/bold cyan] [bold bright_yellow]{res['exp_corners']:.1f}[/bold bright_yellow] [dim](Local: {res.get('exp_corners_home', 0):.1f} | Visita: {res.get('exp_corners_away', 0):.1f})[/dim] | [bold cyan]Tarjetas Esperadas:[/bold cyan] [bold bright_yellow]{res['exp_cards']:.1f}[/bold bright_yellow]\n"
        f"{odds_footer}",
        title=f"[bold bright_yellow]⚽ PROYECCIONES QUANT - {get_formatted_comp_name(comp)}[/bold bright_yellow]", border_style="bright_blue"
    ))

    def c_ev(ev):
        if ev > res['league_ev_thresh']:
            return f"[bold bright_green]+{ev*100:.1f}%[/bold bright_green]"
        elif ev > 0:
            return f"[bold yellow]+{ev*100:.1f}%[/bold yellow]"
        else:
            return f"[bold bright_red]{ev*100:.1f}%[/bold bright_red]"

    def c_stake(stk, pct):
        if stk > 0:
            return f"[bold bright_yellow]${stk:,.2f}[/bold bright_yellow] [dim]({pct*100:.2f}%)[/dim]"
        return "$0.00 [dim](0.00%)[/dim]"

    # Tabla 1: Mercados de Partido (1X2 & Doble Oportunidad)
    t1 = Table(title="1. Mercados de Partido (1X2, Doble Oportunidad & Intervalos IC 95%)", show_header=True, header_style="bold white on blue", border_style="cyan")
    t1.add_column("Mercado", style="bold white")
    t1.add_column("Odds (Efectiva)", justify="center")
    t1.add_column("Prob. Blended", justify="right")
    t1.add_column("IC 95% Bayesiano", justify="center", style="dim cyan")
    t1.add_column("Net EV", justify="right")
    t1.add_column("Stake Kelly Multi ($ / %)", justify="right")

    t1.add_row(f"1 ([bold bright_cyan]{home_off}[/bold bright_cyan])", f"{odds_1:.2f} ({res['eff_1']:.2f})", f"{res['blend_win']*100:.1f}%", f"[{res['ci_win_low']*100:.1f}% - {res['ci_win_high']*100:.1f}%]", c_ev(res['ev_win']), c_stake(res['stake_win'], res['pct_win']))
    t1.add_row("X ([bold yellow]Empate[/bold yellow])", f"{odds_X:.2f} ({res['eff_X']:.2f})", f"{res['blend_draw']*100:.1f}%", f"[{res['ci_draw_low']*100:.1f}% - {res['ci_draw_high']*100:.1f}%]", c_ev(res['ev_draw']), c_stake(res['stake_draw'], res['pct_draw']))
    t1.add_row(f"2 ([bold bright_magenta]{away_off}[/bold bright_magenta])", f"{odds_2:.2f} ({res['eff_2']:.2f})", f"{res['blend_loss']*100:.1f}%", f"[{res['ci_loss_low']*100:.1f}% - {res['ci_loss_high']*100:.1f}%]", c_ev(res['ev_loss']), c_stake(res['stake_loss'], res['pct_loss']))
    t1.add_row("1X / AH +0.5 Local", f"{1.0/(1.0/odds_1+1.0/odds_X):.2f} ({res['eff_1X']:.2f})", f"{(res['blend_win']+res['blend_draw'])*100:.1f}%", "-", c_ev(res['ev_1X']), c_stake(res['stake_1X'], res['pct_1X']))
    t1.add_row("X2 / AH +0.5 Visita", f"{1.0/(1.0/odds_X+1.0/odds_2):.2f} ({res['eff_X2']:.2f})", f"{(res['blend_draw']+res['blend_loss'])*100:.1f}%", "-", c_ev(res['ev_X2']), c_stake(res['stake_X2'], res['pct_X2']))
    console.print(t1)

    # Tabla 2: Mercados Derivados (Goles O/U, BTTS, Córneres & Tarjetas)
    t2 = Table(title="2. Mercados Derivados (Goles O/U, BTTS, Córneres & Tarjetas)", show_header=True, header_style="bold white on magenta", border_style="magenta")
    t2.add_column("Mercado Derivado / Prop", style="bold white")
    t2.add_column("Odds Entrada", justify="center")
    t2.add_column("Prob. Modelo", justify="right")
    t2.add_column("Net EV", justify="right")
    t2.add_column("Stake Kelly Multi ($ / %)", justify="right")

    odd_ou25 = float(res.get('odd_ou25', 1.95))
    odd_ou25_under = float(res.get('odd_ou25_under', 1.85))
    odd_btts = float(res.get('odd_btts', 1.85))
    odd_corn = float(res.get('odd_corners', 1.85))
    odd_corn_un = float(res.get('odd_corners_under', 1.85))
    odd_card = float(res.get('odd_cards', 1.85))

    t2.add_row("Over 2.5 Goles Totales", f"{odd_ou25:.2f}", f"{res['prob_over25_goals']*100:.1f}%", c_ev(res['ev_ou25']), c_stake(res['stake_ou25'], res['pct_ou25']))
    t2.add_row("Under 2.5 Goles Totales", f"{odd_ou25_under:.2f}", f"{res['prob_under25_goals']*100:.1f}%", c_ev(res['ev_ou25_under']), c_stake(res['stake_ou25_under'], res['pct_ou25_under']))
    t2.add_row("Both Teams to Score (BTTS Yes)", f"{odd_btts:.2f}", f"{res['prob_btts_yes']*100:.1f}%", c_ev(res['ev_btts']), c_stake(res['stake_btts'], res['pct_btts']))
    t2.add_row("Over 9.5 Córneres Totales", f"{odd_corn:.2f}", f"{res['prob_over95_corners']*100:.1f}%", c_ev(res['ev_corners']), c_stake(res['stake_corners'], res['pct_corners']))
    t2.add_row("Under 9.5 Córneres Totales", f"{odd_corn_un:.2f}", f"{res['prob_under95_corners']*100:.1f}%", c_ev(res['ev_corners_under']), c_stake(res['stake_corners_under'], res['pct_corners_under']))
    t2.add_row("Over 4.5 Tarjetas Totales", f"{odd_card:.2f}", f"{res['prob_over45_cards']*100:.1f}%", c_ev(res['ev_cards']), c_stake(res['stake_cards'], res['pct_cards']))
    console.print(t2)

    best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']

    console.print("\n[bold]=== RECOMENDACIÓN DE STAKING MULTI-MERCADO INSTITUCIONAL ===[/bold]")
    if best_ev > res['league_ev_thresh']:
        if best_stake <= 0 or best_pct < 0.0001:
            console.print(f"[yellow]El mejor EV ({best_bet_name} +{best_ev*100:.2f}%) supera el umbral de la liga ({res['league_ev_thresh']*100:.2f}%), pero el stake Kelly Multi-Activo es minúsculo (< 0.01%).[/yellow] -> [bold]PASS / NO BET[/bold]")
        else:
            rec = Panel(
                f"[bold bright_green]🚀 MEJOR OPORTUNIDAD VALUE DETECTADA[/bold bright_green]\n\n"
                f"Mercado Seleccionado: [bold white]{best_bet_name}[/bold white] @ [bold bright_yellow]{best_raw_odd:.2f}[/bold bright_yellow] (Efectiva: {best_eff_odd:.2f})\n"
                f"Net EV Proyectado: [bold bright_green]+{best_ev*100:.2f}%[/bold bright_green] (Umbral Liga {get_formatted_comp_name(comp)}: {res['league_ev_thresh']*100:.2f}%)\n"
                f"Stake Kelly Multi-Activo: [bold bright_yellow]${best_stake:,.2f}[/bold bright_yellow] ([bold white]{best_pct*100:.2f}%[/bold white] del bankroll)",
                title="SISTEMA DE STAKING INSTITUCIONAL MULTI-MERCADO", border_style="bright_green"
            )
            console.print(rec)
            
            exp_file = export_trade_signals(res)
            console.print(f"[dim white]✔ Señal de trading exportada exitosamente a: [bold yellow]{exp_file}[/bold yellow][/dim white]")
    elif best_ev > 0:
        console.print(Panel(f"[yellow]EDGE INSUFICIENTE EN TODOS LOS MERCADOS.[/yellow]\nEl mejor EV proyectado fue en '{best_bet_name}' con +{best_ev*100:.2f}%. Umbral liga {get_formatted_comp_name(comp)}: {res['league_ev_thresh']*100:.2f}%. -> [bold]PASS / NO BET[/bold]", title="DECISIÓN SISTEMA", border_style="yellow"))
    else:
        console.print(Panel("[bold red]NO HAY VALUE EN NINGÚN MERCADO EN ESTE PARTIDO.[/bold red]\nLas cuotas del mercado son más eficientes que las proyecciones cuantitativas.", title="DECISIÓN SISTEMA", border_style="red"))

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
