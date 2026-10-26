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
from models.train_shots_on_goal import (
    DualPerspectiveShotsModel,
    PerspectiveDistributionalEnsemble,
    PyTorchMLPRegressor,
    SklearnPyTorchRegressorWrapper,
    build_perspective_shots_features,
    calc_over_probs_bivariate,
)
from models.train_stacker import (
    build_level1_features,
    compute_meta_features,
    TemperatureScaledClassifier,
    RegimeGatedMetaStacker,
)
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
__main__.TemperatureScaledClassifier = TemperatureScaledClassifier
__main__.RegimeGatedMetaStacker = RegimeGatedMetaStacker
__main__.DualPerspectiveShotsModel = DualPerspectiveShotsModel
__main__.PerspectiveDistributionalEnsemble = PerspectiveDistributionalEnsemble
__main__.PyTorchMLPRegressor = PyTorchMLPRegressor
__main__.SklearnPyTorchRegressorWrapper = SklearnPyTorchRegressorWrapper

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
SHOTS_PATH = os.path.join(MODEL_DIR, 'shots_on_goal_model.pkl')
CLV_WIN_PATH = os.path.join(MODEL_DIR, 'clv_model_win.pkl')
CLV_DRAW_PATH = os.path.join(MODEL_DIR, 'clv_model_draw.pkl')
CLV_LOSS_PATH = os.path.join(MODEL_DIR, 'clv_model_loss.pkl')
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

def solve_multi_asset_kelly(ev_vec, odds_vec, prob_vec, corr_matrix, bankroll, max_stake_pct, penalty=1.0, max_liquidity=2000.0, league_kelly=0.015, risk_aversion=2.0, max_daily_pct=0.15):
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
        risk_aversion=risk_aversion,
        max_stake_pct=max_stake_pct,
        max_daily_portfolio_pct=max_daily_pct
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
    # Acoplamiento estocástico sobre la rejilla de marcador (i, j) para mercados de intensidad y fricción
    ind_corn = np.fromfunction(lambda i, j: np.clip(0.30 + 0.12 * (i + j) + 0.05 * (i == j), 0.05, 0.95), (rows, cols)).flatten().astype(float)
    ind_corn_un = 1.0 - ind_corn
    ind_card = np.fromfunction(lambda i, j: np.clip(0.35 + 0.15 * (np.abs(i - j) <= 1) + 0.04 * (i + j), 0.05, 0.95), (rows, cols)).flatten().astype(float)
    ind_sot75 = np.fromfunction(lambda i, j: np.clip(0.32 + 0.16 * (i + j) + 0.03 * np.abs(i - j), 0.05, 0.98), (rows, cols)).flatten().astype(float)
    ind_sot85 = np.fromfunction(lambda i, j: np.clip(0.22 + 0.16 * (i + j) + 0.03 * np.abs(i - j), 0.03, 0.96), (rows, cols)).flatten().astype(float)

    matrix_ind = np.vstack([
        ind_win, ind_draw, ind_loss, ind_1X, ind_X2,
        ind_ou25, ind_ou25_un, ind_btts,
        ind_corn, ind_corn_un, ind_card,
        ind_sot75, ind_sot85
    ])
    
    mean_vec = matrix_ind @ weights
    dev = matrix_ind - mean_vec[:, None]
    weighted_cov = (dev * weights[None, :]) @ dev.T
    
    std_vec = np.sqrt(np.diag(weighted_cov))
    std_vec[std_vec == 0] = 1.0
    corr_matrix = weighted_cov / np.outer(std_vec, std_vec)
    np.fill_diagonal(corr_matrix, 1.0)
    # Amortiguación de colinealidad entre 1X2 y Doble Oportunidad (1X/X2) para evitar canibalización en Markowitz
    for idx_single in [0, 1, 2]:
        for idx_dutch in [3, 4]:
            corr_matrix[idx_single, idx_dutch] *= 0.45
            corr_matrix[idx_dutch, idx_single] *= 0.45
    # Perfect negative correlation between complementary pairs
    corr_matrix[5, 6] = -0.99
    corr_matrix[6, 5] = -0.99
    corr_matrix[8, 9] = -0.99
    corr_matrix[9, 8] = -0.99
    corr_matrix[11, 12] = 0.96
    corr_matrix[12, 11] = 0.96
    return np.clip(corr_matrix, -1.0, 1.0)

def export_trade_signals(res, filepath='trade_signals.json'):
    data = {
        'timestamp': datetime.now().isoformat(),
        'match': f"{res['home_official']} vs {res['away_official']}",
        'competition': res['competition'],
        'xg_expected': {'home': res['xg_scored'], 'away': res['xg_conceded']},
        'shots_on_goal_expected': {
            'home': res.get('exp_sot_home', 0.0),
            'away': res.get('exp_sot_away', 0.0),
            'total': res.get('exp_sot_total', 0.0)
        },
        'bayesian_ci_95': {
            'win': [res['ci_win_low'], res['ci_win_high']],
            'draw': [res['ci_draw_low'], res['ci_draw_high']],
            'loss': [res['ci_loss_low'], res['ci_loss_high']],
            '1X': [res.get('ci_1X_low', 0.0), res.get('ci_1X_high', 1.0)],
            'X2': [res.get('ci_X2_low', 0.0), res.get('ci_X2_high', 1.0)]
        },
        'recommended_bet': {
            'market': res['best_bet'][0],
            'net_ev': res['best_bet'][1],
            'stake_usd': res['best_bet'][2],
            'stake_pct': res['best_bet'][3],
            'entry_odd': res['best_bet'][4],
            'effective_odd': res['best_bet'][5],
            'prob': res['best_bet'][6]
        },
        'same_game_parlay': res.get('best_same_game_parlay')
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
    models_dict['shots_on_goal'] = joblib.load(SHOTS_PATH) if os.path.exists(SHOTS_PATH) else None
    if os.path.exists(CLV_WIN_PATH) and os.path.exists(CLV_DRAW_PATH) and os.path.exists(CLV_LOSS_PATH):
        models_dict['clv'] = {
            'win': joblib.load(CLV_WIN_PATH),
            'draw': joblib.load(CLV_DRAW_PATH),
            'loss': joblib.load(CLV_LOSS_PATH)
        }
    else:
        models_dict['clv'] = None
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

def compute_meta_features_live(df_base, open_odds_win, open_odds_draw, open_odds_loss, comp_id=0, optuna_params=None):
    ow = float(open_odds_win) if pd.notna(open_odds_win) and open_odds_win > 1.01 else np.nan
    od = float(open_odds_draw) if pd.notna(open_odds_draw) and open_odds_draw > 1.01 else np.nan
    ol = float(open_odds_loss) if pd.notna(open_odds_loss) and open_odds_loss > 1.01 else np.nan
    df_orig_live = pd.DataFrame(
        {
            'open_odds_win': [ow],
            'open_odds_draw': [od],
            'open_odds_loss': [ol],
            'odds_win': [ow],
            'odds_draw': [od],
            'odds_loss': [ol],
        },
        index=df_base.index,
    )
    res = compute_meta_features(df_base, df_orig_live, prefix="meta_", params=optuna_params)
    res['meta_competition_id'] = comp_id
    res['competition_id'] = comp_id
    return res.fillna(0.0)

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

def predict_shots_on_goal_live(sog_data, df_input_full, home_row, away_row, xg_s, xg_c):
    """
    Predice los Tiros al Arco esperados (Local, Visitante y Total) y las probabilidades
    calibradas de Over/Under 7.5 y 8.5 Tiros al Arco usando el modelo especialista
    DualPerspectiveShotsModel + Binomial Negativa Condicional NB2 (Cameron-Trivedi).
    """
    if not sog_data or 'model_shots' not in sog_data:
        lam_h = float(np.clip(3.2 + 1.1 * xg_s, 2.0, 9.0))
        lam_a = float(np.clip(2.7 + 1.0 * xg_c, 1.5, 8.0))
        lam_tot = lam_h + lam_a
        p75 = float(1.0 - poisson.cdf(7, lam_tot))
        p85 = float(1.0 - poisson.cdf(8, lam_tot))
        return lam_h, lam_a, lam_tot, p75, 1.0 - p75, p85, 1.0 - p85

    if home_row is None:
        home_row = pd.Series({'team_elo': 1500.0})
    if away_row is None:
        away_row = pd.Series({'team_elo': 1500.0})

    df_sog = df_input_full.copy()
    df_sog['predicted_xg_scored_quant'] = float(xg_s)
    df_sog['predicted_xg_conceded_quant'] = float(xg_c)

    o1 = float(df_sog['odds_win'].iloc[0]) if 'odds_win' in df_sog.columns else 2.20
    ox = float(df_sog['odds_draw'].iloc[0]) if 'odds_draw' in df_sog.columns else 3.30
    o2 = float(df_sog['odds_loss'].iloc[0]) if 'odds_loss' in df_sog.columns else 3.20
    marg = (1.0 / max(o1, 1.01)) + (1.0 / max(ox, 1.01)) + (1.0 / max(o2, 1.01))
    df_sog['implied_prob_home'] = (1.0 / max(o1, 1.01)) / marg
    df_sog['implied_prob_away'] = (1.0 / max(o2, 1.01)) / marg
    df_sog['odds_ratio_home_away'] = float(np.log(max(o2, 1.01) / max(o1, 1.01)))
    op1 = float(df_sog['open_odds_win'].iloc[0]) if 'open_odds_win' in df_sog.columns else o1
    df_sog['market_drift_home'] = float(op1 / max(o1, 1.01))

    h_sot = float(home_row.get('shots_on_target_ema5', 0.0) or 0.0)
    a_sot = float(away_row.get('shots_on_target_ema5', 0.0) or 0.0)
    h_st = float(home_row.get('shots_total_ema5', 0.0) or 0.0)
    a_st = float(away_row.get('shots_total_ema5', 0.0) or 0.0)
    if h_sot <= 0.5: h_sot = float(np.clip(3.3 + 1.05 * xg_s, 2.5, 7.5))
    if a_sot <= 0.5: a_sot = float(np.clip(2.8 + 0.95 * xg_c, 2.0, 6.5))
    if h_st <= 1.0: h_st = h_sot / 0.345
    if a_st <= 1.0: a_st = a_sot / 0.345

    for k_c, v_h, v_a in [
        ('clean_sot_c5', h_sot, a_sot),
        ('clean_sot_c10', h_sot, a_sot),
        ('clean_sot_allow5', max(4.2, a_sot), max(4.6, h_sot)),
        ('clean_sot_allow10', max(4.2, a_sot), max(4.6, h_sot)),
        ('clean_st_c8', h_st, a_st),
        ('clean_st_allow8', max(11.0, a_st), max(12.5, h_st)),
        ('venue_sot_c8', h_sot * 1.04, a_sot * 0.96),
        ('venue_sot_allow8', max(4.0, a_sot), max(4.7, h_sot)),
        ('venue_st_c8', h_st * 1.04, a_st * 0.96),
        ('venue_st_allow8', max(10.8, a_st), max(12.8, h_st)),
        ('comp_venue_sot_prior', 4.75, 3.95),
    ]:
        df_sog[f'h_{k_c}'] = float(home_row.get(f'h_{k_c}', v_h) or v_h)
        df_sog[f'a_{k_c}'] = float(away_row.get(f'a_{k_c}', v_a) or v_a)

    try:
        X_h = build_perspective_shots_features(df_sog, perspective='home')
        X_a = build_perspective_shots_features(df_sog, perspective='away')
        req_cols = sog_data.get('features', list(X_h.columns))
        for c in req_cols:
            if c not in X_h.columns: X_h[c] = 0.0
            if c not in X_a.columns: X_a[c] = 0.0
        preds = sog_data['model_shots'].predict_from_perspectives(X_h, X_a)
        lam_h = float(np.clip(preds[0, 0], 1.5, 11.0))
        lam_a = float(np.clip(preds[0, 1], 1.2, 9.5))
    except Exception:
        lam_h = float(np.clip(h_sot, 2.0, 9.0))
        lam_a = float(np.clip(a_sot, 1.5, 8.0))

    lam_tot = lam_h + lam_a
    alpha_h = float(sog_data.get('alpha_h', 0.06))
    alpha_a = float(sog_data.get('alpha_a', 0.06))
    rho_res = float(sog_data.get('rho_res', -0.05))
    calibs = sog_data.get('calibrators', {})

    raw_probs = calc_over_probs_bivariate(
        np.array([lam_h]), np.array([lam_a]), [7.5, 8.5],
        alpha_h=alpha_h, alpha_a=alpha_a, rho_res=rho_res
    )
    p75 = float(raw_probs.get('prob_over_7.5', raw_probs.get('prob_over_7_5_shots', [0.55]))[0])
    p85 = float(raw_probs.get('prob_over_8.5', raw_probs.get('prob_over_8_5_shots', [0.44]))[0])

    for line_val in [7.5, 8.5]:
        cal_key = f'prob_over_{line_val}_total'
        if cal_key in calibs and calibs[cal_key] is not None:
            try:
                if line_val == 7.5:
                    p75 = float(np.clip(calibs[cal_key].predict(np.array([p75]))[0], 0.02, 0.98))
                else:
                    p85 = float(np.clip(calibs[cal_key].predict(np.array([p85]))[0], 0.02, 0.98))
            except Exception:
                pass

    p85 = min(p85, p75 - 0.01) if p75 > 0.03 else p85
    return lam_h, lam_a, lam_tot, p75, float(1.0 - p75), p85, float(1.0 - p85)

def predict_clv_live(clv_dict, df_final, open_1, open_X, open_2, prob_loss=0.33, prob_draw=0.33, prob_win=0.34, fund_probs=None, comp='E0'):
    """
    Proyecta el Closing Line Value (CLV Lineal %) acoplado mediante normalización Softmax
    utilizando los modelos especialistas XGBoost (clv_model_win/draw/loss.pkl).
    """
    if not clv_dict or not all(k in clv_dict for k in ['win', 'draw', 'loss']):
        return [0.005, 0.005, 0.005]
    try:
        o_w = max(float(open_1), 1.01)
        o_d = max(float(open_X), 1.01)
        o_l = max(float(open_2), 1.01)
        imp_l = 1.0 / o_l
        imp_d = 1.0 / o_d
        imp_w = 1.0 / o_w
        vig_o = imp_l + imp_d + imp_w
        of_l = imp_l / vig_o
        of_d = imp_d / vig_o
        of_w = imp_w / vig_o

        s_l = float(np.clip(prob_loss, 1e-4, 0.999))
        s_d = float(np.clip(prob_draw, 1e-4, 0.999))
        s_w = float(np.clip(prob_win, 1e-4, 0.999))
        if fund_probs is not None and len(fund_probs) == 3:
            f_l, f_d, f_w = float(fund_probs[0]), float(fund_probs[1]), float(fund_probs[2])
        else:
            f_l, f_d, f_w = s_l, s_d, s_w

        net_w = 1.0 + (o_w - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_d = 1.0 + (o_d - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_l = 1.0 + (o_l - 1.0) * (1.0 - TAX_RETENTION_RATE)
        ev_w = (s_w * net_w) - 1.0
        ev_d = (s_d * net_d) - 1.0
        ev_l = (s_l * net_l) - 1.0

        mkt_ent = float(-sum(p * np.log(max(p, 1e-6)) for p in (of_l, of_d, of_w)))
        stk_ent = float(-sum(p * np.log(max(p, 1e-6)) for p in (s_l, s_d, s_w)))

        X_clv = pd.DataFrame({
            'stacker_prob_loss': [s_l],
            'stacker_prob_draw': [s_d],
            'stacker_prob_win': [s_w],
            'fund_prob_loss': [f_l],
            'fund_prob_draw': [f_d],
            'fund_prob_win': [f_w],
            'log_open_win': [float(np.log(o_w))],
            'log_open_draw': [float(np.log(o_d))],
            'log_open_loss': [float(np.log(o_l))],
            'log_open_win_sq': [float(np.log(o_w) ** 2)],
            'log_open_loss_sq': [float(np.log(o_l) ** 2)],
            'implied_ratio_win_loss': [float(of_w / (of_l + 1e-6))],
            'implied_ratio_win_draw': [float(of_w / (of_d + 1e-6))],
            'edge_diff_loss': [float(s_l - of_l)],
            'edge_diff_draw': [float(s_d - of_d)],
            'edge_diff_win': [float(s_w - of_w)],
            'edge_log_loss': [float(np.log(np.clip(s_l / of_l, 1e-6, 1e6)))],
            'edge_log_draw': [float(np.log(np.clip(s_d / of_d, 1e-6, 1e6)))],
            'edge_log_win': [float(np.log(np.clip(s_w / of_w, 1e-6, 1e6)))],
            'edge_win_vs_loss': [float((s_w - of_w) - (s_l - of_l))],
            'edge_win_vs_draw': [float((s_w - of_w) - (s_d - of_d))],
            'market_entropy': [mkt_ent],
            'stacker_entropy': [stk_ent],
            'entropy_diff': [float(stk_ent - mkt_ent)],
            'ev_open_win': [ev_w],
            'ev_open_draw': [ev_d],
            'ev_open_loss': [ev_l],
            'ev_dominance_win': [float(ev_w - max(ev_d, ev_l))],
            'ev_dominance_loss': [float(ev_l - max(ev_w, ev_d))],
            'ev_over_vig_win': [float(ev_w / (vig_o - 1.0 + 1e-4))],
            'ev_over_vig_loss': [float(ev_l / (vig_o - 1.0 + 1e-4))],
            'vig_open': [float(vig_o - 1.0)],
            'open_implied_loss': [imp_l],
            'open_implied_draw': [imp_d],
            'open_implied_win': [imp_w],
            'competition_id': pd.Series([str(comp)], dtype='category')
        })

        req_feats = clv_dict['win'].get('features', list(X_clv.columns))
        for f in req_feats:
            if f not in X_clv.columns:
                X_clv[f] = 0.0
        X_clv = X_clv[req_feats]

        d_w_log = float(clv_dict['win']['model'].predict(X_clv)[0])
        d_d_log = float(clv_dict['draw']['model'].predict(X_clv)[0])
        d_l_log = float(clv_dict['loss']['model'].predict(X_clv)[0])

        pc_w = of_w * np.exp(d_w_log)
        pc_d = of_d * np.exp(d_d_log)
        pc_l = of_l * np.exp(d_l_log)
        tot_pc = max(pc_w + pc_d + pc_l, 1e-6)

        clv_w = float((pc_w / tot_pc) / of_w - 1.0)
        clv_d = float((pc_d / tot_pc) / of_d - 1.0)
        clv_l = float((pc_l / tot_pc) / of_l - 1.0)
        return [clv_l, clv_d, clv_w]
    except Exception:
        return [0.005, 0.005, 0.005]

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
    league_alpha_low = get_param_by_comp(alpha_div_low_dict, comp, 0.85)
    league_alpha_med = get_param_by_comp(alpha_div_med_dict, comp, 0.70)
    league_alpha_high = get_param_by_comp(alpha_div_high_dict, comp, 0.50)

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

    df_fund_raw = pd.DataFrame({
        'predicted_xg_scored_quant': [xg_s], 'predicted_xg_conceded_quant': [xg_c],
        'quant_win_prob': [q_win], 'quant_draw_prob': [q_draw], 'quant_loss_prob': [q_loss],
        'prob_loss_ctx': [ctx_probs[0]], 'prob_draw_ctx': [ctx_probs[1]], 'prob_win_ctx': [ctx_probs[2]],
        'prob_loss_nn': [nn_probs[0]], 'prob_draw_nn': [nn_probs[1]], 'prob_win_nn': [nn_probs[2]],
        'prob_is_draw': [draws_prob]
    })
    df_fund_full = build_level1_features(df_fund_raw, params=fund_data.get('optuna_params'))
    for f_col in fund_data['features']:
        if f_col not in df_fund_full.columns:
            df_fund_full[f_col] = 0.0
    df_fund_input = df_fund_full[fund_data['features']]
    fund_probs = fund_data['model'].predict_proba(df_fund_input)[0]

    df_fund_out = pd.DataFrame({'fund_prob_loss': [fund_probs[0]], 'fund_prob_draw': [fund_probs[1]], 'fund_prob_win': [fund_probs[2]]})
    df_mkt = pd.DataFrame({'prob_loss_mkt': [mkt_probs[0]], 'prob_draw_mkt': [mkt_probs[1]], 'prob_win_mkt': [mkt_probs[2]],
                           'prob_loss_gbm': [gbm_probs[0]], 'prob_draw_gbm': [gbm_probs[1]], 'prob_win_gbm': [gbm_probs[2]]})
    
    df_final = compute_meta_features_live(
        pd.concat([df_fund_out, df_fund_full, df_mkt], axis=1),
        input_data['open_odds_win'], input_data['open_odds_draw'], input_data['open_odds_loss'],
        input_data['competition_id'],
        optuna_params=final_data.get('optuna_params', fund_data.get('optuna_params')),
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

    all_1X_probs = [min(w + d, 0.995) for w, d in zip(all_win_probs, all_draw_probs)]
    all_X2_probs = [min(l + d, 0.995) for l, d in zip(all_loss_probs, all_draw_probs)]
    _, se_1X, ci_1X_low, ci_1X_high, pen_1X = calculate_bayesian_uncertainty(all_1X_probs)
    _, se_X2, ci_X2_low, ci_X2_high, pen_X2 = calculate_bayesian_uncertainty(all_X2_probs)

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

    sog_data = models_dict.get('shots_on_goal')
    exp_sot_home, exp_sot_away, exp_sot_total, prob_over75_sot, prob_under75_sot, prob_over85_sot, prob_under85_sot = predict_shots_on_goal_live(
        sog_data, df_input_full, home_row_corners, away_row_corners, xg_s, xg_c
    )

    clv_dict = models_dict.get('clv')
    pred_clv = predict_clv_live(
        clv_dict, df_final,
        input_data['open_odds_win'], input_data['open_odds_draw'], input_data['open_odds_loss'],
        prob_loss=prob_loss, prob_draw=prob_draw, prob_win=prob_win,
        fund_probs=fund_probs, comp=comp
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
    blend_sum = max(blend_win + blend_draw + blend_loss, 1e-6)
    blend_win, blend_draw, blend_loss = blend_win / blend_sum, blend_draw / blend_sum, blend_loss / blend_sum

    net_odds_1 = 1.0 + (odds_1 - 1.0) * (1.0 - TAX_RETENTION_RATE)
    net_odds_X = 1.0 + (odds_X - 1.0) * (1.0 - TAX_RETENTION_RATE)
    net_odds_2 = 1.0 + (odds_2 - 1.0) * (1.0 - TAX_RETENTION_RATE)

    ev_win = (blend_win * net_odds_1) - 1.0 - EXPECTED_CLV_DROP
    ev_draw = (blend_draw * net_odds_X) - 1.0 - EXPECTED_CLV_DROP
    ev_loss = (blend_loss * net_odds_2) - 1.0 - EXPECTED_CLV_DROP

    # Doble Oportunidad 1X y X2 (Dutching Multi-Mercado Cuantitativo):
    # Preservación del posterior calibrado del Stacker Nivel 2 en alta certeza (p >= 0.68),
    # CLV ponderado por inversión (q_a, q_b) y fricción proporcional al beneficio potencial.
    total_implied_1X = impl_win + impl_draw
    combined_odds_1X = 1.0 / total_implied_1X
    net_combined_1X = 1.0 + (combined_odds_1X - 1.0) * (1.0 - TAX_RETENTION_RATE)
    stacker_1X = prob_win + prob_draw
    mkt_1X = market_prob_win + market_prob_draw
    dutch_alpha_1X = max(league_alpha_low, 0.65) if stacker_1X >= 0.68 else league_alpha_low
    blend_1X = (dutch_alpha_1X * stacker_1X) + ((1.0 - dutch_alpha_1X) * mkt_1X)
    friction_1X = EXPECTED_CLV_DROP * min(combined_odds_1X - 1.0, 0.60)
    ev_1X = (blend_1X * net_combined_1X) - 1.0 - friction_1X
    clv_1X = (impl_win * pred_clv[2] + impl_draw * pred_clv[1]) / total_implied_1X

    total_implied_X2 = impl_draw + impl_loss
    combined_odds_X2 = 1.0 / total_implied_X2
    net_combined_X2 = 1.0 + (combined_odds_X2 - 1.0) * (1.0 - TAX_RETENTION_RATE)
    stacker_X2 = prob_loss + prob_draw
    mkt_X2 = market_prob_loss + market_prob_draw
    dutch_alpha_X2 = max(league_alpha_low, 0.65) if stacker_X2 >= 0.68 else league_alpha_low
    blend_X2 = (dutch_alpha_X2 * stacker_X2) + ((1.0 - dutch_alpha_X2) * mkt_X2)
    friction_X2 = EXPECTED_CLV_DROP * min(combined_odds_X2 - 1.0, 0.60)
    ev_X2 = (blend_X2 * net_combined_X2) - 1.0 - friction_X2
    clv_X2 = (impl_loss * pred_clv[0] + impl_draw * pred_clv[1]) / total_implied_X2

    # Mercados secundarios: Calibración bayesiana contra línea implícita de mercado (sin -100.0% artificial)
    p_mkt_ov25 = float(np.clip(0.72 * prob_over25_goals + 0.28 * 0.50, 0.22, 0.78))
    p_mkt_un25 = 1.0 - p_mkt_ov25
    default_odd_ov25 = round(float(np.clip(1.0 / (p_mkt_ov25 * 1.032), 1.35, 3.20)), 2)
    default_odd_un25 = round(float(np.clip(1.0 / (p_mkt_un25 * 1.032), 1.35, 3.20)), 2)

    odd_ou25 = float(extra_odds.get('ou25_over', default_odd_ov25))
    odd_ou25_under = float(extra_odds.get('ou25_under', default_odd_un25))
    ev_ou25 = (prob_over25_goals * (1.0 + (odd_ou25 - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP
    ev_ou25_under = (prob_under25_goals * (1.0 + (odd_ou25_under - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    p_mkt_btts = float(np.clip(0.72 * prob_btts_yes + 0.28 * 0.51, 0.25, 0.76))
    default_odd_btts = round(float(np.clip(1.0 / (p_mkt_btts * 1.032), 1.40, 2.80)), 2)
    odd_btts = float(extra_odds.get('btts_yes', default_odd_btts))
    ev_btts = (prob_btts_yes * (1.0 + (odd_btts - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    p_mkt_corn_ov = float(np.clip(0.72 * prob_over95_corners + 0.28 * 0.49, 0.25, 0.76))
    p_mkt_corn_un = 1.0 - p_mkt_corn_ov
    default_odd_corn_ov = round(float(np.clip(1.0 / (p_mkt_corn_ov * 1.032), 1.40, 2.80)), 2)
    default_odd_corn_un = round(float(np.clip(1.0 / (p_mkt_corn_un * 1.032), 1.40, 2.80)), 2)
    odd_corners = float(extra_odds.get('corners_over95', default_odd_corn_ov))
    odd_corners_under = float(extra_odds.get('corners_under95', default_odd_corn_un))
    ev_corners = (prob_over95_corners * (1.0 + (odd_corners - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP
    ev_corners_under = (prob_under95_corners * (1.0 + (odd_corners_under - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    p_mkt_card = float(np.clip(0.72 * prob_over45_cards + 0.28 * 0.48, 0.22, 0.76))
    default_odd_card = round(float(np.clip(1.0 / (p_mkt_card * 1.032), 1.40, 2.80)), 2)
    odd_cards = float(extra_odds.get('cards_over45', default_odd_card))
    ev_cards = (prob_over45_cards * (1.0 + (odd_cards - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    p_mkt_sot75 = float(np.clip(0.72 * prob_over75_sot + 0.28 * 0.55, 0.25, 0.80))
    p_mkt_sot85 = float(np.clip(0.72 * prob_over85_sot + 0.28 * 0.45, 0.22, 0.75))
    default_odd_sot75 = round(float(np.clip(1.0 / (p_mkt_sot75 * 1.032), 1.34, 2.85)), 2)
    default_odd_sot85 = round(float(np.clip(1.0 / (p_mkt_sot85 * 1.032), 1.40, 3.10)), 2)
    odd_sot75 = float(extra_odds.get('sot_over75', default_odd_sot75))
    odd_sot85 = float(extra_odds.get('sot_over85', default_odd_sot85))
    ev_sot75 = (prob_over75_sot * (1.0 + (odd_sot75 - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP
    ev_sot85 = (prob_over85_sot * (1.0 + (odd_sot85 - 1.0) * (1.0 - TAX_RETENTION_RATE))) - 1.0 - EXPECTED_CLV_DROP

    max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

    corr_matrix = compute_poisson_market_correlations(poisson_matrix)
    ev_main = [ev_win, ev_draw, ev_loss, ev_1X, ev_X2]
    odds_main = [odds_1, odds_X, odds_2, combined_odds_1X, combined_odds_X2]
    prob_main = [blend_win, blend_draw, blend_loss, blend_1X, blend_X2]

    ev_deriv = [ev_ou25, ev_ou25_under, ev_btts, ev_corners, ev_corners_under, ev_cards, ev_sot75, ev_sot85]
    odds_deriv = [odd_ou25, odd_ou25_under, odd_btts, odd_corners, odd_corners_under, odd_cards, odd_sot75, odd_sot85]
    prob_deriv = [prob_over25_goals, prob_under25_goals, prob_btts_yes, prob_over95_corners, prob_under95_corners, prob_over45_cards, prob_over75_sot, prob_over85_sot]

    avg_penalty = np.mean([pen_win, pen_draw, pen_loss])
    stakes_main, pcts_main = solve_multi_asset_kelly(
        ev_main, odds_main, prob_main, corr_matrix[:5, :5], bankroll, MAX_STAKE_PCT,
        penalty=avg_penalty, max_liquidity=max_liquidity, league_kelly=league_kelly
    )
    stakes_deriv, pcts_deriv = solve_multi_asset_kelly(
        ev_deriv, odds_deriv, prob_deriv, corr_matrix[5:, 5:], bankroll, MAX_STAKE_PCT * 0.75,
        penalty=avg_penalty * 0.45, max_liquidity=max_liquidity, league_kelly=league_kelly,
        risk_aversion=3.5, max_daily_pct=0.05
    )

    s_win, s_draw, s_loss, s_1X, s_X2 = stakes_main
    p_win_stk, p_draw_stk, p_loss_stk, p_1X_stk, p_X2_stk = pcts_main
    s_ou25, s_un25, s_btts, s_corn, s_corn_un, s_cards, s_sot75, s_sot85 = stakes_deriv
    p_ou25_stk, p_un25_stk, p_btts_stk, p_corn_stk, p_corn_un_stk, p_cards_stk, p_sot75_stk, p_sot85_stk = pcts_deriv

    # Desglose exacto de postura Dutching (Equal-Payout Stake Split) para apostador manual
    share_1_in_1X = impl_win / max(total_implied_1X, 1e-6)
    share_X_in_1X = impl_draw / max(total_implied_1X, 1e-6)
    stake_1X_leg1 = s_1X * share_1_in_1X
    stake_1X_legX = s_1X * share_X_in_1X

    share_2_in_X2 = impl_loss / max(total_implied_X2, 1e-6)
    share_X_in_X2 = impl_draw / max(total_implied_X2, 1e-6)
    stake_X2_leg2 = s_X2 * share_2_in_X2
    stake_X2_legX = s_X2 * share_X_in_X2

    eff_1 = calculate_market_slippage(odds_1, s_win, max_liquidity)
    eff_X = calculate_market_slippage(odds_X, s_draw, max_liquidity)
    eff_2 = calculate_market_slippage(odds_2, s_loss, max_liquidity)
    eff_1X = calculate_market_slippage(combined_odds_1X, s_1X, max_liquidity)
    eff_X2 = calculate_market_slippage(combined_odds_X2, s_X2, max_liquidity)

    home_official = get_official_team_name(home_team)
    away_official = get_official_team_name(away_team)

    # Identificar pata segura elegible para Parlays de Bajo Riesgo (p >= 0.72, 1.14 <= odd <= 1.72, EV >= 0.5%)
    safe_parlay_leg = None
    raw_cands_parlay = [
        (f"1 ({home_official})", prob_win, odds_1, pred_clv[2], 'single', 2, None),
        (f"2 ({away_official})", prob_loss, odds_2, pred_clv[0], 'single', 0, None),
        (f"Doble Oportunidad 1X ({home_official}/Empate)", stacker_1X, combined_odds_1X, clv_1X, 'dutching', 2, 1),
        (f"Doble Oportunidad X2 ({away_official}/Empate)", stacker_X2, combined_odds_X2, clv_X2, 'dutching', 0, 1),
    ]
    valid_safe_legs = []
    for lbl_s, p_s, o_s, clv_s, bt_s, bc_s, sc_s in raw_cands_parlay:
        net_o_s = 1.0 + (o_s - 1.0) * (1.0 - TAX_RETENTION_RATE)
        ev_s = (p_s * net_o_s) - 1.0
        if p_s >= 0.72 and 1.14 <= o_s <= 1.72 and ev_s >= 0.005:
            valid_safe_legs.append({
                'label': lbl_s,
                'match_title': f"{home_official} vs {away_official}",
                'prob': float(p_s),
                'raw_odd': float(o_s),
                'net_odd': float(net_o_s),
                'ev': float(ev_s),
                'clv': float(clv_s),
                'bet_type': bt_s,
                'comp': comp,
                'kelly_fraction': float(league_kelly),
                'best_choice': bc_s,
                'secondary_choice': sc_s,
                'odds': [float(odds_2), float(odds_X), float(odds_1)]
            })
    if valid_safe_legs:
        safe_parlay_leg = max(valid_safe_legs, key=lambda x: x['prob'])

    # Motor Bet Builder / Parlay de 2 Patas del Mismo Partido (Dutching / 1X2 + Mercado Derivado con Cópula Bivariada)
    sgp_leg_a_cands = [
        ("Doble Oportunidad 1X", blend_1X, combined_odds_1X, 3, 'dutching_1X'),
        ("Doble Oportunidad X2", blend_X2, combined_odds_X2, 4, 'dutching_X2'),
        (f"1 ({home_official})", blend_win, odds_1, 0, 'single_1'),
        (f"2 ({away_official})", blend_loss, odds_2, 2, 'single_2'),
    ]
    sgp_leg_b_cands = [
        ("Over 2.5 Goles", prob_over25_goals, odd_ou25, 5),
        ("Under 2.5 Goles", prob_under25_goals, odd_ou25_under, 6),
        ("BTTS (Ambos Anotan)", prob_btts_yes, odd_btts, 7),
        ("Over 7.5 Tiros al Arco (SoT)", prob_over75_sot, odd_sot75, 11),
        ("Over 8.5 Tiros al Arco (SoT)", prob_over85_sot, odd_sot85, 12),
        ("Over 9.5 Córneres", prob_over95_corners, odd_corners, 8),
    ]
    same_game_parlays = []
    for name_a, p_a, odd_a, idx_a, kind_a in sgp_leg_a_cands:
        if p_a < 0.24:
            continue
        for name_b, p_b, odd_b, idx_b in sgp_leg_b_cands:
            if p_b < 0.50:
                continue
            rho_raw = float(corr_matrix[idx_a, idx_b])
            if rho_raw < -0.35:
                continue
            # Para Doble Oportunidad (1X/X2) + mercados de volumen ofensivo (Over 2.5, SoT, BTTS, Córneres),
            # la correlación de intensidad de partido (ritmo / transiciones) aporta un acoplamiento mínimo >= 0
            rho_ab = max(rho_raw, 0.04) if (kind_a.startswith('dutching') and idx_b in (5, 7, 8, 11, 12)) else rho_raw
            sig_a = np.sqrt(max(p_a * (1.0 - p_a), 1e-4))
            sig_b = np.sqrt(max(p_b * (1.0 - p_b), 1e-4))
            p_joint = float(np.clip(p_a * p_b + rho_ab * sig_a * sig_b, 0.02, min(p_a, p_b) - 0.005))
            raw_prod_odd = odd_a * odd_b
            # Cuota combinada Bet Builder / Parlay (las cuotas individuales ya incluyen el overround de la casa)
            sgp_odd = float(np.clip(1.0 + (raw_prod_odd - 1.0) * (1.0 - 0.04 * max(rho_ab, 0.0)), 1.20, 6.50))
            net_sgp_odd = 1.0 + (sgp_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
            ev_sgp = (p_joint * net_sgp_odd) - 1.0 - EXPECTED_CLV_DROP
            sigma_sgp = net_sgp_odd * np.sqrt(max(p_joint * (1.0 - p_joint), 0.02))
            sharpe_sgp = (ev_sgp / sigma_sgp) if ev_sgp > 0 else ev_sgp
            kelly_raw_sgp = ((p_joint * net_sgp_odd - 1.0) / max(net_sgp_odd - 1.0, 0.05)) if ev_sgp > 0 else 0.0
            pct_sgp = float(np.clip(kelly_raw_sgp * league_kelly * avg_penalty * 2.2, 0.0, MAX_STAKE_PCT * 0.85)) if ev_sgp > 0 else 0.0
            stake_sgp = float(bankroll * pct_sgp)
            same_game_parlays.append({
                'label': f"{name_a} + {name_b}",
                'leg_a': name_a,
                'leg_b': name_b,
                'kind_a': kind_a,
                'odd_a': round(float(odd_a), 2),
                'odd_b': round(float(odd_b), 2),
                'sgp_odd': round(sgp_odd, 2),
                'prob_a': float(p_a),
                'prob_b': float(p_b),
                'prob_joint': p_joint,
                'corr': rho_ab,
                'ev': float(ev_sgp),
                'sharpe': float(sharpe_sgp),
                'stake': stake_sgp,
                'pct': pct_sgp
            })
    same_game_parlays.sort(key=lambda x: x['sharpe'], reverse=True)
    best_same_game_parlay = same_game_parlays[0] if same_game_parlays else None

    all_bets = [
        (f"1 ({home_official})", ev_win, s_win, p_win_stk, odds_1, eff_1, blend_win),
        ("X (Empate)", ev_draw, s_draw, p_draw_stk, odds_X, eff_X, blend_draw),
        (f"2 ({away_official})", ev_loss, s_loss, p_loss_stk, odds_2, eff_2, blend_loss),
        ("Doble Oportunidad 1X", ev_1X, s_1X, p_1X_stk, combined_odds_1X, eff_1X, blend_1X),
        ("Doble Oportunidad X2", ev_X2, s_X2, p_X2_stk, combined_odds_X2, eff_X2, blend_X2),
        ("Over 2.5 Goles", ev_ou25, s_ou25, p_ou25_stk, odd_ou25, odd_ou25, prob_over25_goals),
        ("Under 2.5 Goles", ev_ou25_under, s_un25, p_un25_stk, odd_ou25_under, odd_ou25_under, prob_under25_goals),
        ("BTTS (Ambos Anotan)", ev_btts, s_btts, p_btts_stk, odd_btts, odd_btts, prob_btts_yes),
        ("Over 7.5 Tiros al Arco (SoT)", ev_sot75, s_sot75, p_sot75_stk, odd_sot75, odd_sot75, prob_over75_sot),
        ("Over 8.5 Tiros al Arco (SoT)", ev_sot85, s_sot85, p_sot85_stk, odd_sot85, odd_sot85, prob_over85_sot),
        ("Over 9.5 Córneres", ev_corners, s_corn, p_corn_stk, odd_corners, odd_corners, prob_over95_corners),
        ("Under 9.5 Córneres", ev_corners_under, s_corn_un, p_corn_un_stk, odd_corners_under, odd_corners_under, prob_under95_corners),
        ("Over 4.5 Tarjetas", ev_cards, s_cards, p_cards_stk, odd_cards, odd_cards, prob_over45_cards)
    ]

    # Criterio Cuantitativo de Selección Principal: Ratio de Sharpe de Bernoulli (EV / sigma_R) con filtro MAX_PLAYABLE_ODDS = 3.40
    # Prioriza mercados con cuotas reales verificadas (1X2 / Dutching + mercados presentes en extra_odds)
    verified_labels = {
        f"1 ({home_official})", "X (Empate)", f"2 ({away_official})",
        "Doble Oportunidad 1X", "Doble Oportunidad X2"
    }
    if 'ou25_over' in extra_odds: verified_labels.add("Over 2.5 Goles")
    if 'ou25_under' in extra_odds: verified_labels.add("Under 2.5 Goles")
    if 'btts_yes' in extra_odds: verified_labels.add("BTTS (Ambos Anotan)")
    if 'sot_over75' in extra_odds: verified_labels.add("Over 7.5 Tiros al Arco (SoT)")
    if 'sot_over85' in extra_odds: verified_labels.add("Over 8.5 Tiros al Arco (SoT)")
    if 'corners_over95' in extra_odds: verified_labels.add("Over 9.5 Córneres")
    if 'corners_under95' in extra_odds: verified_labels.add("Under 9.5 Córneres")
    if 'cards_over45' in extra_odds: verified_labels.add("Over 4.5 Tarjetas")

    def _bernoulli_sharpe_key(b_tuple):
        b_name, b_ev, b_odd, b_prob = b_tuple[0], b_tuple[1], b_tuple[4], b_tuple[6]
        if b_odd > 3.40 or b_odd < 1.14:
            return -1.0
        if b_name not in verified_labels:
            return -0.5 + b_ev
        if b_ev <= 0:
            return b_ev
        net_o = 1.0 + (b_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
        sigma_r = net_o * np.sqrt(max(b_prob * (1.0 - b_prob), 0.02))
        return b_ev / sigma_r

    sorted_bets = sorted(all_bets, key=_bernoulli_sharpe_key, reverse=True)
    best_bet = sorted_bets[0]

    return {
        'home_team': home_team, 'away_team': away_team, 'competition': comp,
        'home_official': home_official, 'away_official': away_official,
        'xg_scored': xg_s, 'xg_conceded': xg_c,
        'exp_corners': exp_corners, 'exp_corners_home': exp_corners_home, 'exp_corners_away': exp_corners_away,
        'exp_sot_total': exp_sot_total, 'exp_sot_home': exp_sot_home, 'exp_sot_away': exp_sot_away,
        'exp_cards': exp_cards,
        'prob_win': prob_win, 'prob_draw': prob_draw, 'prob_loss': prob_loss,
        'ci_win_low': ci_win_low, 'ci_win_high': ci_win_high, 'se_win': se_win,
        'ci_draw_low': ci_draw_low, 'ci_draw_high': ci_draw_high, 'se_draw': se_draw,
        'ci_loss_low': ci_loss_low, 'ci_loss_high': ci_loss_high, 'se_loss': se_loss,
        'ci_1X_low': ci_1X_low, 'ci_1X_high': ci_1X_high, 'se_1X': se_1X,
        'ci_X2_low': ci_X2_low, 'ci_X2_high': ci_X2_high, 'se_X2': se_X2,
        'blend_win': blend_win, 'blend_draw': blend_draw, 'blend_loss': blend_loss,
        'blend_1X': blend_1X, 'blend_X2': blend_X2,
        'prob_over25_goals': prob_over25_goals, 'prob_under25_goals': prob_under25_goals,
        'prob_btts_yes': prob_btts_yes, 'prob_btts_no': prob_btts_no,
        'prob_over95_corners': prob_over95_corners, 'prob_under95_corners': prob_under95_corners,
        'prob_over75_sot': prob_over75_sot, 'prob_under75_sot': prob_under75_sot,
        'prob_over85_sot': prob_over85_sot, 'prob_under85_sot': prob_under85_sot,
        'prob_over45_cards': prob_over45_cards,
        'ev_win': ev_win, 'ev_draw': ev_draw, 'ev_loss': ev_loss,
        'ev_1X': ev_1X, 'ev_X2': ev_X2,
        'pred_clv_win': pred_clv[2], 'pred_clv_draw': pred_clv[1], 'pred_clv_loss': pred_clv[0],
        'pred_clv_1X': clv_1X, 'pred_clv_X2': clv_X2,
        'ev_ou25': ev_ou25, 'ev_ou25_under': ev_ou25_under,
        'ev_btts': ev_btts,
        'odd_ou25': odd_ou25, 'odd_ou25_under': odd_ou25_under,
        'odd_btts': odd_btts, 'odd_corners': odd_corners, 'odd_corners_under': odd_corners_under, 'odd_cards': odd_cards,
        'odd_sot75': odd_sot75, 'odd_sot85': odd_sot85, 'ev_sot75': ev_sot75, 'ev_sot85': ev_sot85,
        'ev_corners': ev_corners, 'ev_corners_under': ev_corners_under,
        'ev_cards': ev_cards,
        'stake_win': s_win, 'stake_draw': s_draw, 'stake_loss': s_loss,
        'stake_1X': s_1X, 'stake_X2': s_X2,
        'stake_1X_leg1': stake_1X_leg1, 'stake_1X_legX': stake_1X_legX,
        'stake_X2_leg2': stake_X2_leg2, 'stake_X2_legX': stake_X2_legX,
        'share_1_in_1X': share_1_in_1X, 'share_X_in_1X': share_X_in_1X,
        'share_2_in_X2': share_2_in_X2, 'share_X_in_X2': share_X_in_X2,
        'stake_ou25': s_ou25, 'stake_ou25_under': s_un25,
        'stake_btts': s_btts,
        'stake_sot75': s_sot75, 'stake_sot85': s_sot85,
        'stake_corners': s_corn, 'stake_corners_under': s_corn_un,
        'stake_cards': s_cards,
        'pct_win': p_win_stk, 'pct_draw': p_draw_stk, 'pct_loss': p_loss_stk,
        'pct_1X': p_1X_stk, 'pct_X2': p_X2_stk,
        'pct_ou25': p_ou25_stk, 'pct_ou25_under': p_un25_stk,
        'pct_btts': p_btts_stk,
        'pct_sot75': p_sot75_stk, 'pct_sot85': p_sot85_stk,
        'pct_corners': p_corn_stk, 'pct_corners_under': p_corn_un_stk,
        'pct_cards': p_cards_stk,
        'eff_1': eff_1, 'eff_X': eff_X, 'eff_2': eff_2, 'eff_1X': eff_1X, 'eff_X2': eff_X2,
        'league_ev_thresh': league_ev_thresh, 'league_kelly': league_kelly,
        'bayesian_penalty': avg_penalty, 'all_bets': all_bets, 'best_bet': best_bet,
        'safe_parlay_leg': safe_parlay_leg,
        'same_game_parlays': same_game_parlays,
        'best_same_game_parlay': best_same_game_parlay
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

def predict_matchday_portfolio(matches_list, comp, bankroll=1000.0, df=None, models_dict=None):
    """
    Motor de Portafolio de Jornada Completa:
      1. Evalúa todos los partidos de la jornada mediante `predict_match`.
      2. Extrae el Pool de Patas Seguras de Bajo Riesgo (p >= 0.72, 1.14 <= odd <= 1.72, EV >= 0.5%)
         compuesto mayoritariamente por Dutching (1X / X2) y Favoritos Claros (1 / 2).
      3. Aplica el Filtro de Estadísticos de Orden (ordenando por probabilidad descendente) para
         construir Parlays Seguros de 2 Patas (Parlay 2-Legs) y 3 Patas (Parlay 3-Legs).
      4. Construye la Matriz de Correlación Analítica Exacta de Productos de Bernoulli y optimiza
         simultáneamente los stakes con Markowitz (MAX_DAILY_PORTFOLIO_PCT = 15%).
    """
    if df is None:
        df = load_data()
    if models_dict is None:
        models_dict = load_all_models()
    if df is None or models_dict is None or not matches_list:
        return None

    candidate_bets = []
    safe_parlay_pool = []
    match_results = []

    for m_idx, m in enumerate(matches_list):
        h_team = m['home_team']
        a_team = m['away_team']
        o1 = float(m['odds_1'])
        ox = float(m['odds_X'])
        o2 = float(m['odds_2'])
        op1 = float(m.get('open_odds_win', o1) or o1)
        opx = float(m.get('open_odds_draw', ox) or ox)
        op2 = float(m.get('open_odds_loss', o2) or o2)

        res_m = predict_match(
            h_team, a_team, comp, o1, ox, o2,
            open_odds_win=op1, open_odds_draw=opx, open_odds_loss=op2,
            extra_odds=m.get('extra_odds', {}), bankroll=bankroll,
            injuries_home=int(m.get('injuries_home', 0)),
            injuries_away=int(m.get('injuries_away', 0)),
            df=df, models_dict=models_dict,
            as_of_date=m.get('date')
        )
        if res_m is None:
            continue

        res_m['match_index'] = m_idx
        res_m['date'] = m.get('date', '')
        match_results.append(res_m)

        h_off = res_m['home_official']
        a_off = res_m['away_official']
        match_title = f"{h_off} vs {a_off}"
        league_ev_thresh = res_m['league_ev_thresh']
        league_kelly = res_m['league_kelly']
        uncert_mult = float(np.clip(res_m.get('bayesian_penalty', 0.90), 0.35, 1.0))

        # Recolectar pata segura para Parlays de Bajo Riesgo
        s_leg = res_m.get('safe_parlay_leg')
        if s_leg is not None:
            s_leg_copy = dict(s_leg)
            s_leg_copy['index'] = m_idx
            safe_parlay_pool.append(s_leg_copy)

        # Seleccionar mejor candidato individual/Dutching del partido por Ratio de Sharpe de Bernoulli
        odd_1X = 1.0 / ((1.0 / o1) + (1.0 / ox))
        odd_X2 = 1.0 / ((1.0 / ox) + (1.0 / o2))
        net_1 = 1.0 + (o1 - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_X = 1.0 + (ox - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_2 = 1.0 + (o2 - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_1X = 1.0 + (odd_1X - 1.0) * (1.0 - TAX_RETENTION_RATE)
        net_X2 = 1.0 + (odd_X2 - 1.0) * (1.0 - TAX_RETENTION_RATE)

        mkt_cands = [
            {
                'label': f"{match_title} -> 1 ({h_off})",
                'bet_type': 'single',
                'best_ev': res_m['ev_win'],
                'raw_odd': o1,
                'net_odd': net_1,
                'prob': res_m['blend_win'],
                'clv': res_m['pred_clv_win'],
                'dutch_split_str': None
            },
            {
                'label': f"{match_title} -> X (Empate)",
                'bet_type': 'single',
                'best_ev': res_m['ev_draw'],
                'raw_odd': ox,
                'net_odd': net_X,
                'prob': res_m['blend_draw'],
                'clv': res_m['pred_clv_draw'],
                'dutch_split_str': None
            },
            {
                'label': f"{match_title} -> 2 ({a_off})",
                'bet_type': 'single',
                'best_ev': res_m['ev_loss'],
                'raw_odd': o2,
                'net_odd': net_2,
                'prob': res_m['blend_loss'],
                'clv': res_m['pred_clv_loss'],
                'dutch_split_str': None
            },
            {
                'label': f"{match_title} -> Doble Oportunidad 1X ({h_off}/Empate)",
                'bet_type': 'dutching',
                'best_ev': res_m['ev_1X'],
                'raw_odd': odd_1X,
                'net_odd': net_1X,
                'prob': res_m['blend_1X'],
                'clv': res_m['pred_clv_1X'],
                'share_a': res_m['share_1_in_1X'],
                'share_b': res_m['share_X_in_1X'],
                'leg_a_name': f"1 ({h_off})",
                'leg_b_name': "X (Empate)"
            },
            {
                'label': f"{match_title} -> Doble Oportunidad X2 ({a_off}/Empate)",
                'bet_type': 'dutching',
                'best_ev': res_m['ev_X2'],
                'raw_odd': odd_X2,
                'net_odd': net_X2,
                'prob': res_m['blend_X2'],
                'clv': res_m['pred_clv_X2'],
                'share_a': res_m['share_2_in_X2'],
                'share_b': res_m['share_X_in_X2'],
                'leg_a_name': f"2 ({a_off})",
                'leg_b_name': "X (Empate)"
            }
        ]

        for c in mkt_cands:
            if c['raw_odd'] > 3.40 or c['raw_odd'] < 1.15:
                c['best_ev'] = -1.0
                continue
            min_clv = -0.0010 if c['bet_type'] == 'dutching' else 0.0001
            if c['clv'] < min_clv:
                c['best_ev'] = -1.0

        mkt_cands.sort(
            key=lambda x: (x['best_ev'] / (x['net_odd'] * np.sqrt(max(x['prob'] * (1.0 - x['prob']), 0.02)))) if x['best_ev'] > 0 else x['best_ev'],
            reverse=True
        )
        best_c = mkt_cands[0]
        eff_thresh = league_ev_thresh * 0.85 if best_c['bet_type'] == 'dutching' else league_ev_thresh

        if best_c['best_ev'] > eff_thresh:
            candidate_bets.append({
                'kelly_fraction': league_kelly * uncert_mult,
                'index': m_idx,
                'match_indices': {m_idx},
                'leg_probs': {m_idx: best_c['prob']},
                'legs': None,
                'chosen_label': best_c['label'],
                'best_ev': best_c['best_ev'],
                'raw_odd': best_c['raw_odd'],
                'net_odd': best_c['net_odd'],
                'prob': best_c['prob'],
                'bet_type': best_c['bet_type'],
                'comp': comp,
                'share_a': best_c.get('share_a'),
                'share_b': best_c.get('share_b'),
                'leg_a_name': best_c.get('leg_a_name'),
                'leg_b_name': best_c.get('leg_b_name')
            })

    # Construir Parlays Seguros de Bajo Riesgo (2 y 3 Patas)
    if len(safe_parlay_pool) >= 2:
        safe_parlay_pool.sort(key=lambda x: x['prob'], reverse=True)
        for idx_p in range(0, min(len(safe_parlay_pool) - 1, 4), 2):
            l1, l2 = safe_parlay_pool[idx_p], safe_parlay_pool[idx_p + 1]
            comb_raw_odd = l1['raw_odd'] * l2['raw_odd']
            comb_net_odd = 1.0 + (comb_raw_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
            comb_prob = l1['prob'] * l2['prob']
            comb_ev = (comb_prob * comb_net_odd) - 1.0 - (EXPECTED_CLV_DROP * 0.5)
            if comb_ev > 0.015:
                candidate_bets.append({
                    'kelly_fraction': 0.5 * (l1['kelly_fraction'] + l2['kelly_fraction']),
                    'index': l1['index'],
                    'match_indices': {l1['index'], l2['index']},
                    'leg_probs': {l1['index']: l1['prob'], l2['index']: l2['prob']},
                    'legs': [l1, l2],
                    'chosen_label': f"Parlay 2-Patas: [{l1['match_title']}: {l1['label']}] + [{l2['match_title']}: {l2['label']}]",
                    'best_ev': comb_ev,
                    'raw_odd': comb_raw_odd,
                    'net_odd': comb_net_odd,
                    'prob': comb_prob,
                    'bet_type': 'parlay',
                    'comp': comp
                })

        if len(safe_parlay_pool) >= 3:
            l1, l2, l3 = safe_parlay_pool[0], safe_parlay_pool[1], safe_parlay_pool[2]
            comb_raw_odd = l1['raw_odd'] * l2['raw_odd'] * l3['raw_odd']
            comb_net_odd = 1.0 + (comb_raw_odd - 1.0) * (1.0 - TAX_RETENTION_RATE)
            comb_prob = l1['prob'] * l2['prob'] * l3['prob']
            comb_ev = (comb_prob * comb_net_odd) - 1.0 - (EXPECTED_CLV_DROP * 0.5)
            if comb_ev > 0.020:
                candidate_bets.append({
                    'kelly_fraction': (l1['kelly_fraction'] + l2['kelly_fraction'] + l3['kelly_fraction']) / 3.0,
                    'index': l1['index'],
                    'match_indices': {l1['index'], l2['index'], l3['index']},
                    'leg_probs': {l1['index']: l1['prob'], l2['index']: l2['prob'], l3['index']: l3['prob']},
                    'legs': [l1, l2, l3],
                    'chosen_label': f"Parlay 3-Patas: [{l1['match_title']}: {l1['label']}] + [{l2['match_title']}: {l2['label']}] + [{l3['match_title']}: {l3['label']}]",
                    'best_ev': comb_ev,
                    'raw_odd': comb_raw_odd,
                    'net_odd': comb_net_odd,
                    'prob': comb_prob,
                    'bet_type': 'parlay',
                    'comp': comp
                })

    if not candidate_bets:
        return {
            'portfolio_bets': [],
            'match_results': match_results,
            'expected_profit_usd': 0.0,
            'total_staked_usd': 0.0,
            'portfolio_sharpe': 0.0
        }

    candidate_bets.sort(key=lambda x: x['best_ev'], reverse=True)
    max_liquidity = get_param_by_comp(MAX_BET_LIQUIDITY, comp, 2000.0)

    ev_vec = [b['best_ev'] for b in candidate_bets]
    odds_vec = [b['net_odd'] for b in candidate_bets]
    prob_vec = [b['prob'] for b in candidate_bets]
    penalties = [b['kelly_fraction'] / 0.015 for b in candidate_bets]
    liquidity_caps = [max_liquidity for _ in candidate_bets]

    corr_matrix = MarkowitzPortfolioOptimizer.build_bernoulli_correlation_matrix(candidate_bets)
    markowitz = MarkowitzPortfolioOptimizer(
        risk_aversion=2.0,
        max_stake_pct=MAX_STAKE_PCT,
        max_daily_portfolio_pct=0.15
    )
    m_res = markowitz.optimize_mean_variance(
        ev_vec=ev_vec,
        odds_vec=odds_vec,
        prob_vec=prob_vec,
        corr_matrix=corr_matrix,
        bankroll=bankroll,
        uncertainty_penalties=penalties,
        liquidity_caps=liquidity_caps
    )

    daily_cap_dollars = bankroll * 0.15
    raw_stakes = [float(s) if i < len(m_res['stakes']) else 0.0 for i, s in enumerate(m_res['stakes'])]
    tot_raw = sum(s for s in raw_stakes if s >= 0.1)
    scale_ratio = (daily_cap_dollars / tot_raw) if tot_raw > daily_cap_dollars else 1.0

    portfolio_bets = []
    total_staked = 0.0
    expected_profit = 0.0
    for idx_b, bet in enumerate(candidate_bets):
        st = raw_stakes[idx_b] * scale_ratio
        if st < 0.50:
            continue
        eff_odd = calculate_market_slippage(bet['raw_odd'], st, max_liquidity)
        b_copy = dict(bet)
        b_copy['stake_usd'] = float(st)
        b_copy['stake_pct'] = float(st / bankroll) if bankroll > 0 else 0.0
        b_copy['eff_odd'] = float(eff_odd)
        if bet['bet_type'] == 'dutching' and bet.get('share_a') is not None:
            b_copy['dutch_stake_a'] = float(st * bet['share_a'])
            b_copy['dutch_stake_b'] = float(st * bet['share_b'])
        portfolio_bets.append(b_copy)
        total_staked += st
        expected_profit += st * bet['best_ev']

    return {
        'portfolio_bets': portfolio_bets,
        'match_results': match_results,
        'expected_profit_usd': expected_profit,
        'total_staked_usd': total_staked,
        'portfolio_sharpe': float(m_res.get('sharpe', 0.0))
    }

def main():
    console.print(Panel.fit(
        "[bold bright_cyan]⚡ PROPHETIA2 QUANT[/bold bright_cyan] • [bold bright_yellow]INSTITUTIONAL VALUE ENGINE 10/10[/bold bright_yellow]\n"
        "[dim white]Live Pinnacle Odds Feed • Safe 2-3 Leg Parlays & Dutching • Correlated Markowitz Portfolio[/dim white]",
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
    selected_idx = None

    if live_matches:
        match_choices = [
            Choice(
                title=f"🚀 [PORTAFOLIO DE JORNADA COMPLETA] Escanear los {len(live_matches)} partidos, armar Parlays Seguros (2-3 Patas) + Dutching y optimizar con Markowitz",
                value=-2
            )
        ]
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
            
        match_choices.append(Choice(title="--> 🎯 Armar Multiapuesta / Portafolio Manual (Ingresar 2+ partidos)", value=-3))
        match_choices.append(Choice(title="--> ⚙️ Ingresar un partido individual y cuotas manualmente", value=-1))
        
        selected_idx = questionary.select(
            "⚽ Selecciona el Modo de Análisis o un Partido Próximo en Vivo:",
            choices=match_choices,
            style=CUSTOM_QUESTIONARY_STYLE
        ).ask()

        if selected_idx in (-2, -3):
            try:
                bankroll_port = float(questionary.text("💰 Bankroll actual para el Portafolio ($):", default="1000", style=CUSTOM_QUESTIONARY_STYLE).ask())
            except (ValueError, TypeError):
                bankroll_port = 1000.0

            matches_to_scan = []
            if selected_idx == -2:
                matches_to_scan = live_matches
            else:
                teams_in_comp = df[df['competition'] == comp]['team'].dropna().unique().tolist()
                team_choices = [Choice(title=get_official_team_name(t), value=t) for t in sorted(teams_in_comp)]
                while True:
                    h_t = questionary.select(f"🏠 Equipo Local (Partido #{len(matches_to_scan)+1}):", choices=team_choices, style=CUSTOM_QUESTIONARY_STYLE).ask()
                    a_t = questionary.select(f"✈️ Equipo Visitante (Partido #{len(matches_to_scan)+1}):", choices=team_choices, style=CUSTOM_QUESTIONARY_STYLE).ask()
                    if not h_t or not a_t:
                        break
                    try:
                        o_1 = float(questionary.text("Cuota Local [1]:", default="1.55", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        o_x = float(questionary.text("Cuota Empate [X]:", default="3.80", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        o_2 = float(questionary.text("Cuota Visitante [2]:", default="5.50", style=CUSTOM_QUESTIONARY_STYLE).ask())
                        matches_to_scan.append({'home_team': h_t, 'away_team': a_t, 'odds_1': o_1, 'odds_X': o_x, 'odds_2': o_2})
                    except (ValueError, TypeError):
                        break
                    if len(matches_to_scan) >= 2:
                        more = questionary.confirm("¿Deseas agregar otro partido al portafolio del día?", default=False, style=CUSTOM_QUESTIONARY_STYLE).ask()
                        if not more:
                            break

            console.print(f"\n[bold cyan]⚡ Analizando {len(matches_to_scan)} partidos con los 10 submódulos + armando Parlays Seguros (2-3 Patas) y Portafolio Markowitz...[/bold cyan]")
            port_res = predict_matchday_portfolio(matches_to_scan, comp, bankroll=bankroll_port, df=df, models_dict=models_dict)
            if not port_res or not port_res['portfolio_bets']:
                console.print(Panel("[bold yellow]No se detectaron oportunidades con ventaja matemática suficiente (EV+) en esta jornada.[/bold yellow] -> [bold]PASS / NO BET[/bold]", title="PORTAFOLIO MARKOWITZ DE JORNADA", border_style="yellow"))
                return

            t_port = Table(
                title=f"🏆 PORTAFOLIO ÓPTIMO DE MARKOWITZ (PARLAYS SEGUROS 2-3 PATAS + DUTCHING + SINGLES) - {get_formatted_comp_name(comp)}",
                show_header=True, header_style="bold white on dark_green", border_style="bright_green"
            )
            t_port.add_column("Tipo Estrategia", style="bold cyan")
            t_port.add_column("Selección / Combinación", style="bold white")
            t_port.add_column("Cuota (Efectiva)", justify="center")
            t_port.add_column("Prob. Acierto", justify="right")
            t_port.add_column("Net EV", justify="right")
            t_port.add_column("Stake Markowitz ($ / %)", justify="right")
            t_port.add_column("Desglose Operativo Dutching / Patas", style="dim yellow")

            type_labels = {
                'parlay': '🔥 PARLAY SEGURO (2-3 Patas)',
                'dutching': '🛡️ DUTCHING (1X / X2)',
                'single': '⚡ SINGLE (1X2)'
            }
            for b in port_res['portfolio_bets']:
                bt_lbl = type_labels.get(b['bet_type'], b['bet_type'])
                detail_str = "-"
                if b['bet_type'] == 'dutching' and 'dutch_stake_a' in b:
                    detail_str = f"${b['dutch_stake_a']:.2f} a {b['leg_a_name']} + ${b['dutch_stake_b']:.2f} a {b['leg_b_name']}"
                elif b['bet_type'] == 'parlay' and b.get('legs'):
                    detail_str = " × ".join([f"{lg['raw_odd']:.2f} ({lg['prob']*100:.0f}%)" for lg in b['legs']])
                t_port.add_row(
                    bt_lbl,
                    b['chosen_label'],
                    f"{b['raw_odd']:.2f} ({b['eff_odd']:.2f})",
                    f"{b['prob']*100:.1f}%",
                    f"[bold bright_green]+{b['best_ev']*100:.2f}%[/bold bright_green]",
                    f"[bold bright_yellow]${b['stake_usd']:,.2f}[/bold bright_yellow] ({b['stake_pct']*100:.2f}%)",
                    detail_str
                )
            console.print(t_port)
            console.print(Panel(
                f"[bold white]Capital Total Expuesto Hoy:[/bold white] [bold bright_yellow]${port_res['total_staked_usd']:,.2f}[/bold bright_yellow] ({port_res['total_staked_usd']/bankroll_port*100:.2f}% del Bankroll | Tope Institucional: 15.0%)\n"
                f"[bold white]Beneficio Neto Esperado Hoy (xProfit):[/bold white] [bold bright_green]+${port_res['expected_profit_usd']:,.2f}[/bold bright_green] (xYield: +{port_res['expected_profit_usd']/max(port_res['total_staked_usd'], 1e-6)*100:.2f}%)\n"
                f"[bold white]Ratio de Sharpe del Portafolio Diario:[/bold white] [bold bright_cyan]{port_res['portfolio_sharpe']:.2f}[/bold bright_cyan]",
                title="RESUMEN EJECUTIVO DE CARTERA MARKOWITZ", border_style="bright_cyan"
            ))
            return
        
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
            "¿Deseas personalizar las cuotas de Mercados Derivados (Over/Under, BTTS, Córneres, Tarjetas, Tiros al Arco)?",
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
            sot75_in = questionary.text("Cuota Over 7.5 Tiros al Arco:", default=str(extra_odds.get('sot_over75', '1.80')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            sot85_in = questionary.text("Cuota Over 8.5 Tiros al Arco:", default=str(extra_odds.get('sot_over85', '2.00')), style=CUSTOM_QUESTIONARY_STYLE).ask()
            extra_odds['ou25_over'] = float(ou25_in)
            extra_odds['ou25_under'] = float(un25_in)
            extra_odds['btts_yes'] = float(btts_in)
            extra_odds['corners_over95'] = float(corn_in)
            extra_odds['corners_under95'] = float(corn_un_in)
            extra_odds['cards_over45'] = float(card_in)
            extra_odds['sot_over75'] = float(sot75_in)
            extra_odds['sot_over85'] = float(sot85_in)
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
        f"[bold cyan]Tiros al Arco Esperados (SoT):[/bold cyan] [bold bright_yellow]{res.get('exp_sot_total', 0):.1f}[/bold bright_yellow] [dim](Local: {res.get('exp_sot_home', 0):.1f} | Visita: {res.get('exp_sot_away', 0):.1f})[/dim]\n"
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

    # Tabla 1: Mercados de Partido (1X2 & Doble Oportunidad con Desglose Exacto de Dutching)
    t1 = Table(title="1. Mercados de Partido (1X2, Doble Oportunidad Dutching & Intervalos IC 95%)", show_header=True, header_style="bold white on blue", border_style="cyan")
    t1.add_column("Mercado", style="bold white")
    t1.add_column("Odds (Efectiva)", justify="center")
    t1.add_column("Prob. Blended", justify="right")
    t1.add_column("IC 95% Bayesiano", justify="center", style="dim cyan")
    t1.add_column("CLV Proy.", justify="right")
    t1.add_column("Net EV", justify="right")
    t1.add_column("Stake Kelly Multi ($ / %)", justify="right")

    dutch_1x_note = f" [dim yellow](${res['stake_1X_leg1']:.2f} a [1] + ${res['stake_1X_legX']:.2f} a [X])[/dim yellow]" if res['stake_1X'] > 0 else ""
    dutch_x2_note = f" [dim yellow](${res['stake_X2_leg2']:.2f} a [2] + ${res['stake_X2_legX']:.2f} a [X])[/dim yellow]" if res['stake_X2'] > 0 else ""

    t1.add_row(f"1 ([bold bright_cyan]{home_off}[/bold bright_cyan])", f"{odds_1:.2f} ({res['eff_1']:.2f})", f"{res['blend_win']*100:.1f}%", f"[{res['ci_win_low']*100:.1f}% - {res['ci_win_high']*100:.1f}%]", f"{res['pred_clv_win']*100:+.2f}%", c_ev(res['ev_win']), c_stake(res['stake_win'], res['pct_win']))
    t1.add_row("X ([bold yellow]Empate[/bold yellow])", f"{odds_X:.2f} ({res['eff_X']:.2f})", f"{res['blend_draw']*100:.1f}%", f"[{res['ci_draw_low']*100:.1f}% - {res['ci_draw_high']*100:.1f}%]", f"{res['pred_clv_draw']*100:+.2f}%", c_ev(res['ev_draw']), c_stake(res['stake_draw'], res['pct_draw']))
    t1.add_row(f"2 ([bold bright_magenta]{away_off}[/bold bright_magenta])", f"{odds_2:.2f} ({res['eff_2']:.2f})", f"{res['blend_loss']*100:.1f}%", f"[{res['ci_loss_low']*100:.1f}% - {res['ci_loss_high']*100:.1f}%]", f"{res['pred_clv_loss']*100:+.2f}%", c_ev(res['ev_loss']), c_stake(res['stake_loss'], res['pct_loss']))
    t1.add_row(f"1X / Dutching Local-Empate{dutch_1x_note}", f"{1.0/(1.0/odds_1+1.0/odds_X):.2f} ({res['eff_1X']:.2f})", f"{res['blend_1X']*100:.1f}%", f"[{res['ci_1X_low']*100:.1f}% - {res['ci_1X_high']*100:.1f}%]", f"{res['pred_clv_1X']*100:+.2f}%", c_ev(res['ev_1X']), c_stake(res['stake_1X'], res['pct_1X']))
    t1.add_row(f"X2 / Dutching Visita-Empate{dutch_x2_note}", f"{1.0/(1.0/odds_X+1.0/odds_2):.2f} ({res['eff_X2']:.2f})", f"{res['blend_X2']*100:.1f}%", f"[{res['ci_X2_low']*100:.1f}% - {res['ci_X2_high']*100:.1f}%]", f"{res['pred_clv_X2']*100:+.2f}%", c_ev(res['ev_X2']), c_stake(res['stake_X2'], res['pct_X2']))
    console.print(t1)

    # Tabla 2: Mercados Derivados (Goles O/U, BTTS, Tiros al Arco, Córneres & Tarjetas)
    t2 = Table(title="2. Mercados Derivados (Goles O/U, BTTS, Tiros al Arco, Córneres & Tarjetas)", show_header=True, header_style="bold white on magenta", border_style="magenta")
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
    odd_sot75 = float(res.get('odd_sot75', 1.80))
    odd_sot85 = float(res.get('odd_sot85', 2.00))

    t2.add_row("Over 2.5 Goles Totales", f"{odd_ou25:.2f}", f"{res['prob_over25_goals']*100:.1f}%", c_ev(res['ev_ou25']), c_stake(res['stake_ou25'], res['pct_ou25']))
    t2.add_row("Under 2.5 Goles Totales", f"{odd_ou25_under:.2f}", f"{res['prob_under25_goals']*100:.1f}%", c_ev(res['ev_ou25_under']), c_stake(res['stake_ou25_under'], res['pct_ou25_under']))
    t2.add_row("Both Teams to Score (BTTS Yes)", f"{odd_btts:.2f}", f"{res['prob_btts_yes']*100:.1f}%", c_ev(res['ev_btts']), c_stake(res['stake_btts'], res['pct_btts']))
    t2.add_row("Over 7.5 Tiros al Arco (SoT)", f"{odd_sot75:.2f}", f"{res['prob_over75_sot']*100:.1f}%", c_ev(res['ev_sot75']), c_stake(res.get('stake_sot75', 0.0), res.get('pct_sot75', 0.0)))
    t2.add_row("Over 8.5 Tiros al Arco (SoT)", f"{odd_sot85:.2f}", f"{res['prob_over85_sot']*100:.1f}%", c_ev(res['ev_sot85']), c_stake(res.get('stake_sot85', 0.0), res.get('pct_sot85', 0.0)))
    t2.add_row("Over 9.5 Córneres Totales", f"{odd_corn:.2f}", f"{res['prob_over95_corners']*100:.1f}%", c_ev(res['ev_corners']), c_stake(res['stake_corners'], res['pct_corners']))
    t2.add_row("Under 9.5 Córneres Totales", f"{odd_corn_un:.2f}", f"{res['prob_under95_corners']*100:.1f}%", c_ev(res['ev_corners_under']), c_stake(res['stake_corners_under'], res['pct_corners_under']))
    t2.add_row("Over 4.5 Tarjetas Totales", f"{odd_card:.2f}", f"{res['prob_over45_cards']*100:.1f}%", c_ev(res['ev_cards']), c_stake(res['stake_cards'], res['pct_cards']))
    console.print(t2)

    # Tabla 3: Parlays de 2 Patas del Mismo Partido (Bet Builder: Dutching / 1X2 + Mercado Derivado)
    sgp_list = res.get('same_game_parlays', [])
    if sgp_list:
        t3 = Table(
            title="3. Parlays de 2 Patas del Partido (Bet Builder: Dutching / 1X2 + Mercado Derivado)",
            show_header=True, header_style="bold white on dark_green", border_style="bright_green"
        )
        t3.add_column("Combinación 2-Patas (Pata 1 + Pata 2)", style="bold white")
        t3.add_column("Cuotas (P1 x P2 -> SGP)", justify="center")
        t3.add_column("Prob. Conjunta", justify="right")
        t3.add_column("Corr. Cópula", justify="center", style="dim cyan")
        t3.add_column("Net EV", justify="right")
        t3.add_column("Stake Kelly SGP ($ / %)", justify="right")

        for sgp in sgp_list[:5]:
            t3.add_row(
                sgp['label'],
                f"{sgp['odd_a']:.2f} x {sgp['odd_b']:.2f} -> [bold bright_yellow]{sgp['sgp_odd']:.2f}[/bold bright_yellow]",
                f"{sgp['prob_joint']*100:.1f}%",
                f"{sgp['corr']:+.2f}",
                c_ev(sgp['ev']),
                c_stake(sgp['stake'], sgp['pct'])
            )
        console.print(t3)

    if res.get('safe_parlay_leg'):
        s_leg = res['safe_parlay_leg']
        console.print(Panel(
            f"[bold bright_cyan]🔗 PATA SEGURA ELEGIBLE PARA PARLAY DE BAJO RIESGO (2-3 PATAS):[/bold bright_cyan]\n"
            f"Selección: [bold white]{s_leg['label']}[/bold white] @ [bold bright_yellow]{s_leg['raw_odd']:.2f}[/bold bright_yellow] | "
            f"Probabilidad Stacker L2: [bold bright_green]{s_leg['prob']*100:.1f}%[/bold bright_green] | Edge Individual: [bold bright_green]+{s_leg['ev']*100:.2f}%[/bold bright_green]\n"
            f"[dim white]Tip Cuantitativo: Combina esta pata con otra selección segura (p >= 72%) de la jornada usando el modo 'Portafolio de Jornada Completa' para multiplicar el Yield (+8.72% histórico en Parlays de 2-3 patas).[/dim white]",
            title="ELEGIBILIDAD MULTIAPUESTA / PARLAY SEGURO", border_style="cyan"
        ))

    best_bet_name, best_ev, best_stake, best_pct, best_raw_odd, best_eff_odd, _ = res['best_bet']
    best_sgp = res.get('best_same_game_parlay')

    console.print("\n[bold]=== RECOMENDACIÓN DE STAKING MULTI-MERCADO INSTITUCIONAL ===[/bold]")
    if best_ev > res['league_ev_thresh'] or (best_sgp and best_sgp['ev'] > res['league_ev_thresh']):
        if best_stake <= 0 and (not best_sgp or best_sgp['stake'] <= 0):
            console.print(f"[yellow]El mejor EV ({best_bet_name} +{best_ev*100:.2f}%) supera el umbral de la liga ({res['league_ev_thresh']*100:.2f}%), pero el stake Kelly Multi-Activo es minúsculo (< 0.01%).[/yellow] -> [bold]PASS / NO BET[/bold]")
        else:
            dutch_extra_line = ""
            if "1X" in best_bet_name and res['stake_1X'] > 0:
                dutch_extra_line = f"\nDesglose Operativo Dutching: [bold bright_cyan]${res['stake_1X_leg1']:,.2f} a Local [1][/bold bright_cyan] + [bold yellow]${res['stake_1X_legX']:,.2f} a Empate [X][/bold yellow]"
            elif "X2" in best_bet_name and res['stake_X2'] > 0:
                dutch_extra_line = f"\nDesglose Operativo Dutching: [bold bright_magenta]${res['stake_X2_leg2']:,.2f} a Visitante [2][/bold bright_magenta] + [bold yellow]${res['stake_X2_legX']:,.2f} a Empate [X][/bold yellow]"

            # Listar posiciones activas adicionales en mercados derivados / principales
            other_active = [
                b for b in res.get('all_bets', [])
                if b[0] != best_bet_name and b[2] > 0.50 and b[1] > 0.0
            ]
            other_lines = ""
            if other_active:
                other_lines = "\n\n[bold bright_cyan]📊 POSICIONES ACTIVAS ADICIONALES (MERCADOS DERIVADOS / PORTAFOLIO MARKOWITZ):[/bold bright_cyan]"
                for ob_name, ob_ev, ob_stk, ob_pct, ob_odd, _, ob_prob in other_active:
                    other_lines += f"\n  • [bold white]{ob_name}[/bold white] @ [bold bright_yellow]{ob_odd:.2f}[/bold bright_yellow] (Prob: {ob_prob*100:.1f}% | EV: [bold bright_green]+{ob_ev*100:.1f}%[/bold bright_green]) -> Stake: [bold bright_yellow]${ob_stk:,.2f}[/bold bright_yellow] ({ob_pct*100:.2f}%)"

            # Recomendar el mejor Parlay de 2 Patas (Dutching / 1X2 + Mercado Derivado)
            sgp_block = ""
            if best_sgp and best_sgp['ev'] > 0.0 and best_sgp['stake'] > 0.50:
                sgp_dutch_split = ""
                if best_sgp['kind_a'] == 'dutching_1X':
                    s1 = best_sgp['stake'] * res['share_1_in_1X']
                    sx = best_sgp['stake'] * res['share_X_in_1X']
                    sgp_dutch_split = f"\n  Desglose Dutching 2-Patas: [bold bright_cyan]${s1:,.2f} a [1 + {best_sgp['leg_b']}][/bold bright_cyan] + [bold yellow]${sx:,.2f} a [X + {best_sgp['leg_b']}][/bold yellow]"
                elif best_sgp['kind_a'] == 'dutching_X2':
                    s2 = best_sgp['stake'] * res['share_2_in_X2']
                    sx = best_sgp['stake'] * res['share_X_in_X2']
                    sgp_dutch_split = f"\n  Desglose Dutching 2-Patas: [bold bright_magenta]${s2:,.2f} a [2 + {best_sgp['leg_b']}][/bold bright_magenta] + [bold yellow]${sx:,.2f} a [X + {best_sgp['leg_b']}][/bold yellow]"

                # Si existe también una opción de Parlay con la otra Doble Oportunidad positiva, mostrarla como alternativa
                alt_sgp_line = ""
                for cand_sgp in sgp_list[1:]:
                    if cand_sgp['ev'] > 0.0 and cand_sgp['kind_a'] != best_sgp['kind_a'] and cand_sgp['kind_a'].startswith('dutching'):
                        alt_sgp_line = f"\n  Alternativa Parlay Dutching: [bold white]{cand_sgp['label']}[/bold white] @ [bold bright_yellow]{cand_sgp['sgp_odd']:.2f}[/bold bright_yellow] (Prob Conjunta: {cand_sgp['prob_joint']*100:.1f}% | EV: [bold bright_green]+{cand_sgp['ev']*100:.1f}%[/bold bright_green] | Stake: [bold bright_yellow]${cand_sgp['stake']:,.2f}[/bold bright_yellow])"
                        break

                sgp_block = (
                    f"\n\n[bold bright_magenta]🔥 RECOMENDACIÓN PARLAY 2-PATAS DEL PARTIDO (DUTCHING / 1X2 + DERIVADO):[/bold bright_magenta]\n"
                    f"  Combinación: [bold white]{best_sgp['label']}[/bold white] @ [bold bright_yellow]{best_sgp['sgp_odd']:.2f}[/bold bright_yellow] "
                    f"([dim]{best_sgp['odd_a']:.2f} x {best_sgp['odd_b']:.2f}[/dim])\n"
                    f"  Prob. Conjunta Cópula: [bold bright_green]{best_sgp['prob_joint']*100:.1f}%[/bold bright_green] | "
                    f"Net EV Parlay: [bold bright_green]+{best_sgp['ev']*100:.2f}%[/bold bright_green] | "
                    f"Stake Kelly SGP: [bold bright_yellow]${best_sgp['stake']:,.2f}[/bold bright_yellow] ({best_sgp['pct']*100:.2f}%)"
                    f"{sgp_dutch_split}{alt_sgp_line}"
                )

            rec = Panel(
                f"[bold bright_green]🚀 MEJOR OPORTUNIDAD VALUE DETECTADA (RATIO DE SHARPE ÓPTIMO)[/bold bright_green]\n\n"
                f"Mercado Principal Seleccionado: [bold white]{best_bet_name}[/bold white] @ [bold bright_yellow]{best_raw_odd:.2f}[/bold bright_yellow] (Efectiva: {best_eff_odd:.2f})\n"
                f"Net EV Proyectado: [bold bright_green]+{best_ev*100:.2f}%[/bold bright_green] (Umbral Liga {get_formatted_comp_name(comp)}: {res['league_ev_thresh']*100:.2f}%)\n"
                f"Stake Kelly Multi-Activo: [bold bright_yellow]${best_stake:,.2f}[/bold bright_yellow] ([bold white]{best_pct*100:.2f}%[/bold white] del bankroll){dutch_extra_line}"
                f"{other_lines}{sgp_block}",
                title="SISTEMA DE STAKING INSTITUCIONAL MULTI-MERCADO & PARLAY 2-PATAS", border_style="bright_green"
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
