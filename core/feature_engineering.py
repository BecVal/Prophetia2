import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict

# Configurar logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'feature_engineering')

INTERIM_DATASET_PATH = '../data/interim/intermediate_dataset.parquet'
OUTPUT_PATH = '../data/processed/matches_dataset.parquet'


def add_ema_features(df, spans=[3, 5, 10]):
    """
    Calcula Promedios Móviles Exponenciales (EMA) históricos evitando Data Leakage.
    Incluye métricas cuantitativas avanzadas como xG Share (Pythagorean Ratio),
    Form Ratios (EMA3/EMA10), Time-Decay por Días de Calendario Reales (Half-Life 30d)
    y métricas de suerte/regresión a la media en definición y defensa.
    """
    logger.info(f"Calculando EMA históricos y métricas avanzadas de xG (spans={spans})...")
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)

    # 1. Crear xG Share (Pythagorean Ratio para fútbol)
    if 'xg_created' in df.columns and 'xg_conceded' in df.columns:
        df['xg_share'] = df['xg_created'] / (df['xg_created'] + df['xg_conceded'] + 1e-6)

    stats_cols = [
        'xg_created',
        'xg_conceded',
        'xg_share',
        'shots_total',
        'shots_on_target',
        'passes_total',
        'passes_completed',
        'pass_accuracy',
        'possession_pct',
        'crosses',
        'corners',
        'through_balls',
        'key_passes',
        'dribbles_completed',
        'pressures',
        'interceptions',
        'clearances',
        'blocks',
        'ball_recoveries',
        'actions_under_pressure',
        'fouls_committed',
        'fouls_won',
        'yellow_cards',
        'red_cards',
        'aerials_won'
    ]

    roll_cols = [c for c in stats_cols if c in df.columns]

    # Agrupar por equipo una sola vez para optimizar rendimiento de Pandas
    grouped_team = df.groupby('team', sort=False)
    
    new_cols_dict = {}
    for span in spans:
        ema_block = grouped_team[roll_cols].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )
        for col in roll_cols:
            new_cols_dict[f'{col}_ema{span}'] = ema_block[col]

    df = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)

    # 2. Form Ratios: Racha a Corto Plazo vs Baseline a Largo Plazo (EMA3 vs EMA10)
    if 'xg_created_ema3' in df.columns and 'xg_created_ema10' in df.columns:
        df['xg_form_ratio'] = (df['xg_created_ema3'] + 1e-4) / (df['xg_created_ema10'] + 1e-4)
    if 'xg_conceded_ema3' in df.columns and 'xg_conceded_ema10' in df.columns:
        df['defensive_form_ratio'] = (df['xg_conceded_ema3'] + 1e-4) / (df['xg_conceded_ema10'] + 1e-4)

    # 3. Decaimiento Exponencial por Días de Calendario Reales (Calendar-Day Time-Decay, Half-Life = 30 días)
    if 'xg_created' in df.columns and 'xg_conceded' in df.columns and 'match_date' in df.columns:
        halflife_days = 30.0
        df['calendar_decay_xg_created'] = df.groupby('team')['xg_created'].transform(
            lambda x: x.shift(1).ewm(halflife=f'{halflife_days}D', times=df.loc[x.index, 'match_date']).mean()
        ).fillna(df.get('xg_created_ema3', 0.0))
        
        df['calendar_decay_xg_conceded'] = df.groupby('team')['xg_conceded'].transform(
            lambda x: x.shift(1).ewm(halflife=f'{halflife_days}D', times=df.loc[x.index, 'match_date']).mean()
        ).fillna(df.get('xg_conceded_ema3', 0.0))

    # 4. Continuidad de Alineación (Retention Rate del 11 titular respecto al partido anterior)
    if 'starting_xi' in df.columns:
        def calc_lineup_overlap(curr, prev):
            if not curr or not prev or pd.isna(curr) or pd.isna(prev):
                return 1.0
            try:
                s1 = set(curr) if isinstance(curr, (list, tuple, set)) else set(str(curr).split(','))
                s2 = set(prev) if isinstance(prev, (list, tuple, set)) else set(str(prev).split(','))
                if not s1 or not s2: return 1.0
                return len(s1.intersection(s2)) / float(max(len(s1), 1))
            except Exception:
                return 1.0

        prev_xi = df.groupby('team')['starting_xi'].shift(1)
        df['lineup_continuity_pct'] = [calc_lineup_overlap(c, p) for c, p in zip(df['starting_xi'], prev_xi)]
    else:
        # Indicador base de estabilidad de la plantilla
        df['lineup_continuity_pct'] = 1.0

    # 5. Métricas de Suerte / Regresión a la media en Definición y Defensa
    if 'goals_scored' in df.columns and 'xg_created_ema3' in df.columns:
        goals_scored_ema3 = df.groupby('team')['goals_scored'].transform(
            lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
        )
        df['xg_finishing_luck'] = goals_scored_ema3 - df['xg_created_ema3']

    if 'goals_conceded' in df.columns and 'xg_conceded_ema3' in df.columns:
        goals_conceded_ema3 = df.groupby('team')['goals_conceded'].transform(
            lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
        )
        df['xg_defensive_luck'] = df['xg_conceded_ema3'] - goals_conceded_ema3

    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def calculate_expected_goals(att_rating, def_rating, is_home=True):
    """
    Calcula los goles esperados (lambda Poisson) mediante una relación log-lineal.
    Evita la distorsión de escala base 10 del ajedrez en favor del modelo Poisson.
    """
    base_goals = 1.45 if is_home else 1.15
    # Escala log-lineal acotada para estabilidad del modelo
    rating_diff = np.clip((att_rating - def_rating) / 400.0, -2.0, 2.0)
    return base_goals * np.exp(rating_diff)


def update_rating(rating, actual, expected, k_factor=20.0):
    """
    Actualiza el rating según la diferencia entre el valor real y el esperado.
    Firma estandarizada: (rating, actual, expected).
    """
    diff = np.clip(actual - expected, -3.0, 3.0)
    return rating + k_factor * diff


def calculate_expected_score(rating_a, rating_b):
    """Calcula la probabilidad esperada de victoria para el equipo A frente al B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating, actual_score, expected_score, k_factor=30.0):
    """Actualiza el ELO clásico según el resultado."""
    return rating + k_factor * (actual_score - expected_score)


def add_elo_ratings(df, home_advantage=80.0):
    """
    Calcula Ratings de Ataque/Defensa (Poisson) y ELO Clásico con ventaja de localía.
    Optimizado vectorialmente mediante arreglos de NumPy para máxima aceleración.
    """
    logger.info("Calculando Ratings de Ataque/Defensa y ELO Clásico (Optimizado)...")
    start_time = time.time()
    df = df.sort_values('match_date').reset_index(drop=True)
    
    n_rows = len(df)
    match_ids = df['match_id'].values
    teams = df['team'].values
    is_home_arr = df['is_home'].values
    goals_arr = df['goals_scored'].values if 'goals_scored' in df.columns else np.zeros(n_rows)
    outcomes = df['outcome'].values if 'outcome' in df.columns else np.zeros(n_rows)
    
    # Agrupar índices por partido eficientemente
    match_to_indices = defaultdict(list)
    for idx, m_id in enumerate(match_ids):
        match_to_indices[m_id].append(idx)
        
    att_dict = {}
    def_dict = {}
    elo_dict = {}
    
    res_att = np.zeros(n_rows)
    res_def = np.zeros(n_rows)
    res_opp_att = np.zeros(n_rows)
    res_opp_def = np.zeros(n_rows)
    res_elo = np.zeros(n_rows)
    res_opp_elo = np.zeros(n_rows)
    res_elo_diff = np.zeros(n_rows)
    
    for m_id, indices in match_to_indices.items():
        if len(indices) != 2:
            continue
            
        idx1, idx2 = indices[0], indices[1]
        
        if is_home_arr[idx1] == 1:
            home_idx, away_idx = idx1, idx2
        else:
            home_idx, away_idx = idx2, idx1
            
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        # Inicialización de Ratings
        if home_team not in att_dict: att_dict[home_team] = 1000.0
        if home_team not in def_dict: def_dict[home_team] = 1000.0
        if home_team not in elo_dict: elo_dict[home_team] = 1500.0
        
        if away_team not in att_dict: att_dict[away_team] = 1000.0
        if away_team not in def_dict: def_dict[away_team] = 1000.0
        if away_team not in elo_dict: elo_dict[away_team] = 1500.0
        
        home_att_pre = att_dict[home_team]
        home_def_pre = def_dict[home_team]
        away_att_pre = att_dict[away_team]
        away_def_pre = def_dict[away_team]
        
        home_elo_pre = elo_dict[home_team]
        away_elo_pre = elo_dict[away_team]
        
        goals_home = goals_arr[home_idx]
        goals_away = goals_arr[away_idx]
        
        # 1. Goles Esperados y Actualización Ataque/Defensa
        exp_goals_home = calculate_expected_goals(home_att_pre, away_def_pre, is_home=True)
        exp_goals_away = calculate_expected_goals(away_att_pre, home_def_pre, is_home=False)
        
        # Ataque: sube si anotas más goles que los esperados
        att_dict[home_team] = update_rating(home_att_pre, actual=goals_home, expected=exp_goals_home)
        att_dict[away_team] = update_rating(away_att_pre, actual=goals_away, expected=exp_goals_away)
        
        # Defensa: sube si encajas MENOS goles de los esperados
        def_dict[home_team] = update_rating(home_def_pre, actual=exp_goals_away, expected=goals_away)
        def_dict[away_team] = update_rating(away_def_pre, actual=exp_goals_home, expected=goals_home)
        
        # 2. ELO Clásico incorporando Ventaja de Localía
        outcome_home = outcomes[home_idx] # 1, 0, -1
        if outcome_home == 1:
            score_home, score_away = 1.0, 0.0
        elif outcome_home == -1:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5
            
        exp_elo_home = calculate_expected_score(home_elo_pre + home_advantage, away_elo_pre)
        exp_elo_away = 1.0 - exp_elo_home
        
        elo_dict[home_team] = update_elo(home_elo_pre, score_home, exp_elo_home)
        elo_dict[away_team] = update_elo(away_elo_pre, score_away, exp_elo_away)
        
        # Almacenar resultados pre-partido
        res_att[home_idx] = home_att_pre
        res_def[home_idx] = home_def_pre
        res_opp_att[home_idx] = away_att_pre
        res_opp_def[home_idx] = away_def_pre
        res_elo[home_idx] = home_elo_pre
        res_opp_elo[home_idx] = away_elo_pre
        res_elo_diff[home_idx] = (home_elo_pre + home_advantage) - away_elo_pre
        
        res_att[away_idx] = away_att_pre
        res_def[away_idx] = away_def_pre
        res_opp_att[away_idx] = home_att_pre
        res_opp_def[away_idx] = home_def_pre
        res_elo[away_idx] = away_elo_pre
        res_opp_elo[away_idx] = home_elo_pre
        res_elo_diff[away_idx] = away_elo_pre - (home_elo_pre + home_advantage)

    df['team_att_rating'] = res_att
    df['team_def_rating'] = res_def
    df['opp_att_rating'] = res_opp_att
    df['opp_def_rating'] = res_opp_def
    df['team_elo'] = res_elo
    df['opp_elo'] = res_opp_elo
    df['elo_diff'] = res_elo_diff
        
    df = df.copy()
    elapsed = time.time() - start_time
    logger.info(f"Cálculo de ELO finalizado en {elapsed:.2f} segundos.")
    return df


def add_contextual_features(df):
    """
    Calcula variables contextuales: descanso, índice de fatiga acotado, rachas,
    momentum MACD de xG, volatilidad y Strength of Schedule (SOS).
    """
    logger.info("Calculando variables contextuales (Días de descanso, SOS y Momentum)...")
    start_time = time.time()

    # 1. Días de descanso y Modelo de Fatiga Acotado (recuperación realista)
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    df['rest_days'] = df.groupby('team')['match_date'].diff().dt.days.fillna(7.0)
    df['rest_days'] = df['rest_days'].clip(upper=21.0)
    
    # Fatiga acotada: cae cuadráticamente hasta 0 cuando rest_days >= 5
    df['fatigue_index'] = np.clip(1.0 - (df['rest_days'] / 5.0), 0.0, 1.0) ** 2.0

    # 2. Inercia (Rachas y MACD de xG)
    df['is_win'] = (df['outcome'] == 1).astype(int)
    df['is_loss'] = (df['outcome'] == -1).astype(int)
    
    df['win_streak_3'] = df.groupby('team')['is_win'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['loss_streak_3'] = df.groupby('team')['is_loss'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    
    if 'xg_created' in df.columns and 'xg_conceded' in df.columns:
        df['xg_diff_raw'] = df['xg_created'] - df['xg_conceded']
    else:
        df['xg_diff_raw'] = 0.0
    
    # Momentum MACD (Diferencia de EMA Corto vs EMA Largo)
    xg_ema3 = df.groupby('team')['xg_diff_raw'].transform(lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
    xg_ema10 = df.groupby('team')['xg_diff_raw'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    df['xg_momentum_macd'] = (xg_ema3 - xg_ema10).fillna(0)
    
    # Volatilidad de rendimiento
    df['xg_volatility_5'] = df.groupby('team')['xg_diff_raw'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)
    
    # Rachas de visitante
    df['is_away'] = (df['is_home'] == 0).astype(int)
    df['away_streak'] = df.groupby('team')['is_away'].transform(
        lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumsum()
    ).fillna(0)
    df = df.drop(columns=['is_away'])
    
    # Strength of Schedule (SOS)
    if 'opp_elo' in df.columns:
        df['schedule_strength_5'] = df.groupby('team')['opp_elo'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(1500.0)
    
    df = df.drop(columns=['is_win', 'is_loss', 'xg_diff_raw'])

    # 3. Cruzar variables del oponente de forma segura
    ema_cols = [c for c in df.columns if '_ema' in c]
    extra_opp_cols = [
        'rest_days', 'fatigue_index', 'win_streak_3', 'loss_streak_3', 
        'xg_momentum_macd', 'xg_volatility_5', 'away_streak',
        'xg_form_ratio', 'defensive_form_ratio', 
        'calendar_decay_xg_created', 'calendar_decay_xg_conceded',
        'lineup_continuity_pct'
    ]
    if 'schedule_strength_5' in df.columns:
        extra_opp_cols.append('schedule_strength_5')
    if 'xg_share' in df.columns:
        extra_opp_cols.append('xg_share')
        
    opp_cols = list(set(ema_cols + extra_opp_cols))
    opp_cols = [c for c in opp_cols if c in df.columns]
    
    opp_df = df[['team', 'match_date'] + opp_cols].copy()
    opp_rename = {c: f"opp_{c}" for c in opp_cols}
    opp_rename['team'] = 'opponent'
    opp_df = opp_df.rename(columns=opp_rename)

    # Merge limpio sin registros duplicados
    df = pd.merge(df, opp_df, on=['opponent', 'match_date'], how='left')

    for c in opp_rename.values():
        if c != 'opponent' and c in df.columns:
            df[c] = df[c].fillna(0.0)

    # 4. Métricas Relativas
    df['rest_diff'] = df['rest_days'] - df['opp_rest_days']
    df['fatigue_diff'] = df['fatigue_index'] - df['opp_fatigue_index']
    
    if 'schedule_strength_5' in df.columns and 'opp_schedule_strength_5' in df.columns:
        df['sos_diff'] = df['schedule_strength_5'] - df['opp_schedule_strength_5']
    
    if 'xg_created_ema3' in df.columns and 'opp_xg_conceded_ema3' in df.columns:
        df['relative_attack_strength'] = df['xg_created_ema3'] - df['opp_xg_conceded_ema3']
            
    if 'xg_volatility_5' in df.columns and 'opp_xg_volatility_5' in df.columns:
        df['volatility_diff'] = df['xg_volatility_5'] - df['opp_xg_volatility_5']
        
    if 'lineup_continuity_pct' in df.columns and 'opp_lineup_continuity_pct' in df.columns:
        df['lineup_continuity_diff'] = df['lineup_continuity_pct'] - df['opp_lineup_continuity_pct']

    df = df.copy().sort_values('match_date').reset_index(drop=True)
    elapsed = time.time() - start_time
    logger.info(f"Variables contextuales procesadas en {elapsed:.2f} segundos.")
    return df


def add_h2h_features(df):
    """
    Calcula variables H2H (enfrentamientos directos) con priores estadísticos
    para evitar sesgos en los primeros encuentros.
    """
    logger.info("Calculando variables H2H (enfrentamientos directos)...")
    df = df.sort_values('match_date').reset_index(drop=True)
    
    if 'outcome' not in df.columns:
        return df
        
    df['h2h_points'] = df['outcome'].map({1: 3, 0: 1, -1: 0})
    df['h2h_games_played'] = df.groupby(['team', 'opponent']).cumcount()
    
    df['h2h_points_last_5'] = df.groupby(['team', 'opponent'])['h2h_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0.0)
    
    # Priores bayesianos por defecto para historial sin partidos previos (33% victoria, 28% empate)
    df['win_shifted'] = df.groupby(['team', 'opponent'])['outcome'].transform(lambda x: (x == 1).astype(float).shift(1))
    df['h2h_win_rate_hist'] = df.groupby(['team', 'opponent'])['win_shifted'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    ).fillna(0.33)
    
    df['draw_shifted'] = df.groupby(['team', 'opponent'])['outcome'].transform(lambda x: (x == 0).astype(float).shift(1))
    df['h2h_draw_rate_hist'] = df.groupby(['team', 'opponent'])['draw_shifted'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    ).fillna(0.28)
    
    df = df.drop(columns=['h2h_points', 'win_shifted', 'draw_shifted'])
    return df


def add_advanced_fatigue(df):
    """Calcula resaca europea y cambio entre competiciones."""
    logger.info("Calculando Fatiga Avanzada (cambio de competiciones)...")
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    
    if 'competition' in df.columns:
        df['prev_competition'] = df.groupby('team')['competition'].shift(1)
        df['is_european_hangover'] = (
            (df['competition'] != df['prev_competition']) & 
            (df['prev_competition'].notna()) & 
            (df['rest_days'] <= 4)
        ).astype(int)
        df = df.drop(columns=['prev_competition'])
    else:
        df['is_european_hangover'] = 0
        
    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def add_squad_value_features(df):
    """
    Integra valores de mercado de Transfermarkt imputando con medianas de liga/temporada
    para evitar Data Leakage y contaminación entre equipos.
    """
    logger.info("Integrando Valores de Mercado de Transfermarkt...")
    
    tm_path = '../data/raw/transfermarkt_squad_values.parquet'
    if not os.path.exists(tm_path):
        logger.warning(f"No se encontró el archivo de Transfermarkt en {tm_path}. Saltando variable.")
        return df
        
    tm_df = pd.read_parquet(tm_path, engine='fastparquet')
    
    df['season_year'] = df['match_date'].dt.year
    df.loc[df['match_date'].dt.month < 7, 'season_year'] -= 1
    
    df = pd.merge(
        df, 
        tm_df[['season_year', 'team', 'squad_value_millions']], 
        on=['season_year', 'team'], 
        how='left'
    ).rename(columns={'squad_value_millions': 'team_squad_value'})
    
    tm_df_opp = tm_df[['season_year', 'team', 'squad_value_millions']].rename(
        columns={'team': 'opponent', 'squad_value_millions': 'opp_squad_value'}
    )
    df = pd.merge(
        df, 
        tm_df_opp, 
        on=['season_year', 'opponent'], 
        how='left'
    )
    
    # Imputación por mediana de temporada sin contaminación cruzada
    season_medians = df.groupby('season_year')['team_squad_value'].transform('median').fillna(15.0)
    df['team_squad_value'] = df['team_squad_value'].fillna(season_medians)
    df['opp_squad_value'] = df['opp_squad_value'].fillna(season_medians)
    
    df['squad_value_diff'] = df['team_squad_value'] - df['opp_squad_value']
    df = df.drop(columns=['season_year']).copy().sort_values('match_date').reset_index(drop=True)
    return df


def build_processed_dataset():
    if not os.path.exists(INTERIM_DATASET_PATH):
        logger.error(
            f"No se encontró el dataset intermedio en: {INTERIM_DATASET_PATH}. "
            "Por favor, ejecuta primero core/data_adapter.py para generarlo.")
        return

    logger.info(f"Cargando dataset intermedio desde {INTERIM_DATASET_PATH}...")
    final_df = pd.read_parquet(INTERIM_DATASET_PATH, engine='fastparquet')
    
    logger.info(f"Procesando {len(final_df)} filas para extracción de características avanzadas...")

    final_df['match_date'] = pd.to_datetime(final_df['match_date'])
    final_df = final_df.sort_values('match_date').reset_index(drop=True)

    final_df = add_ema_features(final_df, spans=[3, 5])
    final_df = add_elo_ratings(final_df)
    final_df = add_contextual_features(final_df)
    final_df = add_h2h_features(final_df)
    final_df = add_advanced_fatigue(final_df)
    final_df = add_squad_value_features(final_df)

    processed_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(processed_dir, exist_ok=True)

    final_df.to_parquet(OUTPUT_PATH, engine='fastparquet', index=False)
    logger.info(f"Dataset de entrenamiento avanzado guardado exitosamente en: {OUTPUT_PATH}")
    logger.info(f"Estructura del dataset final: {final_df.shape}")

    # --- Auditoría Final ---
    logger.info("=== AUDITORÍA DE DATOS Y FEATURE ENGINEERING ===")
    logger.info(f"Total de Filas: {len(final_df)}")
    logger.info(f"Total de Columnas: {len(final_df.columns)}")
    
    nan_counts = final_df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0].sort_values(ascending=False)
    
    if not cols_with_nans.empty:
        logger.info(f"Variables con valores nulos (Top 10):\n{cols_with_nans.head(10).to_string()}")
    else:
        logger.info("No hay valores nulos en el dataset.")
        
    if 'xg_created_ema3' in final_df.columns:
        logger.info(f"NaNs esperados en xg_created_ema3: {final_df['xg_created_ema3'].isna().sum()}")
    
    logger.info("================================================")

    cols_to_show = [
        'team',
        'is_home',
        'xg_created',
        'xg_created_ema3',
        'xg_share_ema3',
        'team_att_rating',
        'team_def_rating',
        'team_elo',
        'outcome']
    
    cols_to_show = [c for c in cols_to_show if c in final_df.columns]
    
    print("\nMuestra del dataset optimizado con variables EMA, ELO y xG Share:")
    print(final_df[cols_to_show].head(4))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    build_processed_dataset()
