import os
import pandas as pd
import numpy as np
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INTERIM_DATASET_PATH = '../data/interim/intermediate_dataset.parquet'
OUTPUT_PATH = '../data/processed/matches_dataset.parquet'


def add_ema_features(df, spans=[3, 5]):
    logger.info(
        f"Calculando EMA históricos (spans={spans}) para evitar Data Leakage...")
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)

    stats_cols = [
        'xg_created',
        'xg_conceded',
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
        'aerials_won']

    roll_cols = [c for c in stats_cols if c in df.columns]

    for col in roll_cols:
        for span in spans:
            # shift(1) asegura que NO usemos los datos del partido actual (Fix Data Leakage)
            df[f'{col}_ema{span}'] = df.groupby('team')[col].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=2).mean()
            )

    df = df.sort_values('match_date').reset_index(drop=True)
    return df

def calculate_expected_goals(att_rating, def_rating, is_home=True):
    # Asumimos una base de goles ligeramente mayor para el local
    base_goals = 1.45 if is_home else 1.15
    return base_goals * (10 ** ((att_rating - def_rating) / 400))

def update_rating(rating, expected, actual, k_factor=20):
    # Limitamos la sorpresa para que goleadas (ej. 8-0) no rompan el sistema
    diff = np.clip(actual - expected, -3, 3)
    return rating + k_factor * diff

def calculate_expected_score(rating_a, rating_b):
    """Calcula la probabilidad esperada de victoria para el equipo A frente al B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected_score, actual_score, k_factor=30):
    """Actualiza el ELO clásico según el resultado."""
    return rating + k_factor * (actual_score - expected_score)

def add_elo_ratings(df):
    logger.info("Calculando Ratings de Ataque/Defensa y ELO Clásico...")
    df = df.sort_values('match_date').reset_index(drop=True)
    
    att_dict = {}  # {team_name: attack_rating}
    def_dict = {}  # {team_name: defense_rating}
    elo_dict = {}  # {team_name: classic_elo}
    
    match_ids = df['match_id'].unique()
    
    for match_id in match_ids:
        match_rows = df[df['match_id'] == match_id]
        if len(match_rows) != 2:
            continue
            
        # Detectar quién es local y visitante
        if match_rows.iloc[0]['is_home'] == 1:
            row_home, row_away = match_rows.iloc[0], match_rows.iloc[1]
        else:
            row_home, row_away = match_rows.iloc[1], match_rows.iloc[0]
            
        home_team = row_home['team']
        away_team = row_away['team']
        
        # Inicializar en 1000/1500
        for t in [home_team, away_team]:
            if t not in att_dict: att_dict[t] = 1000.0
            if t not in def_dict: def_dict[t] = 1000.0
            if t not in elo_dict: elo_dict[t] = 1500.0
            
        home_att_pre = att_dict[home_team]
        home_def_pre = def_dict[home_team]
        away_att_pre = att_dict[away_team]
        away_def_pre = def_dict[away_team]
        
        home_elo_pre = elo_dict[home_team]
        away_elo_pre = elo_dict[away_team]
        
        goals_home = row_home['goals_scored']
        goals_away = row_away['goals_scored']
        
        # 1. Update Attack/Defense
        exp_goals_home = calculate_expected_goals(home_att_pre, away_def_pre, is_home=True)
        exp_goals_away = calculate_expected_goals(away_att_pre, home_def_pre, is_home=False)
        
        att_dict[home_team] = update_rating(home_att_pre, exp_goals_home, goals_home)
        att_dict[away_team] = update_rating(away_att_pre, exp_goals_away, goals_away)
        def_dict[home_team] = update_rating(home_def_pre, goals_away, exp_goals_away)
        def_dict[away_team] = update_rating(away_def_pre, goals_home, exp_goals_home)
        
        # 2. Update Classic ELO
        outcome_home = row_home['outcome'] # 1, 0, -1
        if outcome_home == 1:
            score_home, score_away = 1.0, 0.0
        elif outcome_home == -1:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5
            
        exp_elo_home = calculate_expected_score(home_elo_pre, away_elo_pre)
        exp_elo_away = calculate_expected_score(away_elo_pre, home_elo_pre)
        
        elo_dict[home_team] = update_elo(home_elo_pre, exp_elo_home, score_home)
        elo_dict[away_team] = update_elo(away_elo_pre, exp_elo_away, score_away)
        
        # Guardar en DF
        # Para row_home
        df.loc[df.index == row_home.name, 'team_att_rating'] = home_att_pre
        df.loc[df.index == row_home.name, 'team_def_rating'] = home_def_pre
        df.loc[df.index == row_home.name, 'opp_att_rating'] = away_att_pre
        df.loc[df.index == row_home.name, 'opp_def_rating'] = away_def_pre
        df.loc[df.index == row_home.name, 'team_elo'] = home_elo_pre
        df.loc[df.index == row_home.name, 'opp_elo'] = away_elo_pre
        df.loc[df.index == row_home.name, 'elo_diff'] = home_elo_pre - away_elo_pre
        
        # Para row_away
        df.loc[df.index == row_away.name, 'team_att_rating'] = away_att_pre
        df.loc[df.index == row_away.name, 'team_def_rating'] = away_def_pre
        df.loc[df.index == row_away.name, 'opp_att_rating'] = home_att_pre
        df.loc[df.index == row_away.name, 'opp_def_rating'] = home_def_pre
        df.loc[df.index == row_away.name, 'team_elo'] = away_elo_pre
        df.loc[df.index == row_away.name, 'opp_elo'] = home_elo_pre
        df.loc[df.index == row_away.name, 'elo_diff'] = away_elo_pre - home_elo_pre

    return df


def add_contextual_features(df):
    logger.info(
        "Calculando variables contextuales (Días de descanso, fuerza del oponente y rachas)...")

    # 1. Días de descanso
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    df['rest_days'] = df.groupby('team')['match_date'].diff().dt.days
    # Promedio semanal si es el primer partido
    df['rest_days'] = df['rest_days'].fillna(7.0)

    # 2. Inercia (Rachas y xG Momentum)
    df['is_win'] = (df['outcome'] == 1).astype(int)
    df['is_loss'] = (df['outcome'] == -1).astype(int)
    
    df['win_streak_3'] = df.groupby('team')['is_win'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['loss_streak_3'] = df.groupby('team')['is_loss'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    
    df['xg_diff_raw'] = df['xg_created'] - df['xg_conceded']
    
    def calc_slope(y):
        if len(y) < 2: return 0.0
        x = np.arange(len(y))
        # cov(x,y)/var(x)
        return np.polyfit(x, y, 1)[0]
        
    df['xg_momentum_5'] = df.groupby('team')['xg_diff_raw'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(calc_slope, raw=True)
    ).fillna(0)
    
    df = df.drop(columns=['is_win', 'is_loss', 'xg_diff_raw'])

    # 3. Fuerza del Oponente (Traer el historial 'ema' del rival y sus rachas)
    ema_cols = [c for c in df.columns if '_ema' in c]
    opp_cols = ema_cols + ['rest_days', 'win_streak_3', 'loss_streak_3', 'xg_momentum_5']
    opp_df = df[['team', 'match_date'] + opp_cols].copy()

    # Renombrar columnas para el oponente
    opp_rename = {c: f"opp_{c}" for c in opp_cols}
    opp_rename['team'] = 'opponent'
    opp_df = opp_df.rename(columns=opp_rename)

    # Merge con el dataset principal
    df = pd.merge(df, opp_df, on=['opponent', 'match_date'], how='left')

    # Rellenar nulos de oponentes nuevos
    for c in opp_rename.values():
        if c != 'opponent':
            df[c] = df[c].fillna(0)

    # 4. Métricas Relativas
    df['rest_diff'] = df['rest_days'] - df['opp_rest_days']
    
    # Opcional: Crear métricas de fuerza relativa (Ej: Mi ataque vs Su defensa usando EMA3)
    if 'xg_created_ema3' in df.columns and 'opp_xg_conceded_ema3' in df.columns:
        df['relative_attack_strength'] = df['xg_created_ema3'] - \
            df['opp_xg_conceded_ema3']

    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def add_h2h_features(df):
    logger.info("Calculando variables H2H (enfrentamientos directos)...")
    df = df.sort_values('match_date').reset_index(drop=True)
    
    df['h2h_points'] = df['outcome'].map({1: 3, 0: 1, -1: 0})
    
    df['h2h_games_played'] = df.groupby(['team', 'opponent']).cumcount()
    
    df['h2h_points_last_5'] = df.groupby(['team', 'opponent'])['h2h_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    
    df['h2h_win_rate_hist'] = df.groupby(['team', 'opponent'])['outcome'].transform(
        lambda x: (x.shift(1) == 1).expanding().mean()
    ).fillna(0)
    
    df['h2h_draw_rate_hist'] = df.groupby(['team', 'opponent'])['outcome'].transform(
        lambda x: (x.shift(1) == 0).expanding().mean()
    ).fillna(0)
    
    df = df.drop(columns=['h2h_points'])
    return df


def add_advanced_fatigue(df):
    logger.info("Calculando Fatiga Avanzada (cambio de competiciones)...")
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    
    df['prev_competition'] = df.groupby('team')['competition'].shift(1)
    
    df['is_european_hangover'] = (
        (df['competition'] != df['prev_competition']) & 
        (df['prev_competition'].notna()) & 
        (df['rest_days'] <= 4)
    ).astype(int)
    
    df = df.drop(columns=['prev_competition'])
    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def add_squad_value_features(df):
    logger.info("Integrando Valores de Mercado de Transfermarkt...")
    
    tm_path = '../data/raw/transfermarkt_squad_values.parquet'
    if not os.path.exists(tm_path):
        logger.warning(f"No se encontró el archivo de Transfermarkt en {tm_path}. Saltando variable.")
        return df
        
    tm_df = pd.read_parquet(tm_path, engine='fastparquet')
    
    # Derivar la temporada en el df principal
    # Si el mes es >= 7 (Julio), la temporada es el año actual, si no, el año anterior.
    df['season_year'] = df['match_date'].dt.year
    df.loc[df['match_date'].dt.month < 7, 'season_year'] -= 1
    
    # Merge local
    df = pd.merge(
        df, 
        tm_df[['season_year', 'team', 'squad_value_millions']], 
        on=['season_year', 'team'], 
        how='left'
    )
    df = df.rename(columns={'squad_value_millions': 'team_squad_value'})
    
    # Merge visitante (oponente)
    tm_df_opp = tm_df[['season_year', 'team', 'squad_value_millions']].rename(
        columns={'team': 'opponent', 'squad_value_millions': 'opp_squad_value'}
    )
    df = pd.merge(
        df, 
        tm_df_opp, 
        on=['season_year', 'opponent'], 
        how='left'
    )
    
    # Fill NaN para equipos sin datos de Transfermarkt (ej. recién ascendidos) con el mínimo de la liga
    df['team_squad_value'] = df.groupby(['season_year'])['team_squad_value'].transform(lambda x: x.fillna(x.min() if not pd.isna(x.min()) else 10.0))
    df['opp_squad_value'] = df.groupby(['season_year'])['opp_squad_value'].transform(lambda x: x.fillna(x.min() if not pd.isna(x.min()) else 10.0))
    
    # Calcular diferencia de valor
    df['squad_value_diff'] = df['team_squad_value'] - df['opp_squad_value']
    
    df = df.drop(columns=['season_year'])
    df = df.sort_values('match_date').reset_index(drop=True)
    return df



def build_processed_dataset():
    if not os.path.exists(INTERIM_DATASET_PATH):
        logger.error(
            f"No se encontró el dataset intermedio en: {INTERIM_DATASET_PATH}. "
            "Por favor, ejecuta primero core/data_adapter.py para generarlo.")
        return

    logger.info(f"Cargando dataset intermedio desde {INTERIM_DATASET_PATH}...")
    final_df = pd.read_parquet(INTERIM_DATASET_PATH, engine='fastparquet')
    
    logger.info(
        f"Procesando {len(final_df)} filas para extracción de características avanzadas...")

    # Ordenar por fecha
    final_df['match_date'] = pd.to_datetime(final_df['match_date'])
    final_df = final_df.sort_values('match_date').reset_index(drop=True)

    # Aplicar promedios exponenciales (EMA)
    final_df = add_ema_features(final_df, spans=[3, 5])
    
    # Calcular ELO ratings
    final_df = add_elo_ratings(final_df)

    # Añadir contexto competitivo (Descanso y fuerza del rival)
    final_df = add_contextual_features(final_df)

    # Añadir H2H y Fatiga Avanzada
    final_df = add_h2h_features(final_df)
    final_df = add_advanced_fatigue(final_df)
    
    # Añadir Valor de Plantilla (Transfermarkt)
    final_df = add_squad_value_features(final_df)

    # Crear directorio si no existe
    processed_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(processed_dir, exist_ok=True)

    # Guardar dataset procesado
    final_df.to_parquet(OUTPUT_PATH, engine='fastparquet', index=False)
    logger.info(
        f"Dataset de entrenamiento avanzado guardado exitosamente en: {OUTPUT_PATH}")
    logger.info(
        f"Estructura del dataset final: {final_df.shape}")

    # Mostrar muestra de las nuevas features rolling
    cols_to_show = [
        'team',
        'is_home',
        'xg_created',
        'xg_created_ema3',
        'team_att_rating',
        'team_def_rating',
        'outcome']
    
    # Check if cols exist before showing
    cols_to_show = [c for c in cols_to_show if c in final_df.columns]
    
    print("\nMuestra del dataset con variables EMA y ELO:")
    print(final_df[cols_to_show].head(4))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    build_processed_dataset()
