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
                lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
            )

    df = df.sort_values('match_date').reset_index(drop=True)
    return df


def calculate_expected_score(rating_a, rating_b):
    """Calcula la probabilidad esperada de victoria para el equipo A frente al B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, expected_score, actual_score, k_factor=30):
    """Actualiza el ELO según el resultado."""
    return rating + k_factor * (actual_score - expected_score)


def add_elo_ratings(df):
    logger.info("Calculando ELO ratings secuenciales...")
    # Asegurar que esté ordenado cronológicamente
    df = df.sort_values('match_date').reset_index(drop=True)
    
    elo_dict = {}  # {team_name: elo}
    
    # Necesitamos iterar por partido (2 filas por partido en el df)
    match_ids = df['match_id'].unique()
    
    for match_id in match_ids:
        match_rows = df[df['match_id'] == match_id]
        if len(match_rows) != 2:
            continue # Salto si hay un error en los datos
            
        row1 = match_rows.iloc[0]
        row2 = match_rows.iloc[1]
        
        team1 = row1['team']
        team2 = row2['team']
        
        # Inicializar en 1500 si no existe
        if team1 not in elo_dict:
            elo_dict[team1] = 1500.0
        if team2 not in elo_dict:
            elo_dict[team2] = 1500.0
            
        elo1_pre = elo_dict[team1]
        elo2_pre = elo_dict[team2]
        
        # Determinar puntuación real (1 victoria, 0.5 empate, 0 derrota)
        outcome1 = row1['outcome'] # 1, 0, -1
        if outcome1 == 1:
            score1, score2 = 1.0, 0.0
        elif outcome1 == -1:
            score1, score2 = 0.0, 1.0
        else:
            score1, score2 = 0.5, 0.5
            
        exp1 = calculate_expected_score(elo1_pre, elo2_pre)
        exp2 = calculate_expected_score(elo2_pre, elo1_pre)
        
        elo_dict[team1] = update_elo(elo1_pre, exp1, score1)
        elo_dict[team2] = update_elo(elo2_pre, exp2, score2)
        
        # Guardamos para asignar al df original
        df.loc[df['match_id'] == match_id, 'team_elo'] = [elo1_pre, elo2_pre]
        df.loc[df['match_id'] == match_id, 'opp_elo'] = [elo2_pre, elo1_pre]
        df.loc[df['match_id'] == match_id, 'elo_diff'] = [elo1_pre - elo2_pre, elo2_pre - elo1_pre]

    return df


def add_contextual_features(df):
    logger.info(
        "Calculando variables contextuales (Días de descanso y Fuerza del oponente)...")

    # 1. Días de descanso
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    df['rest_days'] = df.groupby('team')['match_date'].diff().dt.days
    # Promedio semanal si es el primer partido
    df['rest_days'] = df['rest_days'].fillna(7.0)

    # 2. Fuerza del Oponente (Traer el historial 'ema' del rival)
    ema_cols = [c for c in df.columns if '_ema' in c]
    opp_df = df[['team', 'match_date'] + ema_cols].copy()

    # Renombrar columnas para el oponente
    opp_rename = {c: f"opp_{c}" for c in ema_cols}
    opp_rename['team'] = 'opponent'
    opp_df = opp_df.rename(columns=opp_rename)

    # Merge con el dataset principal
    df = pd.merge(df, opp_df, on=['opponent', 'match_date'], how='left')

    # Rellenar nulos de oponentes nuevos
    for c in opp_rename.values():
        if c != 'opponent':
            df[c] = df[c].fillna(0)

    # Opcional: Crear métricas de fuerza relativa (Ej: Mi ataque vs Su defensa usando EMA3)
    if 'xg_created_ema3' in df.columns and 'opp_xg_conceded_ema3' in df.columns:
        df['relative_attack_strength'] = df['xg_created_ema3'] - \
            df['opp_xg_conceded_ema3']

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
        'team_elo',
        'outcome']
    
    # Check if cols exist before showing
    cols_to_show = [c for c in cols_to_show if c in final_df.columns]
    
    print("\nMuestra del dataset con variables EMA y ELO:")
    print(final_df[cols_to_show].head(4))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    build_processed_dataset()
