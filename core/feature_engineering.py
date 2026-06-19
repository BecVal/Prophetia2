import os
import pandas as pd
import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MATCHES_METADATA_PATH = '../data/raw/statsbomb/matches.parquet'
EVENTS_DIR = '../data/raw/statsbomb/events/'
OUTPUT_PATH = '../data/processed/matches_dataset.parquet'

def count_events(df, condition):
    """Función auxiliar para contar eventos de forma segura."""
    try:
        return int(condition.sum())
    except:
        return 0

def extract_team_stats(events_df, team_name, opponent_name):
    """
    Extrae todo el arsenal de métricas tácticas avanzadas para un equipo específico en un partido.
    """
    team_events = events_df[events_df['team'] == team_name]
    opp_events = events_df[events_df['team'] == opponent_name]
    
    # Manejo de columnas opcionales que pueden no existir en partidos muy antiguos
    def col_exists(col):
        return col in events_df.columns
        
    stats = {}
    
    # --- 1. Goles Esperados (xG) y Tiros ---
    if col_exists('type') and col_exists('shot_statsbomb_xg'):
        team_shots = team_events[team_events['type'] == 'Shot']
        opp_shots = opp_events[opp_events['type'] == 'Shot']
        
        stats['xg_created'] = float(team_shots['shot_statsbomb_xg'].sum())
        stats['xg_conceded'] = float(opp_shots['shot_statsbomb_xg'].sum())
        stats['shots_total'] = len(team_shots)
        
        # Tiros a puerta (Goal, Saved, Saved to Post)
        if col_exists('shot_outcome'):
            on_target = team_shots['shot_outcome'].isin(['Goal', 'Saved', 'Saved to Post', 'Saved Off Target'])
            stats['shots_on_target'] = count_events(team_shots, on_target)
        else:
            stats['shots_on_target'] = 0
    else:
        stats['xg_created'] = 0.0
        stats['xg_conceded'] = 0.0
        stats['shots_total'] = 0
        stats['shots_on_target'] = 0

    # --- 2. Posesión y Pases ---
    if col_exists('type'):
        team_passes = team_events[team_events['type'] == 'Pass']
        stats['passes_total'] = len(team_passes)
        
        if col_exists('pass_outcome'):
            # Outcome Nulo = Pase Completo
            completed_mask = team_passes['pass_outcome'].isna() | (team_passes['pass_outcome'] == 'None') | (team_passes['pass_outcome'] == 'nan')
            # Tratamiento especial si hay nulos reales en pandas
            completed_mask = team_passes['pass_outcome'].isna() | (team_passes['pass_outcome'].astype(str).str.lower() == 'nan')
            
            stats['passes_completed'] = count_events(team_passes, completed_mask)
        else:
            stats['passes_completed'] = 0
            
        stats['pass_accuracy'] = stats['passes_completed'] / stats['passes_total'] if stats['passes_total'] > 0 else 0.0
        
    # Posesión (%)
    if col_exists('possession_team'):
        total_poss = len(events_df[events_df['possession_team'].isin([team_name, opponent_name])])
        team_poss = len(team_events[team_events['possession_team'] == team_name])
        stats['possession_pct'] = float(team_poss / total_poss) if total_poss > 0 else 0.5
    else:
        stats['possession_pct'] = 0.5

    # --- 3. Creación Ofensiva Avanzada ---
    stats['crosses'] = 0
    stats['corners'] = 0
    stats['through_balls'] = 0
    stats['key_passes'] = 0
    stats['dribbles_completed'] = 0
    
    if col_exists('pass_type'):
        stats['corners'] = count_events(team_events, team_events['pass_type'] == 'Corner')
        
    if col_exists('pass_cross'):
        stats['crosses'] = count_events(team_events, team_events['pass_cross'] == 'True')
        
    if col_exists('pass_through_ball'):
        stats['through_balls'] = count_events(team_events, team_events['pass_through_ball'] == 'True')
        
    if col_exists('pass_shot_assist') and col_exists('pass_goal_assist'):
        key_passes_mask = (team_events['pass_shot_assist'] == 'True') | (team_events['pass_goal_assist'] == 'True')
        stats['key_passes'] = count_events(team_events, key_passes_mask)
        
    if col_exists('dribble_outcome') and col_exists('type'):
        dribbles_mask = (team_events['type'] == 'Dribble') & (team_events['dribble_outcome'] == 'Complete')
        stats['dribbles_completed'] = count_events(team_events, dribbles_mask)

    # --- 4. Presión y Defensa ---
    stats['pressures'] = count_events(team_events, team_events['type'] == 'Pressure') if col_exists('type') else 0
    stats['interceptions'] = count_events(team_events, team_events['type'] == 'Interception') if col_exists('type') else 0
    stats['clearances'] = count_events(team_events, team_events['type'] == 'Clearance') if col_exists('type') else 0
    stats['blocks'] = count_events(team_events, team_events['type'] == 'Block') if col_exists('type') else 0
    stats['ball_recoveries'] = count_events(team_events, team_events['type'] == 'Ball Recovery') if col_exists('type') else 0
    
    if col_exists('under_pressure'):
        stats['actions_under_pressure'] = count_events(team_events, team_events['under_pressure'] == 'True')
    else:
        stats['actions_under_pressure'] = 0

    # --- 5. Físico y Faltas ---
    stats['fouls_committed'] = count_events(team_events, team_events['type'] == 'Foul Committed') if col_exists('type') else 0
    stats['fouls_won'] = count_events(team_events, team_events['type'] == 'Foul Won') if col_exists('type') else 0
    
    stats['yellow_cards'] = 0
    stats['red_cards'] = 0
    if col_exists('foul_committed_card'):
        stats['yellow_cards'] += count_events(team_events, team_events['foul_committed_card'].isin(['Yellow Card', 'Second Yellow']))
        stats['red_cards'] += count_events(team_events, team_events['foul_committed_card'] == 'Red Card')
    if col_exists('bad_behaviour_card'):
        stats['yellow_cards'] += count_events(team_events, team_events['bad_behaviour_card'].isin(['Yellow Card', 'Second Yellow']))
        stats['red_cards'] += count_events(team_events, team_events['bad_behaviour_card'] == 'Red Card')
    
    aerial_won = 0
    if col_exists('pass_aerial_won'): aerial_won += count_events(team_events, team_events['pass_aerial_won'] == 'True')
    if col_exists('clearance_aerial_won'): aerial_won += count_events(team_events, team_events['clearance_aerial_won'] == 'True')
    if col_exists('shot_aerial_won'): aerial_won += count_events(team_events, team_events['shot_aerial_won'] == 'True')
    stats['aerials_won'] = aerial_won
    
    return stats

def process_match(row):
    """Procesa un partido y devuelve dos filas (Local y Visitante) con sus métricas."""
    match_id = row['match_id']
    home_team = row['home_team']
    away_team = row['away_team']
    
    file_path = os.path.join(EVENTS_DIR, f"{match_id}_events.parquet")
    if not os.path.exists(file_path):
        logger.warning(f"Archivo de eventos no encontrado para el partido {match_id}")
        return []
        
    try:
        events_df = pd.read_parquet(file_path, engine='fastparquet')
    except Exception as e:
        logger.error(f"Error al leer parquet del partido {match_id}: {e}")
        return []
        
    # Extraer estadísticas para ambos equipos
    home_stats = extract_team_stats(events_df, home_team, away_team)
    away_stats = extract_team_stats(events_df, away_team, home_team)
    
    rows = []
    
    # Generar fila del equipo Local
    home_row = {
        'match_id': match_id,
        'match_date': row['match_date'],
        'competition': row.get('competition_name', ''),
        'season': row.get('season_name', ''),
        'competition_stage': row.get('competition_stage', ''),
        'team': home_team,
        'opponent': away_team,
        'is_home': 1,
        'goals_scored': row['home_score'],
        'goals_conceded': row['away_score']
    }
    
    if home_row['goals_scored'] > home_row['goals_conceded']:
        home_row['outcome'] = 1  # Win
    elif home_row['goals_scored'] < home_row['goals_conceded']:
        home_row['outcome'] = -1 # Loss
    else:
        home_row['outcome'] = 0  # Draw
        
    home_row.update(home_stats)
    rows.append(home_row)
    
    # Generar fila del equipo Visitante
    away_row = {
        'match_id': match_id,
        'match_date': row['match_date'],
        'competition': row.get('competition_name', ''),
        'season': row.get('season_name', ''),
        'competition_stage': row.get('competition_stage', ''),
        'team': away_team,
        'opponent': home_team,
        'is_home': 0,
        'goals_scored': row['away_score'],
        'goals_conceded': row['home_score']
    }
    
    if away_row['goals_scored'] > away_row['goals_conceded']:
        away_row['outcome'] = 1  # Win
    elif away_row['goals_scored'] < away_row['goals_conceded']:
        away_row['outcome'] = -1 # Loss
    else:
        away_row['outcome'] = 0  # Draw
        
    away_row.update(away_stats)
    rows.append(away_row)
    
    return rows

def add_rolling_features(df, window_size=3):
    logger.info(f"Calculando promedios móviles históricos (ventana={window_size}) para evitar Data Leakage...")
    df = df.sort_values(['team', 'match_date']).reset_index(drop=True)
    
    stats_cols = [
        'xg_created', 'xg_conceded', 'shots_total', 'shots_on_target',
        'passes_total', 'passes_completed', 'pass_accuracy', 'possession_pct',
        'crosses', 'corners', 'through_balls', 'key_passes', 'dribbles_completed',
        'pressures', 'interceptions', 'clearances', 'blocks', 'ball_recoveries',
        'actions_under_pressure', 'fouls_committed', 'fouls_won', 
        'yellow_cards', 'red_cards', 'aerials_won'
    ]
    
    roll_cols = [c for c in stats_cols if c in df.columns]
    
    for col in roll_cols:
        # shift(1) asegura que NO usemos los datos del partido que estamos prediciendo (Fix Data Leakage)
        df[f'{col}_rolling'] = df.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )
    
    df = df.sort_values('match_date').reset_index(drop=True)
    return df

def build_processed_dataset():
    if not os.path.exists(MATCHES_METADATA_PATH):
        logger.error(f"No se encontró el archivo de partidos en: {MATCHES_METADATA_PATH}")
        return
        
    matches_df = pd.read_parquet(MATCHES_METADATA_PATH, engine='fastparquet')
    logger.info(f"Procesando {len(matches_df)} partidos para extracción de características avanzadas...")
    
    all_rows = []
    
    for idx, row in matches_df.iterrows():
        match_id = row['match_id']
        logger.info(f"Procesando partido {match_id}: {row['home_team']} vs {row['away_team']}")
        
        match_rows = process_match(row)
        all_rows.extend(match_rows)
            
    if not all_rows:
        logger.warning("No se pudieron extraer métricas de ningún partido.")
        return
        
    final_df = pd.DataFrame(all_rows)
    
    # Ordenar por fecha
    final_df['match_date'] = pd.to_datetime(final_df['match_date'])
    final_df = final_df.sort_values('match_date').reset_index(drop=True)
    
    # Aplicar promedios móviles históricos (Fase 1)
    final_df = add_rolling_features(final_df, window_size=3)
    
    # Crear directorio si no existe
    processed_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(processed_dir, exist_ok=True)
        
    # Guardar dataset procesado
    final_df.to_parquet(OUTPUT_PATH, engine='fastparquet', index=False)
    logger.info(f"Dataset de entrenamiento avanzado guardado exitosamente en: {OUTPUT_PATH}")
    logger.info(f"Estructura del dataset final (debe ser 2x los partidos originales): {final_df.shape}")
    
    # Mostrar muestra de las nuevas features rolling
    cols_to_show = ['team', 'is_home', 'xg_created', 'xg_created_rolling', 'corners_rolling', 'outcome']
    print("\nMuestra del dataset con variables históricas (rolling):")
    print(final_df[cols_to_show].head(4))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    build_processed_dataset()
