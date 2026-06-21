import os
import pandas as pd
import logging
from team_mapping import normalize_team_name

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MATCHES_METADATA_PATH = '../data/raw/statsbomb/matches.parquet'
EVENTS_DIR = '../data/raw/statsbomb/events/'
FOOTBALL_DATA_DIR = '../data/raw/football_data/'
INTERMEDIATE_OUTPUT_PATH = '../data/interim/intermediate_dataset.parquet'


def count_events(df, condition):
    """Función auxiliar para contar eventos de forma segura."""
    try:
        return int(condition.sum())
    except BaseException:
        return 0


def extract_team_stats(events_df, team_name, opponent_name):
    """
    Extrae todo el arsenal de métricas tácticas avanzadas para un equipo específico en un partido.
    """
    team_events = events_df[events_df['team'] == team_name]
    opp_events = events_df[events_df['team'] == opponent_name]

    # Manejo de columnas opcionales que pueden no existir en partidos muy
    # antiguos
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
            on_target = team_shots['shot_outcome'].isin(
                ['Goal', 'Saved', 'Saved to Post', 'Saved Off Target'])
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
            # Tratamiento especial si hay nulos reales en pandas
            completed_mask = team_passes['pass_outcome'].isna() | (
                team_passes['pass_outcome'].astype(str).str.lower() == 'nan') | (
                team_passes['pass_outcome'] == 'None')

            stats['passes_completed'] = count_events(
                team_passes, completed_mask)
        else:
            stats['passes_completed'] = 0

        stats['pass_accuracy'] = stats['passes_completed'] / \
            stats['passes_total'] if stats['passes_total'] > 0 else 0.0

    # Posesión (%)
    if col_exists('possession_team'):
        total_poss = len(
            events_df[events_df['possession_team'].isin([team_name, opponent_name])])
        team_poss = len(
            team_events[team_events['possession_team'] == team_name])
        stats['possession_pct'] = float(
            team_poss / total_poss) if total_poss > 0 else 0.5
    else:
        stats['possession_pct'] = 0.5

    # --- 3. Creación Ofensiva Avanzada ---
    stats['crosses'] = 0
    stats['corners'] = 0
    stats['through_balls'] = 0
    stats['key_passes'] = 0
    stats['dribbles_completed'] = 0

    if col_exists('pass_type'):
        stats['corners'] = count_events(
            team_events, team_events['pass_type'] == 'Corner')

    if col_exists('pass_cross'):
        stats['crosses'] = count_events(
            team_events, team_events['pass_cross'] == 'True')

    if col_exists('pass_through_ball'):
        stats['through_balls'] = count_events(
            team_events, team_events['pass_through_ball'] == 'True')

    if col_exists('pass_shot_assist') and col_exists('pass_goal_assist'):
        key_passes_mask = (
            team_events['pass_shot_assist'] == 'True') | (
            team_events['pass_goal_assist'] == 'True')
        stats['key_passes'] = count_events(team_events, key_passes_mask)

    if col_exists('dribble_outcome') and col_exists('type'):
        dribbles_mask = (
            team_events['type'] == 'Dribble') & (
            team_events['dribble_outcome'] == 'Complete')
        stats['dribbles_completed'] = count_events(team_events, dribbles_mask)

    # --- 4. Presión y Defensa ---
    stats['pressures'] = count_events(
        team_events, team_events['type'] == 'Pressure') if col_exists('type') else 0
    stats['interceptions'] = count_events(
        team_events,
        team_events['type'] == 'Interception') if col_exists('type') else 0
    stats['clearances'] = count_events(
        team_events, team_events['type'] == 'Clearance') if col_exists('type') else 0
    stats['blocks'] = count_events(
        team_events,
        team_events['type'] == 'Block') if col_exists('type') else 0
    stats['ball_recoveries'] = count_events(
        team_events,
        team_events['type'] == 'Ball Recovery') if col_exists('type') else 0

    if col_exists('under_pressure'):
        stats['actions_under_pressure'] = count_events(
            team_events, team_events['under_pressure'] == 'True')
    else:
        stats['actions_under_pressure'] = 0

    # --- 5. Físico y Faltas ---
    stats['fouls_committed'] = count_events(
        team_events,
        team_events['type'] == 'Foul Committed') if col_exists('type') else 0
    stats['fouls_won'] = count_events(
        team_events, team_events['type'] == 'Foul Won') if col_exists('type') else 0

    stats['yellow_cards'] = 0
    stats['red_cards'] = 0
    if col_exists('foul_committed_card'):
        stats['yellow_cards'] += count_events(team_events,
                                              team_events['foul_committed_card'].isin(['Yellow Card',
                                                                                       'Second Yellow']))
        stats['red_cards'] += count_events(team_events,
                                           team_events['foul_committed_card'] == 'Red Card')
    if col_exists('bad_behaviour_card'):
        stats['yellow_cards'] += count_events(team_events,
                                              team_events['bad_behaviour_card'].isin(['Yellow Card',
                                                                                      'Second Yellow']))
        stats['red_cards'] += count_events(team_events,
                                           team_events['bad_behaviour_card'] == 'Red Card')

    aerial_won = 0
    if col_exists('pass_aerial_won'):
        aerial_won += count_events(team_events,
                                   team_events['pass_aerial_won'] == 'True')
    if col_exists('clearance_aerial_won'):
        aerial_won += count_events(team_events,
                                   team_events['clearance_aerial_won'] == 'True')
    if col_exists('shot_aerial_won'):
        aerial_won += count_events(team_events,
                                   team_events['shot_aerial_won'] == 'True')
    stats['aerials_won'] = aerial_won

    return stats


def process_statsbomb_match(row):
    """Procesa un partido de StatsBomb y devuelve dos filas (Local y Visitante) con sus métricas."""
    match_id = row['match_id']
    match_date = pd.to_datetime(row['match_date']).strftime('%Y%m%d')
    home_team_raw = row['home_team']
    away_team_raw = row['away_team']
    
    # Normalizar nombres
    home_team = normalize_team_name(home_team_raw)
    away_team = normalize_team_name(away_team_raw)
    
    # Universal Match ID
    universal_id = f"statsbomb_{match_date}_{home_team}_{away_team}".replace(" ", "")

    file_path = os.path.join(EVENTS_DIR, f"{match_id}_events.parquet")
    if not os.path.exists(file_path):
        return []

    try:
        events_df = pd.read_parquet(file_path, engine='fastparquet')
    except Exception:
        return []

    # Extraer estadísticas (pasando los nombres crudos porque así vienen en el parquet de eventos)
    home_stats = extract_team_stats(events_df, home_team_raw, away_team_raw)
    away_stats = extract_team_stats(events_df, away_team_raw, home_team_raw)

    rows = []

    # Generar fila del equipo Local
    home_row = {
        'match_id': universal_id,
        'source': 'statsbomb',
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

    home_row['outcome'] = 1 if home_row['goals_scored'] > home_row['goals_conceded'] else (-1 if home_row['goals_scored'] < home_row['goals_conceded'] else 0)
    home_row.update(home_stats)
    rows.append(home_row)

    # Generar fila del equipo Visitante
    away_row = {
        'match_id': universal_id,
        'source': 'statsbomb',
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

    away_row['outcome'] = 1 if away_row['goals_scored'] > away_row['goals_conceded'] else (-1 if away_row['goals_scored'] < away_row['goals_conceded'] else 0)
    away_row.update(away_stats)
    rows.append(away_row)

    return rows


def build_intermediate_from_statsbomb():
    if not os.path.exists(MATCHES_METADATA_PATH):
        logger.error(f"No se encontró el archivo de StatsBomb en: {MATCHES_METADATA_PATH}")
        return pd.DataFrame()

    matches_df = pd.read_parquet(MATCHES_METADATA_PATH, engine='fastparquet')
    logger.info(f"Generando dataset intermedio a partir de {len(matches_df)} partidos de StatsBomb...")

    all_rows = []
    for idx, row in matches_df.iterrows():
        match_rows = process_statsbomb_match(row)
        all_rows.extend(match_rows)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def build_intermediate_from_footballdata():
    if not os.path.exists(FOOTBALL_DATA_DIR):
        logger.error(f"No se encontró el directorio de Football-Data en: {FOOTBALL_DATA_DIR}")
        return pd.DataFrame()
        
    csv_files = [f for f in os.listdir(FOOTBALL_DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        logger.warning(f"No hay archivos CSV en {FOOTBALL_DATA_DIR}")
        return pd.DataFrame()
        
    all_rows = []
    logger.info(f"Generando dataset intermedio a partir de {len(csv_files)} archivos de Football-Data...")
    
    for f in csv_files:
        file_path = os.path.join(FOOTBALL_DATA_DIR, f)
        try:
            # Algunas columnas pueden tener tipos mixtos, forzamos low_memory=False
            df = pd.read_csv(file_path, low_memory=False)
            
            # Limpiamos filas vacías que a veces vienen en los CSV
            df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'Date'])
            
            for idx, row in df.iterrows():
                # Fechas vienen como DD/MM/YYYY o DD/MM/YY
                match_date_str = str(row['Date'])
                try:
                    match_date = pd.to_datetime(match_date_str, format='mixed', dayfirst=True)
                except:
                    continue # Ignorar si no se puede parsear
                    
                match_date_formatted = match_date.strftime('%Y-%m-%d')
                match_date_id = match_date.strftime('%Y%m%d')
                
                home_team = normalize_team_name(row['HomeTeam'])
                away_team = normalize_team_name(row['AwayTeam'])
                
                universal_id = f"footballdata_{match_date_id}_{home_team}_{away_team}".replace(" ", "")
                competition = row.get('Div', '')
                
                # Extracción segura de datos
                def get_stat(col):
                    try:
                        val = row[col]
                        return float(val) if pd.notna(val) else 0.0
                    except:
                        return 0.0

                # LOCAL
                home_row = {
                    'match_id': universal_id,
                    'source': 'footballdata',
                    'match_date': match_date_formatted,
                    'competition': competition,
                    'season': '', # Podría inferirse del nombre del archivo, se deja en blanco por simplicidad
                    'competition_stage': '',
                    'team': home_team,
                    'opponent': away_team,
                    'is_home': 1,
                    'goals_scored': get_stat('FTHG'),
                    'goals_conceded': get_stat('FTAG'),
                    
                    # Mapeo de estadísticas
                    'xg_created': 0.0,
                    'xg_conceded': 0.0,
                    'shots_total': get_stat('HS'),
                    'shots_on_target': get_stat('HST'),
                    'passes_total': 0.0,
                    'passes_completed': 0.0,
                    'pass_accuracy': 0.0,
                    'possession_pct': 0.5,
                    'crosses': 0.0,
                    'corners': get_stat('HC'),
                    'through_balls': 0.0,
                    'key_passes': 0.0,
                    'dribbles_completed': 0.0,
                    'pressures': 0.0,
                    'interceptions': 0.0,
                    'clearances': 0.0,
                    'blocks': 0.0,
                    'ball_recoveries': 0.0,
                    'actions_under_pressure': 0.0,
                    'fouls_committed': get_stat('HF'),
                    'fouls_won': get_stat('AF'), # Faltas sufridas (cometidas por visitante)
                    'yellow_cards': get_stat('HY'),
                    'red_cards': get_stat('HR'),
                    'aerials_won': 0.0
                }
                
                home_row['outcome'] = 1 if home_row['goals_scored'] > home_row['goals_conceded'] else (-1 if home_row['goals_scored'] < home_row['goals_conceded'] else 0)
                all_rows.append(home_row)
                
                # VISITANTE
                away_row = {
                    'match_id': universal_id,
                    'source': 'footballdata',
                    'match_date': match_date_formatted,
                    'competition': competition,
                    'season': '',
                    'competition_stage': '',
                    'team': away_team,
                    'opponent': home_team,
                    'is_home': 0,
                    'goals_scored': get_stat('FTAG'),
                    'goals_conceded': get_stat('FTHG'),
                    
                    # Mapeo de estadísticas
                    'xg_created': 0.0,
                    'xg_conceded': 0.0,
                    'shots_total': get_stat('AS'),
                    'shots_on_target': get_stat('AST'),
                    'passes_total': 0.0,
                    'passes_completed': 0.0,
                    'pass_accuracy': 0.0,
                    'possession_pct': 0.5,
                    'crosses': 0.0,
                    'corners': get_stat('AC'),
                    'through_balls': 0.0,
                    'key_passes': 0.0,
                    'dribbles_completed': 0.0,
                    'pressures': 0.0,
                    'interceptions': 0.0,
                    'clearances': 0.0,
                    'blocks': 0.0,
                    'ball_recoveries': 0.0,
                    'actions_under_pressure': 0.0,
                    'fouls_committed': get_stat('AF'),
                    'fouls_won': get_stat('HF'), # Faltas sufridas (cometidas por local)
                    'yellow_cards': get_stat('AY'),
                    'red_cards': get_stat('AR'),
                    'aerials_won': 0.0
                }
                
                away_row['outcome'] = 1 if away_row['goals_scored'] > away_row['goals_conceded'] else (-1 if away_row['goals_scored'] < away_row['goals_conceded'] else 0)
                all_rows.append(away_row)
                
        except Exception as e:
            logger.error(f"Error procesando {f}: {e}")
            
    if not all_rows:
        return pd.DataFrame()
        
    return pd.DataFrame(all_rows)

def build_unified_intermediate_dataset():
    logger.info("Iniciando construcción de dataset unificado...")
    
    df_statsbomb = build_intermediate_from_statsbomb()
    logger.info(f"Filas de StatsBomb obtenidas: {len(df_statsbomb)}")
    
    df_footballdata = build_intermediate_from_footballdata()
    logger.info(f"Filas de Football-Data obtenidas: {len(df_footballdata)}")
    
    dfs = []
    if not df_statsbomb.empty:
        dfs.append(df_statsbomb)
    if not df_footballdata.empty:
        dfs.append(df_footballdata)
        
    if not dfs:
        logger.error("No se pudo generar ningún dato. Dataset intermedio no guardado.")
        return
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Ordenar por fecha
    final_df['match_date'] = pd.to_datetime(final_df['match_date'])
    final_df = final_df.sort_values('match_date').reset_index(drop=True)

    # Eliminar posibles duplicados
    # Algunos partidos podrían existir en StatsBomb y en FootballData
    # Vamos a eliminar duplicados basados en 'match_date', 'team', 'opponent' dando prioridad a StatsBomb (que tiene más columnas ricas)
    final_df = final_df.sort_values(by=['match_date', 'team', 'opponent', 'source'], ascending=[True, True, True, False])
    before_drop = len(final_df)
    final_df = final_df.drop_duplicates(subset=['match_date', 'team', 'opponent'], keep='first')
    after_drop = len(final_df)
    logger.info(f"Duplicados eliminados: {before_drop - after_drop}")

    # Crear directorio interim si no existe
    interim_dir = os.path.dirname(INTERMEDIATE_OUTPUT_PATH)
    os.makedirs(interim_dir, exist_ok=True)

    # Guardar dataset procesado
    final_df.to_parquet(INTERMEDIATE_OUTPUT_PATH, engine='fastparquet', index=False)
    logger.info(f"Dataset intermedio universal guardado exitosamente en: {INTERMEDIATE_OUTPUT_PATH}")
    logger.info(f"Estructura del dataset intermedio final: {final_df.shape}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    build_unified_intermediate_dataset()
