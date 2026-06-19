import os
import time
import pandas as pd
from statsbombpy import sb
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = '../data/raw/statsbomb/events/'
COMPETITIONS = ['La Liga', 'Premier League', 'Ligue 1', '1. Bundesliga', 'Champions League', 'FIFA World Cup']

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directorio creado: {path}")

def download_match_events(match_row):
    """Descarga los eventos de un partido y los guarda en un archivo Parquet."""
    match_id = match_row['match_id']
    home_team = match_row['home_team']
    away_team = match_row['away_team']
    output_file = os.path.join(RAW_DATA_DIR, f"{match_id}_events.parquet")
    
    try:
        # Descargar eventos del partido
        events_df = sb.events(match_id=match_id)
        
        # Convertir columnas complejas (listas/diccionarios) a string para Parquet
        for col in events_df.columns:
            if events_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                events_df[col] = events_df[col].astype(str)
        
        # Guardar en parquet
        events_df.to_parquet(output_file, engine='fastparquet', index=False)
        return match_id, True, None
    except Exception as e:
        return match_id, False, str(e)

def ingest_statsbomb_data():
    create_directory_if_not_exists(RAW_DATA_DIR)
    
    logger.info("Obteniendo lista de competiciones...")
    try:
        competitions = sb.competitions()
    except Exception as e:
        logger.error(f"Error al obtener competiciones: {e}")
        return

    # Filtrar por las competiciones especificadas
    selected_competitions = competitions[competitions['competition_name'].isin(COMPETITIONS)]
    
    if selected_competitions.empty:
        logger.warning(f"No se encontraron competiciones coincidentes.")
        return
    
    logger.info(f"Se encontraron {len(selected_competitions)} temporadas/competiciones a procesar.")
    
    # Lista para almacenar los metadatos de todos los partidos
    all_matches_list = []
    
    # Obtener metadatos de todos los partidos para cada competición y temporada
    for _, comp_row in selected_competitions.iterrows():
        comp_id = comp_row['competition_id']
        season_id = comp_row['season_id']
        season_name = comp_row['season_name']
        comp_name = comp_row['competition_name']
        
        logger.info(f"Obteniendo partidos de {comp_name} ({season_name}) (Comp ID: {comp_id}, Season ID: {season_id})")
        
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            all_matches_list.append(matches)
        except Exception as e:
            logger.error(f"Error obteniendo partidos para {comp_name} ({season_name}): {e}")
            continue
            
    if not all_matches_list:
        logger.warning("No se encontraron partidos para procesar.")
        return
        
    # Consolidar metadatos
    all_matches_df = pd.concat(all_matches_list, ignore_index=True)
    all_matches_df = all_matches_df.drop_duplicates(subset=['match_id'])
    
    # Filtrar los partidos que no han sido descargados aún
    matches_to_download = []
    for _, match_row in all_matches_df.iterrows():
        match_id = match_row['match_id']
        output_file = os.path.join(RAW_DATA_DIR, f"{match_id}_events.parquet")
        if not os.path.exists(output_file):
            matches_to_download.append(match_row)
            
    total_matches = len(all_matches_df)
    total_to_download = len(matches_to_download)
    logger.info(f"Total de partidos únicos encontrados: {total_matches}")
    logger.info(f"Partidos ya descargados anteriormente: {total_matches - total_to_download}")
    logger.info(f"Partidos nuevos pendientes de descarga: {total_to_download}")
    
    downloaded_count = 0
    failed_count = 0
    
    if total_to_download > 0:
        logger.info("Iniciando descarga en paralelo usando 16 hilos...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Programar descargas
            futures = {executor.submit(download_match_events, row): row for row in matches_to_download}
            
            # Procesar descargas a medida que terminan
            for future in as_completed(futures):
                match_row = futures[future]
                match_id = match_row['match_id']
                home = match_row['home_team']
                away = match_row['away_team']
                
                res_match_id, success, error_msg = future.result()
                if success:
                    downloaded_count += 1
                    if downloaded_count % 50 == 0 or downloaded_count == total_to_download:
                        logger.info(f"[PROGRESO] Descargados {downloaded_count}/{total_to_download} partidos.")
                else:
                    failed_count += 1
                    logger.error(f"Error descargando partido {match_id} ({home} vs {away}): {error_msg}")
                    
    # Guardar metadatos consolidados de los partidos
    logger.info("Guardando metadatos consolidados de los partidos...")
    
    # Filtrar solo las columnas necesarias para evitar errores de tipo en fastparquet con columnas complejas
    cols_to_keep = [
        'match_id', 'match_date', 'home_team', 'away_team', 
        'home_score', 'away_score', 'competition_name', 
        'season_name', 'competition_stage'
    ]
    cols_to_keep = [c for c in cols_to_keep if c in all_matches_df.columns]
    all_matches_df = all_matches_df[cols_to_keep]
            
    matches_output_path = '../data/raw/statsbomb/matches.parquet'
    all_matches_df.to_parquet(matches_output_path, engine='fastparquet', index=False)
    logger.info(f"Metadatos guardados con éxito en: {matches_output_path}")
    
    logger.info(f"Ingesta finalizada. Se descargaron {downloaded_count} partidos nuevos. Fallidos: {failed_count}.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logger.info("Iniciando ingesta multilingue/multicompetición optimizada.")
    ingest_statsbomb_data()
