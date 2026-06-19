import os
import time
import pandas as pd
from statsbombpy import sb
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = '../data/raw/statsbomb/events/'
COMPETITION_NAME = 'Champions League'

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directorio creado: {path}")

def ingest_champions_league_data(limit=None):
    create_directory_if_not_exists(RAW_DATA_DIR)
    
    logger.info("Obteniendo lista de competiciones...")
    try:
        competitions = sb.competitions()
    except Exception as e:
        logger.error(f"Error al obtener competiciones: {e}")
        return

    # Filtrar por Champions League
    cl_competitions = competitions[competitions['competition_name'] == COMPETITION_NAME]
    
    if cl_competitions.empty:
        logger.warning(f"No se encontraron competiciones con el nombre '{COMPETITION_NAME}'")
        return
    
    logger.info(f"Se encontraron {len(cl_competitions)} temporadas de la {COMPETITION_NAME}")
    
    total_matches_downloaded = 0
    
    # Lista para almacenar los metadatos de todos los partidos
    all_matches_list = []
    
    # Iterar sobre las temporadas de la Champions
    for _, comp_row in cl_competitions.iterrows():
        if limit and total_matches_downloaded >= limit:
            break
            
        comp_id = comp_row['competition_id']
        season_id = comp_row['season_id']
        season_name = comp_row['season_name']
        
        logger.info(f"Procesando temporada {season_name} (Comp ID: {comp_id}, Season ID: {season_id})")
        
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            all_matches_list.append(matches)
        except Exception as e:
            logger.error(f"Error obteniendo partidos para la temporada {season_name}: {e}")
            continue
            
        logger.info(f"  Encontrados {len(matches)} partidos en la temporada {season_name}.")
        
        for _, match_row in matches.iterrows():
            if limit and total_matches_downloaded >= limit:
                break
                
            match_id = match_row['match_id']
            match_date = match_row['match_date']
            home_team = match_row['home_team']
            away_team = match_row['away_team']
            
            output_file = os.path.join(RAW_DATA_DIR, f"{match_id}_events.parquet")
            
            if os.path.exists(output_file):
                logger.info(f"  [OMITIDO] Partido {match_id} ({home_team} vs {away_team}) ya descargado.")
                continue
                
            logger.info(f"  [DESCARGANDO] Partido {match_id}: {home_team} vs {away_team} ({match_date})")
            
            try:
                # Descargar eventos del partido
                events_df = sb.events(match_id=match_id)
                
                # Convertir columnas que puedan ser listas/diccionarios a string para que fastparquet/pyarrow no falle
                for col in events_df.columns:
                    if events_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        events_df[col] = events_df[col].astype(str)
                
                # Guardar en parquet
                events_df.to_parquet(output_file, engine='fastparquet', index=False)
                total_matches_downloaded += 1
                
                # Sleep pequeño para no saturar la API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  Error descargando/guardando partido {match_id}: {e}")
                
    # Guardar metadatos consolidados de los partidos
    if all_matches_list:
        logger.info("Guardando metadatos consolidados de los partidos...")
        all_matches_df = pd.concat(all_matches_list, ignore_index=True)
        
        # Convertir columnas anidadas (ej. managers) a string para Parquet
        for col in all_matches_df.columns:
            if all_matches_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                all_matches_df[col] = all_matches_df[col].astype(str)
                
        matches_output_path = '../data/raw/statsbomb/matches.parquet'
        all_matches_df.to_parquet(matches_output_path, engine='fastparquet', index=False)
        logger.info(f"Metadatos guardados con éxito en: {matches_output_path}")
        
    logger.info(f"Ingesta finalizada. Se descargaron eventos de {total_matches_downloaded} partidos nuevos.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # En producción o cuando el usuario quiera todo, se puede quitar el límite estableciéndolo a None.
    logger.info("Iniciando ingesta completa de Champions League.")
    ingest_champions_league_data(limit=None)
