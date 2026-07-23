import os
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from core.team_mapping import normalize_team_name
from core.logger_config import get_logger

logger = get_logger(__name__, 'fetch_referees')

SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']
LEAGUES = ['E0', 'E1', 'SP1', 'SP2', 'I1', 'I2', 'D1', 'D2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1', 'SC0', 'E2']
EXTRA_LEAGUES = ['USA', 'JPN', 'SWE', 'NOR', 'DNK', 'SWZ', 'AUT']
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_BASE_URL = "https://www.football-data.co.uk/new/{league}.csv"

DATASET_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_with_odds.parquet')
OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_with_referees.parquet')

def fetch_and_merge_referees():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"El dataset principal no existe en {DATASET_PATH}")
        return

    df_main = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    
    ref_dataframes = []
    
    logger.info("Iniciando descarga de datos de árbitros concurrentemente...")
    
    def process_ref_df(url):
        try:
            df_csv = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            
            if 'Home' in df_csv.columns and 'HomeTeam' not in df_csv.columns:
                df_csv.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'}, inplace=True)
                
            if 'Date' not in df_csv.columns or 'Referee' not in df_csv.columns:
                return None
                
            df_csv['Date'] = pd.to_datetime(df_csv['Date'], format='mixed', dayfirst=True)
            
            cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'Referee']
            # Opcionales pero utiles para calcular la severidad
            stats_cols = ['HF', 'AF', 'HY', 'AY', 'HR', 'AR']
            
            # Solo guardamos si tienen al menos las stats de tarjetas y faltas
            for col in stats_cols:
                if col not in df_csv.columns:
                    df_csv[col] = np.nan
                    
            cols_to_keep.extend(stats_cols)
                
            df_csv = df_csv[cols_to_keep].copy()
            df_csv['HomeTeam'] = df_csv['HomeTeam'].apply(normalize_team_name)
            df_csv['AwayTeam'] = df_csv['AwayTeam'].apply(normalize_team_name)
            df_csv['Referee'] = df_csv['Referee'].astype(str).str.strip()
            # Reemplazamos los 'nan' string por np.nan real
            df_csv['Referee'] = df_csv['Referee'].replace('nan', np.nan)
            
            return df_csv.dropna(subset=['Referee'])
        except Exception as e:
            logger.warning(f"Error descargando {url}: {e}")
            return None

    urls = []
    for season in SEASONS:
        for league in LEAGUES:
            urls.append(BASE_URL.format(season=season, league=league))
    for league in EXTRA_LEAGUES:
        urls.append(EXTRA_BASE_URL.format(league=league))
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_ref_df, url): url for url in urls}
        for future in as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                ref_dataframes.append(res)
            
    if not ref_dataframes:
        logger.error("No se pudo descargar ningún archivo de árbitros.")
        return
        
    df_ref_master = pd.concat(ref_dataframes, ignore_index=True)
    df_ref_master = df_ref_master.sort_values('Date').drop_duplicates(subset=['HomeTeam', 'AwayTeam', 'Date'])
    
    logger.info(f"Descargados {len(df_ref_master)} partidos con árbitros.")

    # Calcular métricas históricas de los árbitros
    df_ref_master['total_fouls'] = df_ref_master['HF'].fillna(0) + df_ref_master['AF'].fillna(0)
    df_ref_master['total_yellows'] = df_ref_master['HY'].fillna(0) + df_ref_master['AY'].fillna(0)
    df_ref_master['total_reds'] = df_ref_master['HR'].fillna(0) + df_ref_master['AR'].fillna(0)
    
    # Calcular promedios móviles expandidos (shifted por 1 para no incluir el partido actual)
    df_ref_master = df_ref_master.sort_values(by=['Referee', 'Date'])
    
    df_ref_master['referee_avg_fouls'] = df_ref_master.groupby('Referee')['total_fouls'].transform(lambda x: x.expanding().mean().shift(1))
    df_ref_master['referee_avg_yellows'] = df_ref_master.groupby('Referee')['total_yellows'].transform(lambda x: x.expanding().mean().shift(1))
    df_ref_master['referee_avg_reds'] = df_ref_master.groupby('Referee')['total_reds'].transform(lambda x: x.expanding().mean().shift(1))
    
    # Índice de severidad (cuántas faltas permite por cada tarjeta amarilla)
    # Menor = más estricto
    df_ref_master['referee_fouls_per_yellow'] = np.where(
        df_ref_master['referee_avg_yellows'] > 0, 
        df_ref_master['referee_avg_fouls'] / df_ref_master['referee_avg_yellows'], 
        np.nan
    )
    
    # Rellenar nulos con el promedio global para árbitros debutantes
    global_avg_fouls = df_ref_master['total_fouls'].mean()
    global_avg_yellows = df_ref_master['total_yellows'].mean()
    global_avg_reds = df_ref_master['total_reds'].mean()
    
    df_ref_master['referee_avg_fouls'] = df_ref_master['referee_avg_fouls'].fillna(global_avg_fouls)
    df_ref_master['referee_avg_yellows'] = df_ref_master['referee_avg_yellows'].fillna(global_avg_yellows)
    df_ref_master['referee_avg_reds'] = df_ref_master['referee_avg_reds'].fillna(global_avg_reds)
    
    global_strictness = global_avg_fouls / global_avg_yellows if global_avg_yellows > 0 else 5.0
    df_ref_master['referee_fouls_per_yellow'] = df_ref_master['referee_fouls_per_yellow'].fillna(global_strictness)
    
    cols_to_merge = ['Date', 'HomeTeam', 'AwayTeam', 'Referee', 
                     'referee_avg_fouls', 'referee_avg_yellows', 'referee_avg_reds', 'referee_fouls_per_yellow']
    df_ref_master = df_ref_master[cols_to_merge]
    
    logger.info("Realizando merge con dataset principal...")
    
    df_main['match_date_only'] = pd.to_datetime(df_main['match_date']).dt.normalize()
    is_home = df_main['is_home'] == 1
    df_main['HomeTeam_join'] = np.where(is_home, df_main['team'], df_main['opponent'])
    df_main['AwayTeam_join'] = np.where(is_home, df_main['opponent'], df_main['team'])
    
    df_main['_row_id'] = np.arange(len(df_main))
    
    merged = pd.merge(
        df_main,
        df_ref_master,
        left_on=['HomeTeam_join', 'AwayTeam_join'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    merged['date_diff'] = abs((merged['match_date_only'] - merged['Date']).dt.days)
    merged['is_valid_date'] = merged['date_diff'] <= 2
    
    merged = merged.sort_values(['_row_id', 'is_valid_date', 'date_diff'], ascending=[True, False, True])
    merged = merged.drop_duplicates(subset=['_row_id']).sort_values('_row_id')
    
    # Asignar NaNs a las features de árbitros donde no hizo match válido
    invalid = ~merged['is_valid_date']
    ref_cols = ['Referee', 'referee_avg_fouls', 'referee_avg_yellows', 'referee_avg_reds', 'referee_fouls_per_yellow']
    for c in ref_cols:
        merged.loc[invalid, c] = np.nan
        
    df_final = merged.drop(columns=[
        'match_date_only', 'HomeTeam_join', 'AwayTeam_join', 
        'Date', 'HomeTeam', 'AwayTeam', '_row_id', 'date_diff', 'is_valid_date'
    ])
    
    cobertura = df_final['Referee'].notna().mean() * 100
    logger.info(f"Dataset final con árbitros generado. Cobertura de árbitros: {cobertura:.1f}%")
    
    df_final.to_parquet(OUTPUT_PATH, engine='fastparquet')
    logger.info(f"Nuevo dataset guardado en {OUTPUT_PATH}")

if __name__ == '__main__':
    fetch_and_merge_referees()
