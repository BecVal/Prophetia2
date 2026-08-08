import os
import sys
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure we can import from core
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from core.team_mapping import normalize_team_name

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'fetch_odds')


SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']
LEAGUES = ['E0', 'E1', 'SP1', 'SP2', 'I1', 'I2', 'D1', 'D2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1', 'SC0', 'E2']
EXTRA_LEAGUES = ['USA', 'JPN', 'SWE', 'NOR', 'DNK', 'SWZ', 'AUT', 'MEX']
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_BASE_URL = "https://www.football-data.co.uk/new/{league}.csv"

DATASET_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_dataset.parquet')
OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_with_odds.parquet')

def fetch_and_merge_odds():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"El dataset principal no existe en {DATASET_PATH}")
        return

    df_main = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    
    odds_dataframes = []
    
    logger.info("Iniciando descarga de cuotas históricas concurrentemente...")
    
    def process_odds_df(url):
        try:
            df_odds = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            
            if 'Home' in df_odds.columns and 'HomeTeam' not in df_odds.columns:
                df_odds.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'}, inplace=True)
                
            if 'Date' in df_odds.columns:
                df_odds['Date'] = pd.to_datetime(df_odds['Date'], format='mixed', dayfirst=True)
            else:
                return None
                
            cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam']
            potential_odds_cols = [
                'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA',
                'B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA',
                'AvgH', 'AvgD', 'AvgA', 'AvgCH', 'AvgCD', 'AvgCA',
                'MaxH', 'MaxD', 'MaxA', 'MaxCH', 'MaxCD', 'MaxCA',
                'WHH', 'WHD', 'WHA', 'IWH', 'IWD', 'IWA', 'BWH', 'BWD', 'BWA'
            ]
            for c in potential_odds_cols:
                if c in df_odds.columns:
                    cols_to_keep.append(c)
                
            df_odds = df_odds[cols_to_keep].copy()
            df_odds['HomeTeam'] = df_odds['HomeTeam'].apply(normalize_team_name)
            df_odds['AwayTeam'] = df_odds['AwayTeam'].apply(normalize_team_name)
            return df_odds
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
        futures = {executor.submit(process_odds_df, url): url for url in urls}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                odds_dataframes.append(res)
            
    if not odds_dataframes:
        logger.error("No se pudo descargar ningún archivo de cuotas.")
        return
        
    df_odds_master = pd.concat(odds_dataframes, ignore_index=True)
    df_odds_master = df_odds_master.dropna(subset=['HomeTeam', 'AwayTeam'])
    logger.info(f"Descargados {len(df_odds_master)} partidos con cuotas.")
    
    # 1. Extracción jerárquica de Cuotas de Cierre (Closing Odds)
    # Prioridad: Pinnacle Closing (PSCH) -> Bet365 Closing (B365CH) -> Avg Closing (AvgCH) -> Max Closing (MaxCH)
    def extract_composite_odds(df, priority_cols):
        res = pd.Series(np.nan, index=df.index)
        for col in priority_cols:
            if col in df.columns:
                res = res.fillna(pd.to_numeric(df[col], errors='coerce'))
        return res

    close_win = extract_composite_odds(df_odds_master, ['PSCH', 'B365CH', 'AvgCH', 'MaxCH', 'PSH', 'B365H', 'AvgH', 'MaxH'])
    close_draw = extract_composite_odds(df_odds_master, ['PSCD', 'B365CD', 'AvgCD', 'MaxCD', 'PSD', 'B365D', 'AvgD', 'MaxD'])
    close_loss = extract_composite_odds(df_odds_master, ['PSCA', 'B365CA', 'AvgCA', 'MaxCA', 'PSA', 'B365A', 'AvgA', 'MaxA'])

    # 2. Extracción jerárquica de Cuotas de Apertura (Opening Odds)
    # Prioridad: Pinnacle Open (PSH) -> Bet365 Open (B365H) -> Avg Open (AvgH) -> Max Open (MaxH) -> WHH/BWH/IWH -> Fallback a Closing
    open_win = extract_composite_odds(df_odds_master, ['PSH', 'B365H', 'AvgH', 'MaxH', 'WHH', 'BWH', 'IWH', 'PSCH', 'B365CH', 'AvgCH', 'MaxCH'])
    open_draw = extract_composite_odds(df_odds_master, ['PSD', 'B365D', 'AvgD', 'MaxD', 'WHD', 'BWD', 'IWD', 'PSCD', 'B365CD', 'AvgCD', 'MaxCD'])
    open_loss = extract_composite_odds(df_odds_master, ['PSA', 'B365A', 'AvgA', 'MaxA', 'WHA', 'BWA', 'IWA', 'PSCA', 'B365CA', 'AvgCA', 'MaxCA'])

    # Fallback bidireccional estricto (Open <-> Close)
    open_win = open_win.fillna(close_win)
    open_draw = open_draw.fillna(close_draw)
    open_loss = open_loss.fillna(close_loss)

    close_win = close_win.fillna(open_win)
    close_draw = close_draw.fillna(open_draw)
    close_loss = close_loss.fillna(open_loss)

    df_odds_master['open_w_home'] = open_win
    df_odds_master['open_d'] = open_draw
    df_odds_master['open_l_home'] = open_loss
    df_odds_master['close_w_home'] = close_win
    df_odds_master['close_d'] = close_draw
    df_odds_master['close_l_home'] = close_loss

    cols_master = ['Date', 'HomeTeam', 'AwayTeam', 
                   'open_w_home', 'open_d', 'open_l_home',
                   'close_w_home', 'close_d', 'close_l_home']
    df_odds_master = df_odds_master[cols_master]
    
    # Sort by date and drop duplicates per match just in case
    df_odds_master = df_odds_master.sort_values('Date').drop_duplicates(subset=['HomeTeam', 'AwayTeam', 'Date'])

    logger.info("Realizando merge vectorizado con dataset principal...")
    df_main['match_date_only'] = pd.to_datetime(df_main['match_date']).dt.normalize()
    
    is_home = df_main['is_home'] == 1
    df_main['HomeTeam_join'] = np.where(is_home, df_main['team'], df_main['opponent'])
    df_main['AwayTeam_join'] = np.where(is_home, df_main['opponent'], df_main['team'])

    # Evitamos colisiones de columnas
    cols_to_drop = [c for c in ['odds_win', 'odds_draw', 'odds_loss', 'open_odds_win', 'open_odds_draw', 'open_odds_loss'] if c in df_main.columns]
    if cols_to_drop:
        df_main.drop(columns=cols_to_drop, inplace=True)
    
    # Agregar ID temporal para asegurar que no se multipliquen las filas (Producto cartesiano)
    df_main['_row_id'] = np.arange(len(df_main))
    
    # Como queremos permitir +/- 2 dias, hacemos merge por equipos.
    # ATENCION: Esto generará un producto cartesiano (si Real Madrid y Barça jugaron 20 veces en la decada, se multiplicarán).
    merged = pd.merge(
        df_main,
        df_odds_master,
        left_on=['HomeTeam_join', 'AwayTeam_join'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    # Calcular diferencia de días
    merged['date_diff'] = abs((merged['match_date_only'] - merged['Date']).dt.days)
    merged['is_valid_date'] = merged['date_diff'] <= 2
    
    # Ordenar para que cada partido mantenga su coincidencia más cercana en fechas como primera opción
    merged = merged.sort_values(['_row_id', 'is_valid_date', 'date_diff'], ascending=[True, False, True])
    
    # Eliminar duplicados del producto cartesiano, manteniendo solo 1 fila exacta por cada fila original de df_main
    merged = merged.drop_duplicates(subset=['_row_id']).sort_values('_row_id')
    
    # Asignar NaNs a las cuotas donde la fecha no cuadra o no hubo match
    invalid = ~merged['is_valid_date']
    odds_cols = ['open_w_home', 'open_d', 'open_l_home', 'close_w_home', 'close_d', 'close_l_home']
    for c in odds_cols:
        merged.loc[invalid, c] = np.nan
        
    is_home_mask = merged['is_home'] == 1
    
    # Asignación relativa al equipo en perspectiva
    merged['odds_win'] = np.where(is_home_mask, merged['close_w_home'], merged['close_l_home'])
    merged['odds_draw'] = merged['close_d']
    merged['odds_loss'] = np.where(is_home_mask, merged['close_l_home'], merged['close_w_home'])
    
    merged['open_odds_win'] = np.where(is_home_mask, merged['open_w_home'], merged['open_l_home'])
    merged['open_odds_draw'] = merged['open_d']
    merged['open_odds_loss'] = np.where(is_home_mask, merged['open_l_home'], merged['open_w_home'])
    
    df_final = merged.drop(columns=[
        'match_date_only', 'HomeTeam_join', 'AwayTeam_join', 
        'Date', 'HomeTeam', 'AwayTeam',
        'open_w_home', 'open_d', 'open_l_home',
        'close_w_home', 'close_d', 'close_l_home',
        '_row_id', 'date_diff', 'is_valid_date'
    ])
    
    matches_with_odds = df_final.dropna(subset=['odds_win'])
    partidos_unicos = len(matches_with_odds) // 2
    logger.info(f"Dataset principal tiene {len(df_final)} filas reales (perspectiva por equipo). Partidos únicos con cuotas integradas: {partidos_unicos} ({(len(matches_with_odds)/len(df_final))*100:.1f}% de filas cubiertas)")
    
    df_final.to_parquet(OUTPUT_PATH, engine='fastparquet')
    logger.info(f"Nuevo dataset guardado en {OUTPUT_PATH}")

if __name__ == '__main__':
    fetch_and_merge_odds()
