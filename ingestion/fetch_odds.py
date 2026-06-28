import os
import sys
import pandas as pd
import requests
import logging

# Ensure we can import from core
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from core.team_mapping import normalize_team_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']
LEAGUES = ['E0', 'E1', 'SP1', 'SP2', 'I1', 'I2', 'D1', 'D2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1']
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

DATASET_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_dataset.parquet')
OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'processed', 'matches_with_odds.parquet')

def fetch_and_merge_odds():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"El dataset principal no existe en {DATASET_PATH}")
        return

    df_main = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    
    odds_dataframes = []
    
    logger.info("Iniciando descarga de cuotas históricas desde football-data.co.uk...")
    for season in SEASONS:
        for league in LEAGUES:
            url = BASE_URL.format(season=season, league=league)
            try:
                # Some old CSVs have encoding issues (latin-1)
                df_odds = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
                
                # Standarize Date
                if 'Date' in df_odds.columns:
                    df_odds['Date'] = pd.to_datetime(df_odds['Date'], format='mixed', dayfirst=True)
                else:
                    continue
                    
                # We want B365 and Pinnacle odds if available
                cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam']
                
                # Check for B365
                b365_cols = ['B365H', 'B365D', 'B365A']
                if all(c in df_odds.columns for c in b365_cols):
                    cols_to_keep.extend(b365_cols)
                    
                # Check for Pinnacle Opening
                pin_cols = ['PSH', 'PSD', 'PSA']
                if all(c in df_odds.columns for c in pin_cols):
                    cols_to_keep.extend(pin_cols)
                    
                # Check for Pinnacle Closing
                pin_ch_cols = ['PSCH', 'PSCD', 'PSCA']
                if all(c in df_odds.columns for c in pin_ch_cols):
                    cols_to_keep.extend(pin_ch_cols)
                    
                df_odds = df_odds[cols_to_keep].copy()
                
                # Normalize team names
                df_odds['HomeTeam'] = df_odds['HomeTeam'].apply(normalize_team_name)
                df_odds['AwayTeam'] = df_odds['AwayTeam'].apply(normalize_team_name)
                
                odds_dataframes.append(df_odds)
                
            except Exception as e:
                logger.warning(f"Error descargando {url}: {e}")
                
    if not odds_dataframes:
        logger.error("No se pudo descargar ningún archivo de cuotas.")
        return
        
    df_odds_master = pd.concat(odds_dataframes, ignore_index=True)
    df_odds_master = df_odds_master.dropna(subset=['HomeTeam', 'AwayTeam'])
    
    logger.info(f"Descargados {len(df_odds_master)} partidos con cuotas.")
    
    # Preparamos df_main para el merge. 
    # El df_main está estructurado un registro por equipo por partido.
    df_main['match_date_only'] = pd.to_datetime(df_main['match_date']).dt.normalize()
    
    logger.info("Realizando merge de cuotas con dataset principal...")
    
    def get_odds(row):
        # We look up in df_odds_master
        is_home = row['is_home'] == 1
        team = row['team']
        opp = row['opponent']
        date = row['match_date_only']
        
        home = team if is_home else opp
        away = opp if is_home else team
        
        # We allow a small +/- 2 days window in case of date mismatches due to timezone
        mask = (
            (df_odds_master['HomeTeam'] == home) & 
            (df_odds_master['AwayTeam'] == away) & 
            (abs((df_odds_master['Date'] - date).dt.days) <= 2)
        )
        
        match = df_odds_master[mask]
        
        if len(match) > 0:
            m = match.iloc[0]
            # Extraemos Opening Odds (Prioridad: PSH -> B365H)
            if is_home:
                open_w = m.get('PSH', m.get('B365H', np.nan))
                open_d = m.get('PSD', m.get('B365D', np.nan))
                open_l = m.get('PSA', m.get('B365A', np.nan))
                
                close_w = m.get('PSCH', open_w)
                close_d = m.get('PSCD', open_d)
                close_l = m.get('PSCA', open_l)
            else:
                open_w = m.get('PSA', m.get('B365A', np.nan))
                open_d = m.get('PSD', m.get('B365D', np.nan))
                open_l = m.get('PSH', m.get('B365H', np.nan))
                
                close_w = m.get('PSCA', open_w)
                close_d = m.get('PSCD', open_d)
                close_l = m.get('PSCH', open_l)
                
            # Calcular probabilidades implicitas quitando el Vig (Overround)
            # Solo si no son nulos
            if pd.notna(open_w) and pd.notna(open_d) and pd.notna(open_l):
                inv_w = 1.0 / open_w
                inv_d = 1.0 / open_d
                inv_l = 1.0 / open_l
                margin = inv_w + inv_d + inv_l
                
                prob_w = inv_w / margin
                prob_d = inv_d / margin
                prob_l = inv_l / margin
            else:
                prob_w, prob_d, prob_l = np.nan, np.nan, np.nan
                
            # Mantener compatibilidad: odds_win, odds_draw, odds_loss representarán closing
            return pd.Series({
                'odds_win': close_w, 'odds_draw': close_d, 'odds_loss': close_l,
                'open_odds_win': open_w, 'open_odds_draw': open_d, 'open_odds_loss': open_l,
                'open_prob_win': prob_w, 'open_prob_draw': prob_d, 'open_prob_loss': prob_l
            })
            
        return pd.Series({
            'odds_win': np.nan, 'odds_draw': np.nan, 'odds_loss': np.nan,
            'open_odds_win': np.nan, 'open_odds_draw': np.nan, 'open_odds_loss': np.nan,
            'open_prob_win': np.nan, 'open_prob_draw': np.nan, 'open_prob_loss': np.nan
        })

    import numpy as np
    odds_series = df_main.apply(get_odds, axis=1)
    df_main = pd.concat([df_main, odds_series], axis=1)
    
    df_main = df_main.drop(columns=['match_date_only'])
    
    matches_with_odds = df_main.dropna(subset=['odds_win'])
    logger.info(f"Dataset principal tiene {len(df_main)} filas. Filas con cuotas encontradas: {len(matches_with_odds)} ({(len(matches_with_odds)/len(df_main))*100:.1f}%)")
    
    df_main.to_parquet(OUTPUT_PATH, engine='fastparquet')
    logger.info(f"Nuevo dataset guardado en {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_and_merge_odds()
