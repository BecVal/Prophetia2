import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from core.team_mapping import normalize_team_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map football-data.co.uk league codes to Transfermarkt details (url_name, league_id)
TRANSFERMARKT_LEAGUES = {
    'E0': ('premier-league', 'GB1'),
    'E1': ('championship', 'GB2'),
    'SP1': ('laliga', 'ES1'),
    'SP2': ('laliga2', 'ES2'),
    'I1': ('serie-a', 'IT1'),
    'I2': ('serie-b', 'IT2'),
    'D1': ('bundesliga', 'L1'),
    'D2': ('2-bundesliga', 'L2'),
    'F1': ('ligue-1', 'FR1'),
    'F2': ('ligue-2', 'FR2'),
    'N1': ('eredivisie', 'NL1'),
    'B1': ('jupiler-pro-league', 'BE1'),
    'P1': ('liga-portugal', 'PO1'),
    'T1': ('super-lig', 'TR1'),
    'G1': ('super-league-1', 'GR1'),
}

# The years in transfermarkt map to the start of the season (e.g. 2014 for 1415 season)
# We will use 2014 to 2025
SEASONS_YEARS = list(range(2014, 2026)) 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Referer': 'https://www.google.com/'
}

OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'raw', 'transfermarkt_squad_values.parquet')

def parse_currency_to_millions(value_str):
    """
    Convierte un string como '€1.05bn' o '€450.00m' a un float en millones (1050.0, 450.0).
    """
    if not isinstance(value_str, str) or value_str.strip() == '-' or not value_str:
        return 0.0
    
    value_str = value_str.replace('€', '').strip()
    multiplier = 1.0
    
    if 'bn' in value_str:
        multiplier = 1000.0
        value_str = value_str.replace('bn', '')
    elif 'm' in value_str:
        multiplier = 1.0
        value_str = value_str.replace('m', '')
    elif 'k' in value_str:
        multiplier = 0.001
        value_str = value_str.replace('k', '')
        
    try:
        return float(value_str) * multiplier
    except ValueError:
        return 0.0

def fetch_squad_values():
    results = []
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    for league_code, (league_name, tm_league_id) in TRANSFERMARKT_LEAGUES.items():
        logger.info(f"Procesando liga {league_code} ({league_name})...")
        
        for year in SEASONS_YEARS:
            # Transfermarkt URL format
            url = f"https://www.transfermarkt.com/{league_name}/startseite/wettbewerb/{tm_league_id}/plus/?saison_id={year}"
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=HEADERS, timeout=30)
                    if response.status_code != 200:
                        logger.warning(f"Error {response.status_code} al acceder a {url}")
                        time.sleep(3)
                        continue
                        
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # The table containing the teams
                    table = soup.find('table', class_='items')
                    if not table:
                        logger.warning(f"No se encontró la tabla de equipos en {url}")
                        break # Break retry loop if page loads but no table
                        
                    tbody = table.find('tbody')
                    if not tbody:
                        break
                        
                    rows = tbody.find_all('tr')
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) < 7:
                            continue
                            
                        team_cell = row.find('td', class_='hauptlink')
                        if not team_cell:
                            continue
                            
                        raw_team_name = team_cell.get_text(strip=True)
                        
                        value_cell = row.find_all('td', class_='rechts')
                        if not value_cell:
                            continue
                            
                        raw_value = cols[-1].get_text(strip=True)
                        
                        if '€' not in raw_value and len(value_cell) > 1:
                            raw_value = value_cell[-1].get_text(strip=True)
                        
                        value_millions = parse_currency_to_millions(raw_value)
                        normalized_name = normalize_team_name(raw_team_name)
                        
                        results.append({
                            'season_year': year,
                            'league_code': league_code,
                            'tm_team_name': raw_team_name,
                            'team': normalized_name,
                            'squad_value_millions': value_millions
                        })
                        
                    # Si llegamos aquí, fue exitoso, salimos del retry loop
                    time.sleep(1.5)
                    break
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout en {url}. Reintento {attempt + 1}/{max_retries}...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error procesando {url}: {e}")
                    time.sleep(5)
        
        # Guardado progresivo al terminar cada liga
        if results:
            df = pd.DataFrame(results)
            df.to_parquet(OUTPUT_PATH, engine='fastparquet', index=False)
            logger.info(f"Guardado progresivo: {len(df)} registros acumulados en {OUTPUT_PATH}")
                
    if not results:
        logger.warning("No se extrajo ningún dato de Transfermarkt.")

if __name__ == "__main__":
    fetch_squad_values()
