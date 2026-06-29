import os
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de ligas y temporadas a descargar

LEAGUES = ['E0', 'E1', 'SP1', 'SP2', 'I1', 'I2', 'D1', 'D2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1']
SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
OUTPUT_DIR = '../data/raw/football_data'

def download_football_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Creado directorio: {OUTPUT_DIR}")

    for season in SEASONS:
        for league in LEAGUES:
            url = BASE_URL.format(season=season, league=league)
            output_file = os.path.join(OUTPUT_DIR, f"{league}_{season}.csv")
            
            # Si ya existe, no lo descargamos de nuevo
            if os.path.exists(output_file):
                logger.info(f"El archivo {output_file} ya existe. Omitiendo.")
                continue
                
            logger.info(f"Descargando {url}...")
            try:
                response = requests.get(url)
                response.raise_for_status() # Lanza excepción si el status no es 200
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Guardado: {output_file}")
            except Exception as e:
                logger.error(f"Error descargando {url}: {e}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logger.info("Iniciando descarga de datos de Football-Data.co.uk...")
    download_football_data()
    logger.info("Descarga finalizada.")
