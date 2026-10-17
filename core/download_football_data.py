import os
import requests
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'download_football_data')

# Configuración de ligas y temporadas a descargar
LEAGUES = ['E0', 'E1', 'SP1', 'SP2', 'I1', 'I2', 'D1', 'D2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1', 'SC0', 'E2']
EXTRA_LEAGUES = ['USA', 'JPN', 'SWE', 'NOR', 'DNK', 'SWZ', 'AUT', 'MEX']
SEASONS = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526', '2627']
ACTIVE_SEASONS = ['2425', '2526', '2627']

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_BASE_URL = "https://www.football-data.co.uk/new/{league}.csv"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/football_data'))

def download_football_data(force_refresh_current=True):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Creado directorio: {OUTPUT_DIR}")

    for season in SEASONS:
        is_active = season in ACTIVE_SEASONS
        for league in LEAGUES:
            url = BASE_URL.format(season=season, league=league)
            output_file = os.path.join(OUTPUT_DIR, f"{league}_{season}.csv")
            
            # Si ya existe y no es temporada activa ni forzado, omitir
            if os.path.exists(output_file) and not (force_refresh_current and is_active):
                continue
                
            logger.info(f"Descargando {league}_{season} desde {url}...")
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Guardado: {output_file}")
                elif response.status_code == 404 and is_active:
                    # Temporada futura aún no comenzada
                    pass
                else:
                    logger.warning(f"Status {response.status_code} al descargar {url}")
            except Exception as e:
                logger.error(f"Error descargando {url}: {e}")

    # Descargar ligas extra (siempre activas/actualizadas)
    for league in EXTRA_LEAGUES:
        url = EXTRA_BASE_URL.format(league=league)
        output_file = os.path.join(OUTPUT_DIR, f"{league}.csv")
        if os.path.exists(output_file) and not force_refresh_current:
            continue
            
        logger.info(f"Descargando liga extra {league} desde {url}...")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Guardado liga extra: {output_file}")
            else:
                logger.warning(f"Status {response.status_code} al descargar liga extra {url}")
        except Exception as e:
            logger.error(f"Error descargando liga extra {url}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Descargador de datos históricos y recientes de football-data.co.uk")
    parser.add_argument('--force-refresh', action='store_true', default=True, help="Refrescar temporadas activas y ligas extra")
    parser.add_argument('--enrich-espn', action='store_true', help="Enriquecer ligas extra (Liga MX, MLS, etc.) con estadísticas de ESPN FC")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logger.info("Iniciando descarga y sincronización de datos de Football-Data.co.uk...")
    download_football_data(force_refresh_current=args.force_refresh)
    logger.info("Descarga y sincronización finalizada con éxito.")

    if args.enrich_espn:
        logger.info("Iniciando enriquecimiento con estadísticas de ESPN FC...")
        try:
            from ingestion.espn_match_stats_scraper import ESPNMatchStatsScraper, EXTRA_LEAGUES_ESPN
            scraper = ESPNMatchStatsScraper()
            for lg in EXTRA_LEAGUES_ESPN.keys():
                scraper.enrich_csv(league_code=lg)
            logger.info("Enriquecimiento con ESPN FC completado.")
        except Exception as e:
            logger.error(f"Error al enriquecer con ESPN FC: {e}")

