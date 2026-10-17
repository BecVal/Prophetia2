import os
import sys
import argparse
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración de imports relativos
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.team_mapping import normalize_team_name
from core.logger_config import get_logger

logger = get_logger(__name__, 'espn_match_stats_scraper')

# Mapeo de códigos de Football-Data a slugs de la API de ESPN FC
EXTRA_LEAGUES_ESPN = {
    'MEX': 'mex.1',      # Liga MX
    'USA': 'usa.1',      # Major League Soccer (MLS)
    'JPN': 'jpn.1',      # J1 League
    'SWE': 'swe.1',      # Allsvenskan
    'NOR': 'nor.1',      # Eliteserien
    'DNK': 'den.1',      # Danish Superliga
    'SWZ': 'sui.1',      # Swiss Super League
    'AUT': 'aut.1',      # Austrian Bundesliga
}

RAW_FOOTBALL_DATA_DIR = os.path.join(root_dir, 'data', 'raw', 'football_data')
CACHE_BASE_DIR = os.path.join(root_dir, 'data', 'raw', 'espn_cache')

HEADERS = {
    'User-Agent': 'Mozilla/5.0'
}


def american_to_decimal(ml):
    """Convierte momio americano a cuota decimal."""
    if ml is None:
        return None
    try:
        val = float(ml)
        if val > 0:
            return round((val / 100.0) + 1.0, 2)
        elif val < 0:
            return round((100.0 / abs(val)) + 1.0, 2)
    except Exception:
        pass
    return None


class ESPNMatchStatsScraper:
    """
    Scraper y extractor de estadísticas avanzadas de partidos desde ESPN FC API.
    Enriquece los CSVs de ligas complementarias (Liga MX, MLS, etc.) con:
    - Córneres: HC, AC
    - Tiros: HS, AS, HST, AST
    - Faltas y Tarjetas: HF, AF, HY, AY, HR, AR
    - Árbitros: Referee
    - Cuotas: Over/Under 2.5 y Asian Handicap
    """
    def __init__(self, max_workers=15, use_cache=True):
        self.max_workers = max_workers
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_scoreboard_events(self, espn_sport, start_year, end_year):
        """
        Descarga el listado completo de partidos/eventos para un rango de años desde ESPN Scoreboard API.
        """
        events = []
        seen_ids = set()
        logger.info(f"Consultando calendario ESPN para '{espn_sport}' entre {start_year} y {end_year}...")

        for year in range(start_year, end_year + 1):
            dates_param = f"{year}0101-{year}1231"
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_sport}/scoreboard?dates={dates_param}&limit=600"
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    year_events = data.get('events', [])
                    new_count = 0
                    for ev in year_events:
                        ev_id = str(ev.get('id', ''))
                        if ev_id and ev_id not in seen_ids:
                            seen_ids.add(ev_id)
                            events.append(ev)
                            new_count += 1
                    logger.info(f"Año {year}: {new_count} eventos encontrados (total acumulado: {len(events)})")
                else:
                    logger.warning(f"Error {resp.status_code} al consultar scoreboard para {espn_sport} en {year}")
            except Exception as e:
                logger.error(f"Excepción al consultar scoreboard para {espn_sport} en {year}: {e}")

        return events

    def get_match_summary(self, espn_sport, event_id, league_code, force_refresh=False):
        """
        Obtiene el resumen (boxscore + metadata) de un partido, usando caché en disco si existe.
        Omite la descarga si el partido ya está guardado con estadísticas o en estado finalizado.
        """
        league_cache_dir = os.path.join(CACHE_BASE_DIR, league_code)
        os.makedirs(league_cache_dir, exist_ok=True)
        cache_file = os.path.join(league_cache_dir, f"{event_id}.json")

        if self.use_cache and not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Verificar si ya tiene estadísticas completas o si el partido finalizó
                box_teams = cached_data.get('boxscore', {}).get('teams', [])
                has_stats = bool(box_teams and len(box_teams[0].get('statistics', [])) > 0)
                state = cached_data.get('header', {}).get('competitions', [{}])[0].get('status', {}).get('type', {}).get('state', '')
                
                # Si el partido ya finalizó o tiene estadísticas, reutilizar caché sin peticiones de red
                if has_stats or state == 'post':
                    return cached_data
                # Si era un partido futuro ('pre') o pendiente, se intentará consultar de nuevo para capturar resultados recientes
            except Exception:
                pass

        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_sport}/summary?event={event_id}"
        try:
            resp = self.session.get(url, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                if self.use_cache:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                    except Exception:
                        pass
                return data
            else:
                return None
        except Exception:
            return None

    def parse_match_summary(self, event, summary_data):
        """
        Extrae y estandariza todas las métricas del partido a formato Football-Data.
        """
        if not summary_data:
            return None

        try:
            # Metadata básica del evento
            event_id = event.get('id', '')
            event_date_raw = event.get('date', '')
            try:
                match_date = pd.to_datetime(event_date_raw).strftime('%Y-%m-%d')
            except Exception:
                match_date = None

            # Equipos desde header / boxscore
            boxscore = summary_data.get('boxscore', {})
            teams_box = boxscore.get('teams', [])
            
            home_team_raw, away_team_raw = '', ''
            home_stats, away_stats = {}, {}

            if teams_box and len(teams_box) >= 2:
                for t in teams_box:
                    team_info = t.get('team', {})
                    display_name = team_info.get('displayName', '')
                    stats_list = t.get('statistics', [])
                    stats_dict = {s.get('name'): s.get('displayValue') for s in stats_list if isinstance(s, dict)}

                    home_away = t.get('homeAway')
                    if not home_away:
                        for comp in event.get('competitions', [{}])[0].get('competitors', []):
                            if str(comp.get('id')) == str(team_info.get('id')):
                                home_away = comp.get('homeAway')

                    if home_away == 'home':
                        home_team_raw = display_name
                        home_stats = stats_dict
                    elif home_away == 'away':
                        away_team_raw = display_name
                        away_stats = stats_dict
            
            # Fallback a competitions competitors si no se pudo determinar por boxscore
            if not home_team_raw or not away_team_raw:
                competitions = event.get('competitions', [])
                if competitions:
                    for comp in competitions[0].get('competitors', []):
                        if comp.get('homeAway') == 'home':
                            home_team_raw = comp.get('team', {}).get('displayName', '')
                        elif comp.get('homeAway') == 'away':
                            away_team_raw = comp.get('team', {}).get('displayName', '')

            if not home_team_raw or not away_team_raw:
                return None

            home_team = normalize_team_name(home_team_raw)
            away_team = normalize_team_name(away_team_raw)

            def parse_num(stats_dict, key, default=np.nan):
                val = stats_dict.get(key)
                if val is None:
                    return default
                try:
                    return float(val)
                except Exception:
                    return default

            # Córneres
            hc = parse_num(home_stats, 'wonCorners')
            ac = parse_num(away_stats, 'wonCorners')

            # Tiros totales
            hs = parse_num(home_stats, 'totalShots')
            as_ = parse_num(away_stats, 'totalShots')

            # Tiros a puerta
            hst = parse_num(home_stats, 'shotsOnTarget')
            ast = parse_num(away_stats, 'shotsOnTarget')

            # Faltas
            hf = parse_num(home_stats, 'foulsCommitted')
            af = parse_num(away_stats, 'foulsCommitted')

            # Tarjetas amarillas
            hy = parse_num(home_stats, 'yellowCards')
            ay = parse_num(away_stats, 'yellowCards')

            # Tarjetas rojas
            hr = parse_num(home_stats, 'redCards')
            ar = parse_num(away_stats, 'redCards')

            # Árbitro (Referee)
            referee = np.nan
            game_info = summary_data.get('gameInfo', {})
            officials = game_info.get('officials', [])
            if officials and isinstance(officials, list):
                ref_name = officials[0].get('displayName', '').strip()
                if ref_name:
                    referee = ref_name

            # Cuotas Over/Under y Asian Handicap desde ESPN (si están disponibles)
            ou25_over, ou25_under = np.nan, np.nan
            ah_line, ah_home, ah_away = np.nan, np.nan, np.nan

            odds_list = summary_data.get('odds', [])
            if odds_list and isinstance(odds_list, list) and len(odds_list) > 0:
                odd_item = odds_list[0]
                # Over/Under
                ou_val = odd_item.get('overUnder')
                if ou_val == 2.5 or (ou_val is not None and float(ou_val) == 2.5):
                    over_odds = odd_item.get('overOdds')
                    under_odds = odd_item.get('underOdds')
                    if over_odds is not None:
                        ou25_over = american_to_decimal(over_odds)
                    if under_odds is not None:
                        ou25_under = american_to_decimal(under_odds)
                
                # Spread / Handicap
                spread_val = odd_item.get('spread')
                if spread_val is not None:
                    try:
                        ah_line = float(spread_val)
                    except Exception:
                        pass
                
                home_odds_obj = odd_item.get('homeTeamOdds', {})
                away_odds_obj = odd_item.get('awayTeamOdds', {})
                if home_odds_obj and isinstance(home_odds_obj, dict):
                    ah_home = american_to_decimal(home_odds_obj.get('spreadOdds'))
                if away_odds_obj and isinstance(away_odds_obj, dict):
                    ah_away = american_to_decimal(away_odds_obj.get('spreadOdds'))

            # Validación de estadísticas no nulas
            has_stats = not (
                (hc == 0 or np.isnan(hc)) and
                (ac == 0 or np.isnan(ac)) and
                (hs == 0 or np.isnan(hs)) and
                (as_ == 0 or np.isnan(as_)) and
                (hf == 0 or np.isnan(hf)) and
                (af == 0 or np.isnan(af))
            )

            return {
                'espn_event_id': event_id,
                'espn_date': match_date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'HC': hc if has_stats else np.nan,
                'AC': ac if has_stats else np.nan,
                'HS': hs if has_stats else np.nan,
                'AS': as_ if has_stats else np.nan,
                'HST': hst if has_stats else np.nan,
                'AST': ast if has_stats else np.nan,
                'HF': hf if has_stats else np.nan,
                'AF': af if has_stats else np.nan,
                'HY': hy if has_stats else np.nan,
                'AY': ay if has_stats else np.nan,
                'HR': hr if has_stats else np.nan,
                'AR': ar if has_stats else np.nan,
                'Referee': referee,
                'Avg>2.5': ou25_over,
                'Avg<2.5': ou25_under,
                'AHh': ah_line,
                'AvgAHH': ah_home,
                'AvgAHA': ah_away,
                'has_boxscore_stats': has_stats
            }
        except Exception as e:
            logger.debug(f"Error parseando summary: {e}")
            return None

    def scrape_league_stats(self, league_code, start_year=2014, end_year=2026, force_refresh=False):
        """
        Descarga y parsea todos los partidos de una liga en el rango de años indicado.
        """
        espn_sport = EXTRA_LEAGUES_ESPN.get(league_code)
        if not espn_sport:
            logger.error(f"Código de liga desconocido: {league_code}. Ligas soportadas: {list(EXTRA_LEAGUES_ESPN.keys())}")
            return pd.DataFrame()

        events = self.fetch_scoreboard_events(espn_sport, start_year, end_year)
        if not events:
            logger.warning(f"No se obtuvieron eventos para la liga {league_code}")
            return pd.DataFrame()

        logger.info(f"Descargando resúmenes de {len(events)} partidos concurrentemente con {self.max_workers} hilos...")
        parsed_matches = []

        def process_event(ev):
            ev_id = ev.get('id')
            if not ev_id:
                return None
            summary = self.get_match_summary(espn_sport, ev_id, league_code, force_refresh=force_refresh)
            if summary:
                return self.parse_match_summary(ev, summary)
            return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ev = {executor.submit(process_event, ev): ev for ev in events}
            for future in as_completed(future_to_ev):
                res = future.result()
                if res is not None:
                    parsed_matches.append(res)

        if not parsed_matches:
            logger.warning(f"No se pudieron parsear resúmenes para la liga {league_code}")
            return pd.DataFrame()

        df_espn = pd.DataFrame(parsed_matches)
        logger.info(f"Total de partidos parseados de ESPN para {league_code}: {len(df_espn)} (con estadísticas completas: {df_espn['has_boxscore_stats'].sum()})")
        return df_espn

    def enrich_csv(self, league_code, start_year=2014, end_year=2026, force_refresh=False, dry_run=False):
        """
        Funde las estadísticas de ESPN con el archivo CSV de Football-Data correspondiente.
        """
        csv_file = os.path.join(RAW_FOOTBALL_DATA_DIR, f"{league_code}.csv")
        if not os.path.exists(csv_file):
            logger.error(f"No se encontró el archivo CSV en: {csv_file}")
            return False

        try:
            df_fd = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            df_fd = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')

        logger.info(f"CSV de Football-Data cargado: {len(df_fd)} filas, {len(df_fd.columns)} columnas.")

        # Estandarizar nombres de columnas si vienen como Home / Away
        home_col = 'HomeTeam' if 'HomeTeam' in df_fd.columns else 'Home'
        away_col = 'AwayTeam' if 'AwayTeam' in df_fd.columns else 'Away'
        date_col = 'Date'

        if home_col not in df_fd.columns or away_col not in df_fd.columns or date_col not in df_fd.columns:
            logger.error(f"Columnas esenciales no encontradas en {csv_file}")
            return False

        # Descargar datos de ESPN
        df_espn = self.scrape_league_stats(league_code, start_year, end_year, force_refresh=force_refresh)
        if df_espn.empty:
            logger.error(f"No se obtuvieron datos de ESPN para {league_code}")
            return False

        # Columnas a transferir / enriquecer
        enrich_cols = ['HC', 'AC', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR', 'Referee']
        optional_odds_cols = ['Avg>2.5', 'Avg<2.5', 'AHh', 'AvgAHH', 'AvgAHA']

        # Evitar colisiones de nombres si el CSV ya contenía estas columnas de ejecuciones previas
        for c in enrich_cols + optional_odds_cols:
            if c in df_fd.columns:
                df_fd.drop(columns=[c], inplace=True)

        # Preparar dataframes para merge difuso por fecha
        df_fd['_row_id'] = np.arange(len(df_fd))
        df_fd['Home_norm'] = df_fd[home_col].astype(str).apply(normalize_team_name)
        df_fd['Away_norm'] = df_fd[away_col].astype(str).apply(normalize_team_name)
        df_fd['Date_parsed'] = pd.to_datetime(df_fd[date_col], format='mixed', dayfirst=True)

        df_espn['espn_date_parsed'] = pd.to_datetime(df_espn['espn_date'])

        logger.info("Realizando merge de estadísticas ESPN con CSV Football-Data...")

        # Merge por equipos
        merged = pd.merge(
            df_fd,
            df_espn,
            left_on=['Home_norm', 'Away_norm'],
            right_on=['HomeTeam', 'AwayTeam'],
            how='left'
        )

        # Filtrar por diferencia de fecha <= 2 días
        merged['date_diff'] = abs((merged['Date_parsed'] - merged['espn_date_parsed']).dt.days)
        merged['is_valid_match'] = (merged['date_diff'] <= 2) & (merged['has_boxscore_stats'] == True)

        # Ordenar para priorizar coincidencia válida y menor diferencia de días
        merged = merged.sort_values(by=['_row_id', 'is_valid_match', 'date_diff'], ascending=[True, False, True])
        merged = merged.drop_duplicates(subset=['_row_id']).set_index('_row_id')

        # Si no fue coincidencia válida, limpiar estadísticas
        invalid_mask = ~merged['is_valid_match']
        for col in enrich_cols + optional_odds_cols:
            if col in merged.columns:
                merged.loc[invalid_mask, col] = np.nan

        # Transferir columnas enriquecidas a df_fd alineadas por _row_id (index)
        for col in enrich_cols:
            if col in merged.columns:
                df_fd[col] = merged[col]

        for col in optional_odds_cols:
            if col in merged.columns and merged[col].notna().any():
                df_fd[col] = merged[col]

        # Limpiar columnas auxiliares
        temp_cols = ['_row_id', 'Home_norm', 'Away_norm', 'Date_parsed']
        for tc in temp_cols:
            if tc in df_fd.columns:
                df_fd.drop(columns=[tc], inplace=True)

        # Reporte de cobertura
        total_rows = len(df_fd)
        corners_cov = (df_fd['HC'].notna().sum() / total_rows) * 100 if 'HC' in df_fd.columns else 0.0
        shots_cov = (df_fd['HS'].notna().sum() / total_rows) * 100 if 'HS' in df_fd.columns else 0.0
        cards_cov = (df_fd['HY'].notna().sum() / total_rows) * 100 if 'HY' in df_fd.columns else 0.0
        ref_cov = (df_fd['Referee'].notna().sum() / total_rows) * 100 if 'Referee' in df_fd.columns else 0.0

        logger.info(f"=== RESULTADOS DE ENRIQUECIMIENTO PARA {league_code} ===")
        logger.info(f"Total de partidos en CSV: {total_rows}")
        logger.info(f"Cobertura Córneres (HC/AC): {corners_cov:.1f}% ({df_fd['HC'].notna().sum()} partidos)")
        logger.info(f"Cobertura Tiros (HS/AS/HST/AST): {shots_cov:.1f}% ({df_fd['HS'].notna().sum()} partidos)")
        logger.info(f"Cobertura Faltas/Tarjetas (HF/AF/HY/AY/HR/AR): {cards_cov:.1f}% ({df_fd['HY'].notna().sum()} partidos)")
        logger.info(f"Cobertura Árbitros (Referee): {ref_cov:.1f}% ({df_fd['Referee'].notna().sum()} partidos)")
        logger.info("=========================================================")

        if not dry_run:
            df_fd.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"Archivo enriquecido guardado exitosamente en: {csv_file}")
        else:
            logger.info("[DRY RUN] No se realizaron modificaciones en el disco.")

        return True


def main():
    parser = argparse.ArgumentParser(description="Scraper de Estadísticas y Eventos ESPN para Ligas Complementarias (Liga MX, MLS, etc.)")
    parser.add_argument('--league', type=str, default='MEX', help="Código de liga a procesar (MEX, USA, JPN, SWE, NOR, DNK, SWZ, AUT o ALL). Por defecto: MEX")
    parser.add_argument('--start-year', type=int, default=2014, help="Año inicial a descargar (por defecto: 2014)")
    parser.add_argument('--end-year', type=int, default=2026, help="Año final a descargar (por defecto: 2026)")
    parser.add_argument('--max-workers', type=int, default=15, help="Número de hilos concurrentes para descarga (por defecto: 15)")
    parser.add_argument('--force-refresh', action='store_true', help="Forzar re-descarga omitiendo caché local")
    parser.add_argument('--dry-run', action='store_true', help="Ejecutar análisis sin guardar cambios en los archivos CSV")
    args = parser.parse_args()

    leagues_to_process = list(EXTRA_LEAGUES_ESPN.keys()) if args.league.upper() == 'ALL' else [args.league.upper()]

    scraper = ESPNMatchStatsScraper(max_workers=args.max_workers, use_cache=not args.force_refresh)

    for lg in leagues_to_process:
        logger.info(f"\n--- Iniciando procesamiento para {lg} ---")
        scraper.enrich_csv(
            league_code=lg,
            start_year=args.start_year,
            end_year=args.end_year,
            force_refresh=args.force_refresh,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
