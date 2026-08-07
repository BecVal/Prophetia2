import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import team mapping
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.team_mapping import normalize_team_name

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"

COMP_TO_ODDS_API_SPORT = {
    'E0': 'soccer_epl', 'Premier League': 'soccer_epl',
    'SP1': 'soccer_spain_la_liga', 'La Liga': 'soccer_spain_la_liga',
    'D1': 'soccer_germany_bundesliga', '1. Bundesliga': 'soccer_germany_bundesliga',
    'I1': 'soccer_italy_serie_a', 'Serie A': 'soccer_italy_serie_a',
    'F1': 'soccer_france_ligue_one', 'Ligue 1': 'soccer_france_ligue_one',
    'E1': 'soccer_efl_champ', 'Championship': 'soccer_efl_champ',
    'E2': 'soccer_england_league1', 'League One': 'soccer_england_league1',
    'SP2': 'soccer_spain_segunda_division', 'La Liga 2': 'soccer_spain_segunda_division',
    'D2': 'soccer_germany_bundesliga2', '2. Bundesliga': 'soccer_germany_bundesliga2',
    'F2': 'soccer_france_ligue_two', 'Ligue 2': 'soccer_france_ligue_two',
    'I2': 'soccer_italy_serie_b', 'Serie B': 'soccer_italy_serie_b',
    'B1': 'soccer_belgium_first_div', 'Jupiler Pro League': 'soccer_belgium_first_div',
    'N1': 'soccer_netherlands_eredivisie', 'Eredivisie': 'soccer_netherlands_eredivisie',
    'P1': 'soccer_portugal_primeira_liga', 'Primeira Liga': 'soccer_portugal_primeira_liga',
    'T1': 'soccer_turkey_super_league', 'Süper Lig': 'soccer_turkey_super_league',
    'J1': 'soccer_japan_j_league', 'J-League 1': 'soccer_japan_j_league', 'JPN': 'soccer_japan_j_league',
    'G1': 'soccer_greece_super_league', 'Super League': 'soccer_greece_super_league',
    'SC0': 'soccer_scotland_premiership', 'Scottish Premiership': 'soccer_scotland_premiership',
    'SWE': 'soccer_sweden_allsvenskan', 'Allsvenskan': 'soccer_sweden_allsvenskan',
    'NOR': 'soccer_norway_eliteserien', 'Eliteserien': 'soccer_norway_eliteserien',
    'DNK': 'soccer_denmark_superliga', 'Superligaen': 'soccer_denmark_superliga',
    'SWZ': 'soccer_switzerland_superleague', 'Swiss Super League': 'soccer_switzerland_superleague',
    'AUT': 'soccer_austria_bundesliga', 'Austrian Bundesliga': 'soccer_austria_bundesliga',
    'MLS': 'soccer_usa_mls', 'Major League Soccer': 'soccer_usa_mls',
    'MEX': 'soccer_mexico_ligamx', 'MEX1': 'soccer_mexico_ligamx', 'Liga MX': 'soccer_mexico_ligamx',
    'CL': 'soccer_uefa_champs_league', 'Champions League': 'soccer_uefa_champs_league',
    'EL': 'soccer_uefa_europa_league', 'Europa League': 'soccer_uefa_europa_league'
}

COMP_TO_ESPN_SPORT = {
    'E0': 'eng.1', 'Premier League': 'eng.1',
    'E1': 'eng.2', 'Championship': 'eng.2',
    'E2': 'eng.3', 'League One': 'eng.3',
    'SP1': 'esp.1', 'La Liga': 'esp.1',
    'SP2': 'esp.2', 'La Liga 2': 'esp.2',
    'D1': 'ger.1', '1. Bundesliga': 'ger.1',
    'D2': 'ger.2', '2. Bundesliga': 'ger.2',
    'I1': 'ita.1', 'Serie A': 'ita.1',
    'I2': 'ita.2', 'Serie B': 'ita.2',
    'F1': 'fra.1', 'Ligue 1': 'fra.1',
    'F2': 'fra.2', 'Ligue 2': 'fra.2',
    'B1': 'bel.1', 'Jupiler Pro League': 'bel.1',
    'N1': 'ned.1', 'Eredivisie': 'ned.1',
    'P1': 'por.1', 'Primeira Liga': 'por.1',
    'T1': 'tur.1', 'Süper Lig': 'tur.1',
    'J1': 'jpn.1', 'J-League 1': 'jpn.1', 'JPN': 'jpn.1',
    'G1': 'gre.1', 'Super League': 'gre.1',
    'SC0': 'sco.1', 'Scottish Premiership': 'sco.1',
    'SWE': 'swe.1', 'Allsvenskan': 'swe.1',
    'NOR': 'nor.1', 'Eliteserien': 'nor.1',
    'DNK': 'dnk.1', 'Superligaen': 'dnk.1',
    'SWZ': 'sui.1', 'Swiss Super League': 'sui.1',
    'AUT': 'aut.1', 'Austrian Bundesliga': 'aut.1',
    'MLS': 'usa.1', 'Major League Soccer': 'usa.1',
    'MEX': 'mex.1', 'MEX1': 'mex.1', 'Liga MX': 'mex.1',
    'CL': 'uefa.champions', 'Champions League': 'uefa.champions',
    'EL': 'uefa.europa', 'Europa League': 'uefa.europa'
}

COMP_MAPPING_REVERSE = {
    'Premier League': 'E0', 'La Liga': 'SP1', '1. Bundesliga': 'D1', 'Serie A': 'I1', 'Ligue 1': 'F1',
    'Championship': 'E1', 'La Liga 2': 'SP2', '2. Bundesliga': 'D2', 'Ligue 2': 'F2', 'Serie B': 'I2',
    'Jupiler Pro League': 'B1', 'Eredivisie': 'N1', 'Primeira Liga': 'P1', 'Süper Lig': 'T1',
    'J1': 'JPN', 'J-League 1': 'JPN', 'Super League': 'G1', 'Scottish Premiership': 'SC0',
    'Allsvenskan': 'SWE', 'Eliteserien': 'NOR', 'Superligaen': 'DNK', 'Swiss Super League': 'SWZ',
    'Austrian Bundesliga': 'AUT', 'Major League Soccer': 'MLS', 'Liga MX': 'MEX', 'Champions League': 'CL', 'Europa League': 'EL'
}

def american_to_decimal(ml):
    if ml is None: return None
    try:
        val = float(ml)
        if val > 0: return round((val / 100.0) + 1.0, 2)
        elif val < 0: return round((100.0 / abs(val)) + 1.0, 2)
    except:
        pass
    return None

class LiveOddsFeedFetcher:
    """
    Cliente para la ingesta OBLIGATORIA de partidos y cuotas en tiempo real directamente desde:
    1. The-Odds-API (Pinnacle Sportsbook Feed si se cuenta con API key)
    2. ESPN Live Scoreboard & Fixture API (Oficial, en tiempo real para todas las ligas)
    3. Football-Data.co.uk Fixtures Feed (BOM handling arreglado)
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('THE_ODDS_API_KEY') or os.environ.get('ODDS_API_KEY')

    def fetch_from_the_odds_api(self, comp):
        """Consulta cuotas en vivo de Pinnacle vía The-Odds-API."""
        if not self.api_key:
            return None
            
        sport_key = COMP_TO_ODDS_API_SPORT.get(comp, 'soccer_epl')
        try:
            url = f"{THE_ODDS_API_URL}/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'eu',
                'markets': 'h2h,totals',
                'bookmakers': 'pinnacle'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            matches = []
            for item in data:
                home = normalize_team_name(item.get('home_team', ''))
                away = normalize_team_name(item.get('away_team', ''))
                bms = item.get('bookmakers', [])
                odds_1, odds_X, odds_2 = None, None, None
                ou25 = 1.95
                for bm in bms:
                    if bm.get('key') == 'pinnacle':
                        for mkt in bm.get('markets', []):
                            if mkt.get('key') == 'h2h':
                                for out in mkt.get('outcomes', []):
                                    if out.get('name') == item.get('home_team'): odds_1 = float(out.get('price'))
                                    elif out.get('name') == item.get('away_team'): odds_2 = float(out.get('price'))
                                    elif out.get('name') == 'Draw': odds_X = float(out.get('price'))
                            elif mkt.get('key') == 'totals':
                                for out in mkt.get('outcomes', []):
                                    if out.get('name') == 'Over' and out.get('point') == 2.5:
                                        ou25 = float(out.get('price'))
                if odds_1 and odds_X and odds_2:
                    matches.append({
                        'home_team': home, 'away_team': away, 'competition': comp,
                        'odds_1': odds_1, 'odds_X': odds_X, 'odds_2': odds_2,
                        'open_odds_win': odds_1, 'open_odds_draw': odds_X, 'open_odds_loss': odds_2,
                        'extra_odds': {'ou25_over': ou25, 'btts_yes': 1.85},
                        'date': item.get('commence_time', ''),
                        'source': 'The-Odds-API (Pinnacle)'
                    })
            return matches
        except Exception:
            return None

    def fetch_from_espn_api(self, comp):
        """Consulta el calendario oficial completo de partidos próximos y cuotas de ESPN API."""
        sport_code = COMP_TO_ESPN_SPORT.get(comp)
        if not sport_code:
            sport_code = COMP_TO_ESPN_SPORT.get(COMP_MAPPING_REVERSE.get(comp, ''), 'esp.1')

        today = datetime.now()
        start_str = (today - timedelta(days=1)).strftime('%Y%m%d')
        end_str = (today + timedelta(days=35)).strftime('%Y%m%d')

        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{sport_code}/scoreboard?dates={start_str}-{end_str}&limit=100"
        matches = self._parse_espn_url(url, comp)
        
        if not matches:
            url_fallback = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{sport_code}/scoreboard?limit=100"
            matches = self._parse_espn_url(url_fallback, comp)

        return matches

    def _parse_espn_url(self, url, comp):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            events = data.get('events', [])
            matches = []

            for event in events:
                try:
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue
                    comp_info = competitions[0]
                    competitors = comp_info.get('competitors', [])
                    
                    home_raw, away_raw = '', ''
                    for c in competitors:
                        if c.get('homeAway') == 'home':
                            home_raw = c.get('team', {}).get('displayName', '')
                        elif c.get('homeAway') == 'away':
                            away_raw = c.get('team', {}).get('displayName', '')

                    if not home_raw or not away_raw:
                        continue

                    home = normalize_team_name(home_raw)
                    away = normalize_team_name(away_raw)

                    odds_1, odds_X, odds_2 = 2.10, 3.30, 3.40
                    open_1, open_X, open_2 = 2.10, 3.30, 3.40
                    ou25 = 1.95

                    has_live_odds = False
                    odds_list = comp_info.get('odds', [])
                    if odds_list and isinstance(odds_list, list) and len(odds_list) > 0:
                        o = odds_list[0]
                        if isinstance(o, dict):
                            ml = o.get('moneyline') or {}
                            if isinstance(ml, dict):
                                h_dict = ml.get('home') or {}
                                d_dict = ml.get('draw') or {}
                                a_dict = ml.get('away') or {}
                                
                                h_close = h_dict.get('close', {}).get('odds') if isinstance(h_dict, dict) and isinstance(h_dict.get('close'), dict) else None
                                h_open = h_dict.get('open', {}).get('odds') if isinstance(h_dict, dict) and isinstance(h_dict.get('open'), dict) else None

                                d_close = d_dict.get('close', {}).get('odds') if isinstance(d_dict, dict) and isinstance(d_dict.get('close'), dict) else None
                                d_open = d_dict.get('open', {}).get('odds') if isinstance(d_dict, dict) and isinstance(d_dict.get('open'), dict) else None

                                a_close = a_dict.get('close', {}).get('odds') if isinstance(a_dict, dict) and isinstance(a_dict.get('close'), dict) else None
                                a_open = a_dict.get('open', {}).get('odds') if isinstance(a_dict, dict) and isinstance(a_dict.get('open'), dict) else None

                                d1_close = american_to_decimal(h_close)
                                d1_open = american_to_decimal(h_open)

                                dX_close = american_to_decimal(d_close)
                                dX_open = american_to_decimal(d_open)

                                d2_close = american_to_decimal(a_close)
                                d2_open = american_to_decimal(a_open)

                                if d1_close and dX_close and d2_close:
                                    odds_1, odds_X, odds_2 = d1_close, dX_close, d2_close
                                    open_1 = d1_open if d1_open else odds_1
                                    open_X = dX_open if dX_open else odds_X
                                    open_2 = d2_open if d2_open else odds_2
                                    has_live_odds = True
                                elif d1_open and dX_open and d2_open:
                                    odds_1, odds_X, odds_2 = d1_open, dX_open, d2_open
                                    open_1, open_X, open_2 = d1_open, dX_open, d2_open
                                    has_live_odds = True

                            tot_dict = o.get('total') or {}
                            if isinstance(tot_dict, dict):
                                ov_dict = tot_dict.get('over') or {}
                                if isinstance(ov_dict, dict):
                                    cl_dict = ov_dict.get('close') or {}
                                    tot = cl_dict.get('line') if isinstance(cl_dict, dict) else None
                                    if tot:
                                        try:
                                            ou25 = float(str(tot).replace('o', ''))
                                        except Exception:
                                            pass

                    date_str = event.get('date', '')
                    matches.append({
                        'home_team': home,
                        'away_team': away,
                        'competition': comp,
                        'odds_1': odds_1,
                        'odds_X': odds_X,
                        'odds_2': odds_2,
                        'open_odds_win': open_1,
                        'open_odds_draw': open_X,
                        'open_odds_loss': open_2,
                        'extra_odds': {'ou25_over': ou25, 'btts_yes': 1.85},
                        'date': date_str,
                        'has_live_odds': has_live_odds,
                        'source': 'ESPN Live Schedule & Odds' if has_live_odds else 'ESPN Live Schedule (Cuotas Aún No Publicadas)'
                    })
                except Exception:
                    continue

            return matches
        except Exception:
            return []

    def fetch_from_football_data_fixtures(self, comp):
        """Descarga e interpreta partidos próximos reales de fixtures.csv (utf-8-sig)."""
        try:
            div_code = COMP_MAPPING_REVERSE.get(comp, comp)
            matches = []
            
            try:
                df_fix = pd.read_csv(FIXTURES_URL, encoding='utf-8-sig', on_bad_lines='skip')
                if 'Div' in df_fix.columns and div_code in df_fix['Div'].values:
                    df_league = df_fix[df_fix['Div'] == div_code].copy()
                    matches = self._parse_football_data_df(df_league, comp)
            except Exception:
                pass

            return matches
        except Exception:
            return []

    def _parse_football_data_df(self, df_league, comp):
        matches = []
        for _, row in df_league.iterrows():
            h_raw = row.get('HomeTeam', row.get('Home', ''))
            a_raw = row.get('AwayTeam', row.get('Away', ''))
            if not h_raw or not a_raw or pd.isna(h_raw) or pd.isna(a_raw):
                continue

            home = normalize_team_name(str(h_raw))
            away = normalize_team_name(str(a_raw))

            p_win = float(row.get('PSCH', row.get('PSH', row.get('B365H', 0.0))))
            p_draw = float(row.get('PSCD', row.get('PSD', row.get('B365D', 0.0))))
            p_loss = float(row.get('PSCA', row.get('PSA', row.get('B365A', 0.0))))

            open_win = float(row.get('PSH', row.get('B365H', p_win)))
            open_draw = float(row.get('PSD', row.get('B365D', p_draw)))
            open_loss = float(row.get('PSA', row.get('B365A', p_loss)))

            ou25 = float(row.get('B365>2.5', row.get('Avg>2.5', row.get('Max>2.5', 1.95))))
            btts = float(row.get('B365BBTS', row.get('AvgBBTS', 1.85)))

            if p_win > 1.01 and p_draw > 1.01 and p_loss > 1.01:
                date_str = str(row.get('Date', ''))
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'competition': comp,
                    'odds_1': p_win,
                    'odds_X': p_draw,
                    'odds_2': p_loss,
                    'open_odds_win': open_win,
                    'open_odds_draw': open_draw,
                    'open_odds_loss': open_loss,
                    'extra_odds': {'ou25_over': ou25 if ou25 > 1.0 else 1.95, 'btts_yes': btts if btts > 1.0 else 1.85},
                    'date': date_str,
                    'source': 'Football-Data Live Feed'
                })
        return matches

    def get_live_league_fixtures(self, comp):
        """Descarga OBLIGATORIA de los partidos próximos y cuotas en vivo de la liga."""
        # 1. Intentar The-Odds-API (Pinnacle) si hay API key
        if self.api_key:
            results = self.fetch_from_the_odds_api(comp)
            if results:
                return results

        # 2. Intentar ESPN Live Schedule API
        results = self.fetch_from_espn_api(comp)
        if results:
            return results

        # 3. Intentar Football-Data fixtures.csv
        results = self.fetch_from_football_data_fixtures(comp)
        return results

if __name__ == '__main__':
    fetcher = LiveOddsFeedFetcher()
    matches = fetcher.get_live_league_fixtures('La Liga')
    print(f"Descargados {len(matches)} partidos en vivo para La Liga.")
    if matches:
        print("Ejemplo de partido descargado:", matches[0])
