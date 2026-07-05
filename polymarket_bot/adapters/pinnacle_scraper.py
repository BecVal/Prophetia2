import requests
import time
import logging

logger = logging.getLogger('polymarket_bot.adapters.pinnacle')

class FreeOddsAPIClient:
    """
    Cliente para obtener cuotas en vivo. 
    Dado que Pinnacle no tiene API gratuita pública, utilizaremos 
    'The Odds API' (https://the-odds-api.com/) que tiene un tier gratuito.
    Alternativamente, aquí se podría integrar un scraper de Selenium para OddsPortal.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or "TU_API_KEY_GRATUITA_AQUI"
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def get_live_odds(self, sport="soccer_epl", region="eu", markets="h2h"):
        """
        Obtiene las cuotas en vivo de bookmakers europeos (incluyendo Pinnacle si está disponible en tu tier).
        """
        if self.api_key == "TU_API_KEY_GRATUITA_AQUI":
            logger.warning("Falta la API KEY de The Odds API. Retornando datos simulados para pruebas locales.")
            return self._get_mock_live_odds()

        try:
            url = f"{self.base_url}/{sport}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': region,
                'markets': markets,
                'bookmakers': 'pinnacle', # Filtrar solo Pinnacle si es posible
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo cuotas en vivo: {e}")
            return None

    def _get_mock_live_odds(self):
        """Devuelve datos simulados para probar la latencia localmente."""
        return [
            {
                "id": "mock_match_1",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "last_update": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.10}, # Implied prob ~ 47.6%
                                    {"name": "Chelsea", "price": 3.40}, # Implied prob ~ 29.4%
                                    {"name": "Draw", "price": 3.50}     # Implied prob ~ 28.5%
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = FreeOddsAPIClient()
    print(client.get_live_odds())
