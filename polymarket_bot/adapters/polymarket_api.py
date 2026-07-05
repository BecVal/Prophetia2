import requests
import logging

logger = logging.getLogger('polymarket_bot.adapters.polymarket')

class PolymarketClient:
    """
    Cliente REST simple para el Central Limit Order Book (CLOB) de Polymarket.
    Para WebSockets reales (streaming), se recomienda usar la librería oficial 'py_clob_client'.
    """
    def __init__(self, host="https://clob.polymarket.com"):
        self.host = host

    def get_order_book(self, token_id):
        """
        Obtiene el Order Book completo (L2) para un token específico.
        """
        try:
            # Ejemplo de token_id: "0x... (hash del token de Arsenal Ganador)"
            url = f"{self.host}/book?token_id={token_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo order book de Polymarket: {e}")
            return self._get_mock_order_book()

    def get_market_price(self, token_id):
        """Obtiene el mid-price actual (promedio entre mejor bid y mejor ask)"""
        book = self.get_order_book(token_id)
        if not book or not book.get('bids') or not book.get('asks'):
            return None
        
        best_bid = float(book['bids'][0]['price'])
        best_ask = float(book['asks'][0]['price'])
        return (best_bid + best_ask) / 2.0

    def _get_mock_order_book(self):
        """Devuelve un order book simulado."""
        return {
            "bids": [
                {"price": "0.45", "size": "1500"},
                {"price": "0.44", "size": "3000"},
                {"price": "0.40", "size": "10000"}
            ],
            "asks": [
                {"price": "0.47", "size": "500"},
                {"price": "0.48", "size": "2000"},
                {"price": "0.50", "size": "5000"}
            ]
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = PolymarketClient()
    # Token ID dummy
    print("Order book simulado:")
    print(client.get_order_book("dummy_token"))
    print(f"Mid Price: {client.get_market_price('dummy_token')}")
