import time
import logging
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polymarket_bot.adapters.pinnacle_scraper import FreeOddsAPIClient
from polymarket_bot.adapters.polymarket_api import PolymarketClient
from polymarket_bot.adapters.predictor import ProphetiaPredictor
from polymarket_bot.core.vwap_slicer import VWAPSlicer
from polymarket_bot.core.portfolio_manager import PortfolioManager
from polymarket_bot.core.web3_client import Web3ExecutionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('polymarket_bot.main')

class PolymarketBot:
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        
        self.pinnacle_client = FreeOddsAPIClient()
        self.poly_client = PolymarketClient()
        
        core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
        try:
            self.predictor = ProphetiaPredictor(core_dir)
            logger.info("Meta-Modelo Prophetia2 cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando Meta-Modelo: {e}")
            self.predictor = None
            
        self.vwap_slicer = VWAPSlicer(max_slippage_tolerance=0.015)
        self.portfolio = PortfolioManager(core_dir=core_dir, initial_bankroll=1000.0)
        self.execution = Web3ExecutionEngine()
        
    def run(self):
        logger.info("==================================================")
        logger.info(f"Iniciando Bot Quant HFT de Polymarket (Intervalo: {self.check_interval}s)")
        logger.info("==================================================")
        
        if not self.predictor:
            logger.error("No se puede iniciar el bot sin el Meta-Modelo.")
            return
            
        while True:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Error crítico en el loop principal: {e}")
                
            logger.info(f"Esperando {self.check_interval} segundos para el próximo escaneo...\n")
            time.sleep(self.check_interval)
            
    def tick(self):
        logger.info("Realizando escaneo de mercado...")
        
        live_odds_data = self.pinnacle_client.get_live_odds()
        if not live_odds_data:
            logger.warning("No se recibieron cuotas vivas.")
            return

        for match in live_odds_data:
            home_team = match['home_team']
            away_team = match['away_team']
            match_name = f"{home_team} vs {away_team}"
            
            bookmakers = match.get('bookmakers', [])
            if not bookmakers: continue
            
            pinnacle_odds = None
            for b in bookmakers:
                if b['key'] == 'pinnacle':
                    pinnacle_odds = b['markets'][0]['outcomes']
                    break
            
            if not pinnacle_odds: continue
            
            try:
                odds_1 = next(o['price'] for o in pinnacle_odds if o['name'] == home_team)
                odds_2 = next(o['price'] for o in pinnacle_odds if o['name'] == away_team)
                odds_X = next(o['price'] for o in pinnacle_odds if o['name'] == 'Draw')
            except StopIteration:
                logger.warning(f"Formato de cuotas inesperado para {match_name}")
                continue

            logger.info(f"[{match_name}] Consultando Meta-Modelo...")
            
            # 1. Obtener Probabilidades True del Meta-Modelo (que ya incluye el Dynamic Alpha Blending)
            true_probs = self.predictor.predict_match(
                home_team=home_team, 
                away_team=away_team, 
                odds_1=odds_1, odds_X=odds_X, odds_2=odds_2
            )
            
            if not true_probs:
                continue
                
            competition = true_probs['competition']
                
            # 2. Consultar Precios en Polymarket para las 3 opciones
            # Asumimos una convención de tokens ficticia para el mock
            tokens = {
                "home": f"token_{home_team}_win",
                "draw": f"token_{match_name}_draw",
                "away": f"token_{away_team}_win"
            }
            
            poly_prices = {
                "home": self.poly_client.get_market_price(tokens['home']) or 0.0,
                "draw": self.poly_client.get_market_price(tokens['draw']) or 0.0,
                "away": self.poly_client.get_market_price(tokens['away']) or 0.0
            }
            
            # Si no hay precios válidos en Polymarket, saltamos
            if not any(poly_prices.values()):
                logger.warning(f"[{match_name}] No se encontraron precios en Polymarket.")
                continue

            # 3. Evaluar Oportunidades y Gestión de Capital usando la lógica del simulador
            pred_clv = {
                "home": true_probs.get("pred_clv_win", 0),
                "draw": true_probs.get("pred_clv_draw", 0),
                "away": true_probs.get("pred_clv_loss", 0)
            }
            
            bet_plan = self.portfolio.evaluate_opportunities(
                match_id=match_name,
                competition=competition,
                probs=true_probs,
                poly_prices=poly_prices,
                pred_clv=pred_clv
            )
            
            if bet_plan:
                logger.info(f"🚨 APROBADO 🚨 Ejecutando {bet_plan['type']} en Polymarket")
                
                # 4. Smart Routing y Ejecución Web3
                for order in bet_plan['orders']:
                    outcome = order['outcome']
                    target_stake_usd = order['stake']
                    token_id = tokens[outcome]
                    
                    if target_stake_usd < 5:
                        logger.warning(f"Stake para {outcome} es muy bajo (${target_stake_usd:.2f}). Saltando.")
                        continue
                        
                    poly_price = poly_prices[outcome]
                    shares_to_buy = int(target_stake_usd / poly_price)
                    
                    # Llamar al VWAP slicer (necesita el order book del token)
                    order_book = self.poly_client.get_order_book(token_id)
                    execution_plan = self.vwap_slicer.slice_order(order_book, shares_to_buy)
                    
                    for slice_order in execution_plan:
                        if slice_order['type'] == 'TAKER':
                            self.execution.execute_market_order(token_id, target_stake_usd, "BUY")
                        else:
                            limit_price = slice_order['price']
                            slice_usd = slice_order['shares'] * limit_price
                            self.execution.place_limit_order(token_id, limit_price, slice_usd, "BUY")
                    
                    self.portfolio.register_position(token_id, target_stake_usd)
            else:
                logger.debug(f"[{match_name}] Sin Edge suficiente tras impuestos y EV Threshold.")

if __name__ == "__main__":
    bot = PolymarketBot(check_interval=60)
    bot.run()
