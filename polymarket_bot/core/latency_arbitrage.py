import logging

logger = logging.getLogger('polymarket_bot.core.latency_arbitrage')

class LatencyArbitrageEngine:
    def __init__(self, threshold_edge=0.03):
        """
        threshold_edge: El margen de diferencia (EV) requerido entre Pinnacle y Polymarket 
                        para considerar disparar una orden. Ejemplo: 0.03 = 3%.
        """
        self.threshold_edge = threshold_edge

    def calculate_implied_probability(self, pinnacle_odds):
        """
        Remueve el margen (vig) de las cuotas de Pinnacle para obtener la probabilidad 'True' o 'Sharp'.
        """
        if not pinnacle_odds or len(pinnacle_odds) != 3:
            return None
            
        implied_probs = [1 / float(outcome['price']) for outcome in pinnacle_odds]
        vig = sum(implied_probs)
        
        true_probs = {
            outcome['name']: (1 / float(outcome['price'])) / vig 
            for outcome in pinnacle_odds
        }
        return true_probs

    def check_arbitrage_opportunity(self, match_name, pinnacle_odds, polymarket_price, target_outcome):
        """
        Compara la probabilidad True de Pinnacle con el precio de Polymarket.
        Si la diferencia supera el threshold_edge, alerta sobre un arbitraje.
        """
        true_probs = self.calculate_implied_probability(pinnacle_odds)
        
        if not true_probs or target_outcome not in true_probs:
            logger.warning(f"No se pudo calcular probabilidad true para {target_outcome}")
            return False, 0.0

        pinnacle_prob = true_probs[target_outcome]
        
        # El precio de Polymarket (ej. 0.45) se traduce a 45% de probabilidad implicita.
        # Si Pinnacle cree que la probabilidad es 50% (0.50), hay un Edge de 5%.
        edge = pinnacle_prob - polymarket_price
        
        if edge > self.threshold_edge:
            logger.info(f"🚨 OPORTUNIDAD ARBITRAJE [{match_name} - {target_outcome}] 🚨")
            logger.info(f"Pinnacle True Prob: {pinnacle_prob:.4f} | Polymarket Price: {polymarket_price:.4f} | Edge: {edge:.4f}")
            return True, edge
            
        elif edge < -self.threshold_edge:
            # Polymarket está sobre-valorando el evento en comparación con Pinnacle
            logger.info(f"📉 OPORTUNIDAD VENTA/LAY [{match_name} - {target_outcome}] 📉")
            logger.info(f"Pinnacle True Prob: {pinnacle_prob:.4f} | Polymarket Price: {polymarket_price:.4f} | Edge: {edge:.4f}")
            # Retorna verdadero para "Lay" (vender Yes o comprar No)
            return True, edge

        logger.debug(f"[{match_name}] Mercado eficiente. Edge actual: {edge:.4f} (Threshold: {self.threshold_edge})")
        return False, edge
