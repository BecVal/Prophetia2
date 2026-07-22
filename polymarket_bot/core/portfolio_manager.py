import os
import json
import logging
import numpy as np

logger = logging.getLogger('polymarket_bot.core.portfolio_manager')

class PortfolioManager:
    """
    Gestiona el tamaño de las apuestas y la evaluación de EV 
    incorporando la lógica avanzada de simulate_bankroll.py
    (Kelly optimizado por liga, Umbrales de EV, Fricciones de Mercado, y Dutching).
    """
    def __init__(self, core_dir, initial_bankroll=1000.0):
        self.bankroll = initial_bankroll
        self.active_positions = {}
        
        # Parámetros de Fricción (de simulate_bankroll.py)
        self.tax_retention_rate = 0.0075
        self.expected_clv_drop = 0.015
        self.max_stake_pct = 0.6 # 60% máximo por orden
        
        self.max_bet_liquidity = {
            'D1': 2000.0, 'SP1': 2000.0, 'I1': 2000.0, 'G1': 2000.0, 'F1': 2000.0,
            'D2': 2000.0, 'F2': 2000.0,
            'T1': 2000.0,
            'DEFAULT': 2000.0
        }
        
        # Cargar parámetros óptimos por liga
        self.kelly_fractions = {'DEFAULT': 0.015}
        self.ev_thresholds = {'DEFAULT': 0.015}
        self.alpha_div_low = {'DEFAULT': 0.85}
        self.alpha_div_med = {'DEFAULT': 0.70}
        self.alpha_div_high = {'DEFAULT': 0.50}
        
        params_file = os.path.join(core_dir, '..', 'data', 'processed', 'models_best_parameters', 'optimal_bankroll_params.json')
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    data = json.load(f)
                    self.kelly_fractions.update(data.get('KELLY_FRACTIONS', {}))
                    self.ev_thresholds.update(data.get('EV_THRESHOLDS', {}))
                    self.alpha_div_low.update(data.get('ALPHA_DIV_LOW', {}))
                    self.alpha_div_med.update(data.get('ALPHA_DIV_MED', {}))
                    self.alpha_div_high.update(data.get('ALPHA_DIV_HIGH', {}))
                logger.info(f"Parámetros óptimos cargados de {params_file}")
            except Exception as e:
                logger.error(f"Error leyendo parámetros óptimos: {e}")

    def evaluate_opportunities(self, match_id, competition, probs, poly_prices, pred_clv):
        """
        Calcula el EV considerando impuestos y CLV drop. 
        Soporta Single Bets y Dutching (1X, X2) adaptado al formato de Polymarket (comprar 2 tokens).
        
        probs: dict {'home_prob': 0.60, 'draw_prob': 0.25, 'away_prob': 0.15}
        poly_prices: dict {'home': 0.50, 'draw': 0.25, 'away': 0.25}
        pred_clv: dict {'home': 0.012, 'draw': -0.005, 'away': -0.015}
        """
        league_ev_thresh = self.ev_thresholds.get(competition, self.ev_thresholds.get('DEFAULT', 0.015))
        league_kelly = self.kelly_fractions.get(competition, self.kelly_fractions.get('DEFAULT', 0.015))
        league_alpha_low = self.alpha_div_low.get(competition, self.alpha_div_low.get('DEFAULT', 0.85))
        league_alpha_med = self.alpha_div_med.get(competition, self.alpha_div_med.get('DEFAULT', 0.70))
        league_alpha_high = self.alpha_div_high.get(competition, self.alpha_div_high.get('DEFAULT', 0.50))
        
        # Calcular Net Odds y EV por cada token en Polymarket
        net_odds = {}
        evs = {}
        
        for outcome in ['home', 'draw', 'away']:
            if outcome in poly_prices and poly_prices[outcome] > 0:
                # REGLA CLV: Si el CLV esperado es menor a 0.005 (0.5%), rechazamos el trade inmediatamente
                clv_val = pred_clv.get(outcome, 0)
                if clv_val < 0.005:
                    logger.debug(f"[{match_id}] {outcome} rechazado por bajo CLV esperado ({clv_val*100:.2f}%)")
                    evs[outcome] = -1.0
                    continue
                    
                # El "precio" en PM equivale a la probabilidad implícita (market_prob)
                market_prob = poly_prices[outcome]
                gross_odd = 1.0 / market_prob
                # Aplicamos impuesto sobre las ganancias
                net_odd = 1 + (gross_odd - 1) * (1 - self.tax_retention_rate)
                
                # Dynamic Alpha Blending
                prob = probs[f"{outcome}_prob"]
                divergence = abs(prob - market_prob)
                if divergence > 0.20:
                    dynamic_alpha = 0.30
                elif divergence > 0.15:
                    dynamic_alpha = 0.50
                elif divergence > 0.10:
                    dynamic_alpha = league_alpha_high
                elif divergence > 0.05:
                    dynamic_alpha = league_alpha_med
                else:
                    dynamic_alpha = league_alpha_low
                    
                blended_prob = (dynamic_alpha * prob) + ((1 - dynamic_alpha) * market_prob)
                
                ev = (blended_prob * net_odd) - 1 - self.expected_clv_drop
                
                net_odds[outcome] = net_odd
                evs[outcome] = ev
                # Guardamos la blended_prob para cálculos posteriores
                probs[f"{outcome}_prob"] = blended_prob
            else:
                evs[outcome] = -1.0

        # Identificar la mejor opción simple
        best_single_outcome = max(evs, key=evs.get)
        best_single_ev = evs[best_single_outcome]
        
        # Límite absoluto de liquidez para la liga
        league_liquidity = self.max_bet_liquidity.get(competition, self.max_bet_liquidity.get('DEFAULT', 2000.0))
        
        # Evaluar Dutching (Doble Oportunidad Local/Empate o Empate/Visita)
        # En Polymarket, dutchear significa repartir capital entre los tokens de Local y Empate
        bet_plan = None
        
        if evs['home'] > league_ev_thresh and evs['draw'] > league_ev_thresh:
            logger.info(f"[{match_id}] Dutching 1X Detectado (Ambos superan EV Thresh)")
            impl_home = poly_prices['home']
            impl_draw = poly_prices['draw']
            
            combined_implied = impl_home + impl_draw
            combined_odd = 1 / combined_implied
            net_combined_odd = 1 + (combined_odd - 1) * (1 - self.tax_retention_rate)
            
            blended_prob_1X = probs['home_prob'] + probs['draw_prob']
            ev_1X = (blended_prob_1X * net_combined_odd) - 1 - self.expected_clv_drop
            
            if ev_1X > league_ev_thresh:
                # Calcular Kelly para Dutching
                b = net_combined_odd - 1
                kelly_ev = min(ev_1X, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if combined_odd < 1.30:
                    kelly_pct = min(kelly_pct, 0.01)
                stake_pct = min(kelly_pct * league_kelly, self.max_stake_pct)
                
                if stake_pct >= 0.001:
                    total_stake = min(self.bankroll * stake_pct, league_liquidity)
                    stake_home = total_stake * (impl_home / combined_implied)
                    stake_draw = total_stake * (impl_draw / combined_implied)
                    
                    bet_plan = {
                        "type": "DUTCHING",
                        "ev": ev_1X,
                        "total_stake": total_stake,
                        "orders": [
                            {"outcome": "home", "stake": stake_home, "price": impl_home},
                            {"outcome": "draw", "stake": stake_draw, "price": impl_draw}
                        ]
                    }
                    
        elif evs['away'] > league_ev_thresh and evs['draw'] > league_ev_thresh:
            logger.info(f"[{match_id}] Dutching X2 Detectado")
            impl_away = poly_prices['away']
            impl_draw = poly_prices['draw']
            
            combined_implied = impl_away + impl_draw
            combined_odd = 1 / combined_implied
            net_combined_odd = 1 + (combined_odd - 1) * (1 - self.tax_retention_rate)
            
            blended_prob_X2 = probs['away_prob'] + probs['draw_prob']
            ev_X2 = (blended_prob_X2 * net_combined_odd) - 1 - self.expected_clv_drop
            
            if ev_X2 > league_ev_thresh:
                b = net_combined_odd - 1
                kelly_ev = min(ev_X2, 0.15)
                kelly_pct = kelly_ev / b if b > 0 else 0
                if combined_odd < 1.30:
                    kelly_pct = min(kelly_pct, 0.01)
                stake_pct = min(kelly_pct * league_kelly, self.max_stake_pct)
                
                if stake_pct >= 0.001:
                    total_stake = min(self.bankroll * stake_pct, league_liquidity)
                    stake_away = total_stake * (impl_away / combined_implied)
                    stake_draw = total_stake * (impl_draw / combined_implied)
                    
                    bet_plan = {
                        "type": "DUTCHING",
                        "ev": ev_X2,
                        "total_stake": total_stake,
                        "orders": [
                            {"outcome": "away", "stake": stake_away, "price": impl_away},
                            {"outcome": "draw", "stake": stake_draw, "price": impl_draw}
                        ]
                    }
                    
        # Si no hay Dutching, evaluamos el Single
        if not bet_plan and best_single_ev > league_ev_thresh:
            b = net_odds[best_single_outcome] - 1
            kelly_ev = min(best_single_ev, 0.15)
            kelly_pct = kelly_ev / b if b > 0 else 0
            
            raw_odd = 1.0 / poly_prices[best_single_outcome]
            if raw_odd < 1.30:
                kelly_pct = min(kelly_pct, 0.01)
                
            stake_pct = min(kelly_pct * league_kelly, self.max_stake_pct)
            if stake_pct >= 0.001:
                stake = min(self.bankroll * stake_pct, league_liquidity)
                bet_plan = {
                    "type": "SINGLE",
                    "ev": best_single_ev,
                    "total_stake": stake,
                    "orders": [
                        {"outcome": best_single_outcome, "stake": stake, "price": poly_prices[best_single_outcome]}
                    ]
                }
                
        if bet_plan:
            logger.info(f"[{match_id}] Estrategia Elegida: {bet_plan['type']} (EV Neto: +{bet_plan['ev']*100:.2f}%) - Stake Total: ${bet_plan['total_stake']:.2f}")
            
        return bet_plan

    def register_position(self, token_id, amount):
        """Registra una nueva posición y resta el bankroll."""
        self.active_positions[token_id] = self.active_positions.get(token_id, 0) + amount
        self.bankroll -= amount
        logger.info(f"✅ Posición Confirmada: ${amount:.2f} en {token_id}. Bankroll restante: ${self.bankroll:.2f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pm = PortfolioManager(core_dir=".", initial_bankroll=1000.0)
    
    probs_mock = {'home_prob': 0.60, 'draw_prob': 0.25, 'away_prob': 0.15}
    poly_prices_mock = {'home': 0.50, 'draw': 0.30, 'away': 0.20}
    
    plan = pm.evaluate_opportunities("Arsenal_vs_Chelsea", "E1", probs_mock, poly_prices_mock)
    print(plan)
