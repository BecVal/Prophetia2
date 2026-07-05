import logging
import math

logger = logging.getLogger('polymarket_bot.core.vwap_slicer')

class VWAPSlicer:
    def __init__(self, max_slippage_tolerance=0.015):
        """
        max_slippage_tolerance: Slippage máximo permitido frente al precio mid de mercado.
                                Ejemplo: 0.015 = 1.5%. Si el impacto de la orden supera esto,
                                la orden se divide (Slicer) y se ejecuta como Maker.
        """
        self.max_slippage_tolerance = max_slippage_tolerance

    def calculate_taker_execution(self, order_book, target_shares_to_buy):
        """
        Calcula el precio promedio si compramos 'target_shares_to_buy' 
        inmediatamente consumiendo la liquidez del Order Book (Taker).
        Retorna:
            - vwap (Volume Weighted Average Price)
            - slippage (Impacto en el precio comparado con el mejor precio disponible)
        """
        asks = order_book.get('asks', [])
        if not asks:
            return 0.0, 0.0
            
        # Asks vienen ordenados de menor a mayor precio
        # Ej: [{"price": "0.45", "size": "100"}, {"price": "0.46", "size": "300"}]
        
        shares_remaining = target_shares_to_buy
        total_cost = 0.0
        
        best_ask = float(asks[0]['price'])
        
        for ask in asks:
            price = float(ask['price'])
            size = float(ask['size'])
            
            if size >= shares_remaining:
                total_cost += shares_remaining * price
                shares_remaining = 0
                break
            else:
                total_cost += size * price
                shares_remaining -= size

        if shares_remaining > 0:
            logger.warning("Liquidez insuficiente en el Order Book para llenar toda la orden.")
            return float('inf'), float('inf') # Indica imposibilidad de llenado

        vwap = total_cost / target_shares_to_buy
        slippage = (vwap - best_ask) / best_ask
        
        return vwap, slippage

    def slice_order(self, order_book, target_shares, num_slices=3):
        """
        Si el slippage es inaceptable, esta función fragmenta la orden grande 
        en pequeñas limit orders que actúan como Maker.
        """
        vwap, slippage = self.calculate_taker_execution(order_book, target_shares)
        
        if slippage <= self.max_slippage_tolerance:
            logger.info(f"Slippage aceptable ({slippage:.2%}). Ejecutando como Market/Taker al VWAP de {vwap:.4f}")
            return [{"type": "TAKER", "shares": target_shares}]
        
        logger.info(f"Slippage inaceptable ({slippage:.2%} > {self.max_slippage_tolerance:.2%}). Slicing orden (Maker)...")
        
        shares_per_slice = math.floor(target_shares / num_slices)
        slices = []
        
        best_ask = float(order_book['asks'][0]['price'])
        
        for i in range(num_slices):
            # En un entorno real, colocarías las órdenes en el mejor bid o ligeramente por debajo del mejor ask
            # Aquí lo simulamos pegándonos un centavo por debajo del mejor ask
            limit_price = round(best_ask - 0.005, 3) 
            
            # El último slice toma el remanente por problemas de redondeo
            if i == num_slices - 1:
                shares = target_shares - (shares_per_slice * (num_slices - 1))
            else:
                shares = shares_per_slice
                
            slices.append({
                "type": "MAKER_LIMIT",
                "price": limit_price,
                "shares": shares
            })
            
        return slices

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    slicer = VWAPSlicer()
    mock_book = {
        "asks": [
            {"price": "0.50", "size": "500"},
            {"price": "0.51", "size": "1000"},
            {"price": "0.53", "size": "5000"}
        ]
    }
    
    # 1. Comprar 300 acciones (Debería entrar sin slippage)
    print(slicer.slice_order(mock_book, 300))
    
    print("-" * 40)
    
    # 2. Comprar 2000 acciones (Comerá liquidez de .50, .51 y .53 -> Alto slippage)
    print(slicer.slice_order(mock_book, 2000))
