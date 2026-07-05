import os
import logging
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

logger = logging.getLogger('polymarket_bot.models.train_microstructure')

class MicrostructureModel:
    """
    Reciclando los conceptos de Prophetia2 (train_market.py), este modelo 
    busca predecir la trayectoria del precio a corto plazo basado en 
    el Order Flow Imbalance y la volatilidad del Spread.
    """
    def __init__(self, model_path="microstructure_xgb.pkl"):
        self.model_path = model_path
        self.model = None

    def build_features(self, df_orderflow):
        """
        df_orderflow debe contener snapshots del orderbook en el tiempo.
        Columnas esperadas: timestamp, best_bid, best_ask, bid_vol, ask_vol
        """
        df = df_orderflow.copy()
        
        # 1. Spread Volatility
        df['spread'] = df['best_ask'] - df['best_bid']
        df['spread_rolling_std'] = df['spread'].rolling(window=10).std().fillna(0)
        
        # 2. Order Flow Imbalance (OFI)
        # Ratio de volumen entre Bid (Demanda) y Ask (Oferta)
        df['vol_imbalance'] = (df['bid_vol'] - df['ask_vol']) / (df['bid_vol'] + df['ask_vol'] + 1e-5)
        
        # 3. Momentum (Cambio en Mid Price)
        df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
        df['price_momentum_5m'] = df['mid_price'].pct_change(periods=5).fillna(0)
        
        # 4. Target: Cambio de precio futuro en los próximos 10 periodos
        df['target_price_change'] = df['mid_price'].shift(-10) - df['mid_price']
        
        return df.dropna()

    def train(self, df_raw):
        logger.info("Construyendo features de microestructura...")
        df = self.build_features(df_raw)
        
        features = ['spread', 'spread_rolling_std', 'vol_imbalance', 'price_momentum_5m']
        X = df[features]
        y = df['target_price_change']
        
        logger.info("Entrenando XGBoost Regressor para predecir trayectoria de precio...")
        self.model = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3,
            random_state=42
        )
        self.model.fit(X, y)
        
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo de microestructura guardado en {self.model_path}")
        
    def predict_trajectory(self, current_snapshot):
        """
        Predice si el precio subirá o bajará a corto plazo.
        Si la predicción es negativa, conviene esperar para comprar más barato.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                logger.warning("No hay modelo entrenado.")
                return 0.0
                
        df = pd.DataFrame([current_snapshot])
        # Nota: En producción real, current_snapshot necesita contexto histórico (rolling stats)
        # Aquí simplificamos asumiendo que ya viene con las variables calculadas
        features = ['spread', 'spread_rolling_std', 'vol_imbalance', 'price_momentum_5m']
        
        pred = self.model.predict(df[features])
        return pred[0]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Dummy data generator
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
        'best_bid': np.random.uniform(0.40, 0.50, 1000),
        'best_ask': np.random.uniform(0.42, 0.52, 1000),
        'bid_vol': np.random.randint(100, 10000, 1000),
        'ask_vol': np.random.randint(100, 10000, 1000)
    })
    
    # Clean up dummy data so ask > bid
    dummy_data['best_ask'] = dummy_data[['best_bid', 'best_ask']].max(axis=1)
    dummy_data['best_bid'] = dummy_data[['best_bid', 'best_ask']].min(axis=1)
    
    model = MicrostructureModel()
    model.train(dummy_data)
