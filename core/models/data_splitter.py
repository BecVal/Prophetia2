import os
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Intenta cargar el dataset con cuotas, si no, usa el base.
DATASET_PATH = '../data/processed/matches_with_odds.parquet'
FALLBACK_DATASET = '../data/processed/matches_dataset.parquet'

def get_base_dataset():
    """
    Carga el dataset principal procesado por feature_engineering.py,
    lo ordena cronológicamente y filtra para evitar Double-Row Betting.
    """
    path_to_load = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    if not os.path.exists(path_to_load):
        raise FileNotFoundError(f"Dataset no encontrado en {path_to_load}. Ejecuta feature_engineering.py primero.")

    logger.info(f"Cargando dataset procesado: {path_to_load}...")
    df = pd.read_parquet(path_to_load, engine='fastparquet')

    if 'match_date' in df.columns:
        logger.info("Ordenando el dataset cronológicamente para evitar Data Leakage...")
        df = df.sort_values('match_date').reset_index(drop=True)
    else:
        logger.warning("No se encontró columna 'match_date'. Posible Data Leakage.")

    logger.info("Filtrando eventos solo desde la perspectiva local para evitar Double-Row Betting...")
    df = df[df['is_home'] == 1].reset_index(drop=True)
    
    return df

def get_train_test_split(df, train_ratio=0.8):
    """
    Devuelve el índice exacto donde se divide Train y Test temporalmente.
    Todos los modelos DEBEN usar esta misma función para garantizar alineación.
    """
    split_idx = int(len(df) * train_ratio)
    logger.info(f"Split Index temporal: {split_idx} (Train: {split_idx}, Test: {len(df) - split_idx})")
    return split_idx

def get_cv_strategy(n_splits=5):
    """
    Estrategia de validación cruzada consistente (TimeSeriesSplit) para todos los modelos.
    """
    return TimeSeriesSplit(n_splits=n_splits)
