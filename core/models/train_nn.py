import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = '../core/save_models/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'nn_model.pkl')
PROCESSED_DIR = '../data/processed'

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class SklearnPyTorchWrapper:
    def __init__(self, input_dim, num_classes=3, epochs=20, batch_size=64, lr=1e-3, device='cuda'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()
        
    def _init_model(self):
        self.model = PyTorchMLP(self.input_dim, self.num_classes).to(self.device)
        
    def fit(self, X, y, sample_weight=None):
        self._init_model() # Reinicializar para asegurar pesos limpios por cada fold
        
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        if sample_weight is not None:
            w_vals = sample_weight.values if hasattr(sample_weight, 'values') else sample_weight
            w_tensor = torch.FloatTensor(w_vals).to(self.device)
        else:
            w_tensor = torch.ones(len(X)).to(self.device)
            
        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        self.model.train()
        for epoch in range(self.epochs):
            for bx, by, bw in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss = (loss * bw).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
        
    def get_params(self):
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': self.device
        }

def train_nn():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    base_stats = [
        'shots_total', 'shots_on_target',
        'passes_total', 'passes_completed', 'pass_accuracy', 'possession_pct',
        'crosses', 'corners', 'through_balls', 'key_passes',
        'dribbles_completed', 'pressures', 'interceptions', 'clearances',
        'blocks', 'ball_recoveries', 'actions_under_pressure',
        'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
        'aerials_won'
    ]
    
    feature_cols = [
        'is_home', 'rest_days', 'rest_diff',
        'team_squad_value', 'opp_squad_value', 'squad_value_diff',
        'h2h_games_played', 'h2h_points_last_5', 'h2h_win_rate_hist', 'h2h_draw_rate_hist', 'is_european_hangover',
        'win_streak_3', 'loss_streak_3', 'xg_momentum_macd', 
        'opp_win_streak_3', 'opp_loss_streak_3', 'opp_xg_momentum_macd',
        'fatigue_index', 'fatigue_diff', 'xg_volatility_5', 'opp_xg_volatility_5', 'volatility_diff'
    ]
    
    for stat in base_stats:
        feature_cols.append(f"{stat}_ema3")
        feature_cols.append(f"{stat}_ema5")
        
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Faltan las siguientes columnas en NN: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    logger.info("Configurando Red Neuronal (PyTorch)...")
    
    input_dim = X_train.shape[1]
    # Parámetros fijos, o podrías usar Optuna aquí también, pero para NN un buen default suele bastar
    nn_best = SklearnPyTorchWrapper(input_dim=input_dim, epochs=25, batch_size=128, lr=0.001)
    
    logger.info("Calculando predicciones OOF para Train (NN)...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando KFold
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para evitar Leakage...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for kf_train, kf_val in kf.split(X_first):
        X_kf_train, y_kf_train = X_first.iloc[kf_train], y_first.iloc[kf_train]
        X_kf_val = X_first.iloc[kf_val]
        
        dates_kf_train = dates_first.iloc[kf_train] if dates_first is not None else None
        w_tr = get_time_weights(dates_kf_train)
        
        kf_estimator = SklearnPyTorchWrapper(**nn_best.get_params())
        kf_estimator.fit(X_kf_train, y_kf_train, sample_weight=w_tr)
        
        val_indices_in_original = first_train_idx[kf_val]
        pred_probs_train[val_indices_in_original] = kf_estimator.predict_proba(X_kf_val)

    # 2. Expanding Windows estándar para el resto
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)
        
        fold_estimator = SklearnPyTorchWrapper(**nn_best.get_params())
        fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)
        
    logger.info("Entrenando Modelo NN final y prediciendo Test...")
    final_w_tr = get_time_weights(train_dates)
    nn_best.fit(X_train, y_train, sample_weight=final_w_tr)
    pred_probs_test = nn_best.predict_proba(X_test)
    
    # LOGS: Verificacion de calibracion
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO NN ===")
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = pred_probs_train[:, 0].mean()
    pred_draw = pred_probs_train[:, 1].mean()
    pred_win = pred_probs_train[:, 2].mean()
    
    logger.info(f" - Derrota (Loss) | Predicha: {pred_loss*100:.1f}% | Real en Dataset: {real_loss*100:.1f}%")
    logger.info(f" - Empate (Draw)  | Predicha: {pred_draw*100:.1f}% | Real en Dataset: {real_draw*100:.1f}%")
    logger.info(f" - Victoria (Win) | Predicha: {pred_win*100:.1f}% | Real en Dataset: {real_win*100:.1f}%")
    
    # Guardar OOF
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_nn', 'prob_draw_nn', 'prob_win_nn'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_nn', 'prob_draw_nn', 'prob_win_nn'], index=X_test.index)
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_nn_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_nn_test.parquet'), engine='fastparquet')
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump({'model': nn_best, 'features': feature_cols}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO NN FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_nn()
