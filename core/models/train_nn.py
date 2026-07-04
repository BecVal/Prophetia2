import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

logger = get_logger(__name__, 'train_nn')


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'core/save_models/')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'nn_model.pkl')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed/')

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3, num_classes=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class SklearnPyTorchWrapper:
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3, weight_decay=1e-2, num_classes=3, epochs=100, batch_size=128, lr=1e-3, device='cuda', patience=7):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def _init_model(self):
        self.model = PyTorchMLP(self.input_dim, self.hidden_dims, self.dropout_rate, self.num_classes).to(self.device)
        
    def fit(self, X, y, sample_weight=None):
        self._init_model() # Reinicializar para asegurar pesos limpios por cada fold
        
        # Validacion Interna para Early Stopping (Split primero para evitar Data Leakage)
        val_idx = int(len(X) * 0.85)
        
        X_tr_raw = X.iloc[:val_idx] if hasattr(X, 'iloc') else X[:val_idx]
        X_val_raw = X.iloc[val_idx:] if hasattr(X, 'iloc') else X[val_idx:]
        
        X_tr_imputed = self.imputer.fit_transform(X_tr_raw)
        X_val_imputed = self.imputer.transform(X_val_raw)
        
        X_tr = self.scaler.fit_transform(X_tr_imputed)
        X_val = self.scaler.transform(X_val_imputed)
        
        y_tr = y.iloc[:val_idx] if hasattr(y, 'iloc') else y[:val_idx]
        y_val = y.iloc[val_idx:] if hasattr(y, 'iloc') else y[val_idx:]
        
        w_tr = sample_weight.iloc[:val_idx] if (sample_weight is not None and hasattr(sample_weight, 'iloc')) else (sample_weight[:val_idx] if sample_weight is not None else None)
        w_val = sample_weight.iloc[val_idx:] if (sample_weight is not None and hasattr(sample_weight, 'iloc')) else (sample_weight[val_idx:] if sample_weight is not None else None)
        
        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.LongTensor(y_tr.values if hasattr(y_tr, 'values') else y_tr).to(self.device)
        w_tr_t = torch.FloatTensor(w_tr.values if hasattr(w_tr, 'values') else w_tr).to(self.device) if w_tr is not None else torch.ones(len(X_tr)).to(self.device)
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val.values if hasattr(y_val, 'values') else y_val).to(self.device)
        w_val_t = torch.FloatTensor(w_val.values if hasattr(w_val, 'values') else w_val).to(self.device) if w_val is not None else torch.ones(len(X_val)).to(self.device)
            
        dataset = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Sin pesos de clase ni label smoothing para no distorsionar las probabilidades reales
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.epochs):
            self.model.train()
            for bx, by, bw in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss = (loss * bw).sum() / bw.sum()
                loss.backward()
                optimizer.step()
                
            # Validacion
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t)
                val_loss = ((val_loss * w_val_t).sum() / w_val_t.sum()).item()
                
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
        
    def get_params(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': self.device,
            'patience': self.patience
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

    X = df[feature_cols].copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
    
    cv_strategy = get_cv_strategy(n_splits=5)
    
    logger.info("Configurando Red Neuronal y Optuna (PyTorch)...")
    
    input_dim = X_train.shape[1]
    
    # === OPTUNA HYPERPARAMETER TUNING ===
    def objective(trial):
        # Hyperparameter search space
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_categorical(f'n_units_l{i}', [64, 128, 256, 512]))
            
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        
        # Validacion Cruzada para evaluar estos parametros
        kf = TimeSeriesSplit(n_splits=3) # Usamos 3 splits en Optuna para velocidad
        fold_losses = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
            w_tr = get_time_weights(dates_tr)
            
            # Usar paciencia menor y menos epochs para Optuna para acelerar (max 50 epochs)
            estimator = SklearnPyTorchWrapper(
                input_dim=input_dim, 
                hidden_dims=hidden_dims, 
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                epochs=50, 
                batch_size=batch_size, 
                lr=lr, 
                patience=4
            )
            
            estimator.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = estimator.predict_proba(X_val)
            fold_loss = log_loss(y_val, preds)
            fold_losses.append(fold_loss)
            
        return np.mean(fold_losses)
        
    # Puedes ajustar 'n_trials' aquí si tarda demasiado. 50 es exhaustivo.
    n_trials = 50 
    logger.info(f"Iniciando Optuna con {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info(f"Mejores hiperparámetros encontrados por Optuna: {best_params}")
    logger.info(f"Mejor Log-Loss en validación (3 folds): {study.best_value:.4f}")
    
    # Reconstruir parametros del mejor trial
    best_n_layers = best_params['n_layers']
    best_hidden_dims = [best_params[f'n_units_l{i}'] for i in range(best_n_layers)]
    best_dropout = best_params['dropout_rate']
    best_lr = best_params['lr']
    best_wd = best_params['weight_decay']
    best_bs = best_params['batch_size']
    
    # Configurar el mejor modelo final (con patience normal y epochs completas)
    nn_best = SklearnPyTorchWrapper(
        input_dim=input_dim, 
        hidden_dims=best_hidden_dims,
        dropout_rate=best_dropout,
        weight_decay=best_wd,
        epochs=100, 
        batch_size=best_bs, 
        lr=best_lr, 
        patience=7
    )
    
    logger.info("Calculando predicciones OOF para Train (NN) con el MEJOR modelo...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan
    
    splits = list(cv_strategy.split(X_train, y_train))
    
    # 1. Resolver el Leakage del Fold Inicial usando KFold
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None
    
    from sklearn.model_selection import KFold
    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con KFold(5) para obtener OOF completos...")
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
    
    # Calcular y mostrar métricas OOF
    valid_idx = ~np.isnan(pred_probs_train[:, 0]) # Excluir NaNs del primer split
    oof_preds_clean = pred_probs_train[valid_idx]
    y_train_clean = y_train.iloc[valid_idx]
    
    oof_acc = accuracy_score(y_train_clean, np.argmax(oof_preds_clean, axis=1))
    oof_logloss = log_loss(y_train_clean, oof_preds_clean)
    
    logger.info(f"OOF Accuracy: {oof_acc*100:.2f}%")
    logger.info(f"OOF Log-Loss: {oof_logloss:.4f}")
    logger.info("-" * 40)
    
    real_loss = (y_train == 0).mean()
    real_draw = (y_train == 1).mean()
    real_win = (y_train == 2).mean()
    
    pred_loss = oof_preds_clean[:, 0].mean()
    pred_draw = oof_preds_clean[:, 1].mean()
    pred_win = oof_preds_clean[:, 2].mean()
    
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
