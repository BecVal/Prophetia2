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
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss, accuracy_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson
import json
import optuna

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
# Cambia RUN_OPTUNA a True si deseas volver a buscar los mejores hiperparámetros.
# De lo contrario (False), cargará los mejores guardados en el archivo JSON.
RUN_OPTUNA = True
OPTUNA_TRIALS = 30
# ==============================================================================

logger = get_logger(__name__, 'train_shots_on_goal')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_shots.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/save_models'))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'shots_on_goal_model.pkl')
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def get_time_weights(dates, half_life_days=365):
    if dates is None:
        return None
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days)

class PyTorchMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1)) # Output 1 para regresión continua
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x).squeeze(-1) # Shape: (batch_size,)

class SklearnPyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3, weight_decay=1e-2, epochs=100, batch_size=128, lr=1e-3, device='cuda', patience=7):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def _init_model(self):
        self.model = PyTorchMLPRegressor(self.input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        
    def fit(self, X, y, sample_weight=None):
        self._init_model()
        
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
        y_tr_t = torch.FloatTensor(y_tr.values if hasattr(y_tr, 'values') else y_tr).to(self.device)
        w_tr_t = torch.FloatTensor(w_tr.values if hasattr(w_tr, 'values') else w_tr).to(self.device) if w_tr is not None else torch.ones(len(X_tr)).to(self.device)
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).to(self.device)
        w_val_t = torch.FloatTensor(w_val.values if hasattr(w_val, 'values') else w_val).to(self.device) if w_val is not None else torch.ones(len(X_val)).to(self.device)
            
        dataset = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss(reduction='none')
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
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss = (loss * bw).sum() / bw.sum()
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = criterion(val_preds, y_val_t)
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

    def predict(self, X):
        self.model.eval()
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor)
        return preds.cpu().numpy()
        
    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': self.device,
            'patience': self.patience
        }

class StackingRegressorEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, mlp_params, ridge_alpha=1.0, meta_alpha=1.0):
        self.mlp_params = mlp_params
        self.ridge_alpha = ridge_alpha
        self.meta_alpha = meta_alpha
        
        self.mlp_model = SklearnPyTorchRegressorWrapper(**mlp_params)
        
        # Necesitamos un escalador e imputer externo para Ridge
        self.ridge_imputer = SimpleImputer(strategy='median')
        self.ridge_scaler = StandardScaler()
        self.ridge_model = Ridge(alpha=ridge_alpha)
        
        self.meta_model = Ridge(alpha=meta_alpha)
        
    def fit(self, X, y, sample_weight=None):
        # 1. Split interno para entrenar Meta-Model sin Data Leakage
        # Como es serie temporal, tomamos el 80% para Base Models, 20% para Meta
        split_idx = int(len(X) * 0.8)
        
        X_base = X.iloc[:split_idx]
        y_base = y.iloc[:split_idx]
        w_base = sample_weight.iloc[:split_idx] if hasattr(sample_weight, 'iloc') else (sample_weight[:split_idx] if sample_weight is not None else None)
        
        X_meta = X.iloc[split_idx:]
        y_meta = y.iloc[split_idx:]
        w_meta = sample_weight.iloc[split_idx:] if hasattr(sample_weight, 'iloc') else (sample_weight[split_idx:] if sample_weight is not None else None)
        
        # 2. Entrenar Base Models
        self.mlp_model.fit(X_base, y_base, sample_weight=w_base)
        
        X_base_scaled = self.ridge_scaler.fit_transform(self.ridge_imputer.fit_transform(X_base))
        self.ridge_model.fit(X_base_scaled, y_base, sample_weight=w_base)
        
        # 3. Generar Predicciones Base para Meta-Model
        meta_mlp_preds = self.mlp_model.predict(X_meta)
        X_meta_scaled = self.ridge_scaler.transform(self.ridge_imputer.transform(X_meta))
        meta_ridge_preds = self.ridge_model.predict(X_meta_scaled)
        
        # 4. Entrenar Meta-Model
        # Agregamos las predicciones como características, y podemos agregar unas cuantas features originales clave
        meta_X = np.column_stack([meta_mlp_preds, meta_ridge_preds])
        self.meta_model.fit(meta_X, y_meta, sample_weight=w_meta)
        
        # 5. Re-entrenar Base Models en todo el dataset (X, y) para máximo rendimiento final
        self.mlp_model.fit(X, y, sample_weight=sample_weight)
        
        X_full_scaled = self.ridge_scaler.fit_transform(self.ridge_imputer.fit_transform(X))
        self.ridge_model.fit(X_full_scaled, y, sample_weight=sample_weight)
        
        return self
        
    def predict(self, X):
        mlp_preds = self.mlp_model.predict(X)
        X_scaled = self.ridge_scaler.transform(self.ridge_imputer.transform(X))
        ridge_preds = self.ridge_model.predict(X_scaled)
        
        meta_X = np.column_stack([mlp_preds, ridge_preds])
        final_preds = self.meta_model.predict(meta_X)
        
        # Evitar tiros al arco negativos
        return np.maximum(final_preds, 0.1)


def calc_over_probs(lambda_vals, lines):
    probs = {}
    for line in lines:
        k = int(np.floor(line))
        prob_over = 1.0 - poisson.cdf(k, lambda_vals)
        probs[f'prob_over_{line}'] = prob_over
    return probs


def train_shots_on_goal():
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    if 'shots_on_target' not in df.columns:
        logger.error("No se encontró la columna 'shots_on_target' en el dataset.")
        return

    # === CARGAR xG DEL MODELO QUANT ===
    quant_train_path = os.path.join(PROCESSED_DIR, 'oof_quant_train.parquet')
    quant_test_path = os.path.join(PROCESSED_DIR, 'oof_quant_test.parquet')
    
    if os.path.exists(quant_train_path) and os.path.exists(quant_test_path):
        logger.info("Cargando variables xG del modelo Quant Advanced...")
        q_tr = pd.read_parquet(quant_train_path)
        q_ts = pd.read_parquet(quant_test_path)
        q_full = pd.concat([q_tr, q_ts])
        
        df = df.join(q_full[['predicted_xg_scored_quant', 'predicted_xg_conceded_quant']], how='left')
    else:
        logger.warning("No se encontraron predicciones Quant. Ejecuta train_quant_advanced.py primero.")
        df['predicted_xg_scored_quant'] = 0.0
        df['predicted_xg_conceded_quant'] = 0.0
    
    # Features Clave Solicitadas
    feature_cols = [
        'shots_total_ema3', 'shots_total_ema5',
        'pass_accuracy_ema3', 'pass_accuracy_ema5',
        'team_squad_value', 'opp_squad_value',
        'predicted_xg_scored_quant', 'predicted_xg_conceded_quant',
        'is_home', 'rest_days',
        'possession_pct_ema3', 'possession_pct_ema5'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan variables en Tiros al Arco: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['shots_on_target'].copy()
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
        
    logger.info("=== ENTRENANDO MODELO DE TIROS AL ARCO (STACKING MLP+RIDGE) ===")
    
    # Preparamos split para Optuna
    opt_split = int(len(X_train) * 0.8)
    X_opt_train, y_opt_train = X_train.iloc[:opt_split], y_train.iloc[:opt_split]
    X_opt_val, y_opt_val = X_train.iloc[opt_split:], y_train.iloc[opt_split:]
    w_opt_train = get_time_weights(train_dates.iloc[:opt_split]) if train_dates is not None else None
    
    input_dim = X_train.shape[1]
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dims = [trial.suggest_categorical(f'n_units_l{i}', [32, 64, 128]) for i in range(n_layers)]
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 100.0, log=True)
        meta_alpha = trial.suggest_float('meta_alpha', 0.1, 10.0, log=True)
        
        mlp_params = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'lr': lr,
            'epochs': 50, # Reducido para Optuna
            'batch_size': 256,
            'patience': 4
        }
        
        stack = StackingRegressorEnsemble(mlp_params, ridge_alpha, meta_alpha)
        stack.fit(X_opt_train, y_opt_train, sample_weight=w_opt_train)
        preds = stack.predict(X_opt_val)
        
        mse = mean_squared_error(y_opt_val, preds)
        return mse

    if RUN_OPTUNA:
        logger.info(f"Optimizando Stacking con Optuna ({OPTUNA_TRIALS} Trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Mejores parámetros guardados en {OPTUNA_PARAMS_FILE}")
    else:
        logger.info("Cargando mejores parámetros guardados...")
        if os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
        else:
            logger.warning(f"Archivo {OPTUNA_PARAMS_FILE} no encontrado. Ejecutando Optuna...")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params, f, indent=4)
                
    best_mlp = {
        'input_dim': input_dim,
        'hidden_dims': [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])],
        'dropout_rate': best_params['dropout_rate'],
        'lr': best_params['lr'],
        'epochs': 100, # Vuelve a normal para OOF
        'batch_size': 256,
        'patience': 7
    }
    
    # OUT OF FOLD PREDICTIONS CON EXPANDING WINDOW
    tscv = get_cv_strategy(n_splits=5)
    preds_train = np.zeros(len(X_train))
    preds_train[:] = np.nan
    
    splits = list(tscv.split(X_train))
    
    # 1. Fold Cero KFold
    first_train_idx = splits[0][0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    logger.info(f"  -> Fold Inicial KFold ({len(first_train_idx)} muestras)...")
    for kf_train, kf_val in kf.split(X_train.iloc[first_train_idx]):
        X_kf_tr, y_kf_tr = X_train.iloc[first_train_idx].iloc[kf_train], y_train.iloc[first_train_idx].iloc[kf_train]
        X_kf_va = X_train.iloc[first_train_idx].iloc[kf_val]
        
        dates_kf_train = train_dates.iloc[first_train_idx].iloc[kf_train] if train_dates is not None else None
        w_tr = get_time_weights(dates_kf_train) if dates_kf_train is not None else None
        
        stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'])
        stack.fit(X_kf_tr, y_kf_tr, sample_weight=w_tr)
        
        val_idx_orig = first_train_idx[kf_val]
        preds_train[val_idx_orig] = stack.predict(X_kf_va)

    # 2. Expanding Windows
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Fold Temporal {i+1}/{len(splits)}...")
        w_tr = get_time_weights(train_dates.iloc[train_idx]) if train_dates is not None else None
        
        stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'])
        stack.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], sample_weight=w_tr)
        preds_train[val_idx] = stack.predict(X_train.iloc[val_idx])
        
    logger.info("Entrenando Modelo Final Stacking en todo Train...")
    final_weights = get_time_weights(train_dates)
    final_stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'])
    final_stack.fit(X_train, y_train, sample_weight=final_weights)
    preds_test = final_stack.predict(X_test)
    
    # --- CONVOLUCIÓN POISSON Y LÍNEAS DE APUESTAS ---
    logger.info("=== CALCULANDO LÍNEAS POISSON Y MÉTRICAS DE APUESTAS ===")
    
    df_train_tmp = df.iloc[:split_idx].copy()
    df_train_tmp['pred_shots'] = preds_train
    
    df_test_tmp = df.iloc[split_idx:].copy()
    df_test_tmp['pred_shots'] = preds_test
    
    # Opponent Shots
    df_train_tmp['opp_pred_shots'] = df_train_tmp.groupby('match_id')['pred_shots'].transform('sum') - df_train_tmp['pred_shots']
    df_test_tmp['opp_pred_shots'] = df_test_tmp.groupby('match_id')['pred_shots'].transform('sum') - df_test_tmp['pred_shots']
    
    # Total Shots
    df_train_tmp['lambda_total'] = df_train_tmp['pred_shots'] + df_train_tmp['opp_pred_shots']
    df_test_tmp['lambda_total'] = df_test_tmp['pred_shots'] + df_test_tmp['opp_pred_shots']
    
    df_train_tmp['opp_shots'] = df_train_tmp.groupby('match_id')['shots_on_target'].transform('sum') - df_train_tmp['shots_on_target']
    df_train_tmp['true_total_shots'] = df_train_tmp['shots_on_target'] + df_train_tmp['opp_shots']
    
    df_test_tmp['opp_shots'] = df_test_tmp.groupby('match_id')['shots_on_target'].transform('sum') - df_test_tmp['shots_on_target']
    df_test_tmp['true_total_shots'] = df_test_tmp['shots_on_target'] + df_test_tmp['opp_shots']
    
    # Definición de líneas
    team_lines = [3.5, 4.5, 5.5]
    total_lines = [7.5, 8.5, 9.5]
    
    # Probabilidades de Equipo
    train_team_probs = calc_over_probs(df_train_tmp['pred_shots'].values, team_lines)
    test_team_probs = calc_over_probs(df_test_tmp['pred_shots'].values, team_lines)
    for col, vals in train_team_probs.items(): df_train_tmp[col + '_team'] = vals
    for col, vals in test_team_probs.items(): df_test_tmp[col + '_team'] = vals
        
    # Probabilidades Totales
    train_total_probs = calc_over_probs(df_train_tmp['lambda_total'].values, total_lines)
    test_total_probs = calc_over_probs(df_test_tmp['lambda_total'].values, total_lines)
    for col, vals in train_total_probs.items(): df_train_tmp[col + '_total'] = vals
    for col, vals in test_total_probs.items(): df_test_tmp[col + '_total'] = vals
        
    # Calibración Isotónica para Líneas Totales y de Equipo
    calibrators = {}
    
    # Calibrar Líneas de Equipo
    for line in team_lines:
        col = f'prob_over_{line}_team'
        true_over_train = (df_train_tmp['shots_on_target'] > line).astype(int).values
        ir = IsotonicRegression(out_of_bounds='clip')
        df_train_tmp[col] = ir.fit_transform(df_train_tmp[col].values, true_over_train)
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir
        
    # Calibrar Líneas Totales
    for line in total_lines:
        col = f'prob_over_{line}_total'
        true_over_train = (df_train_tmp['true_total_shots'] > line).astype(int).values
        ir = IsotonicRegression(out_of_bounds='clip')
        df_train_tmp[col] = ir.fit_transform(df_train_tmp[col].values, true_over_train)
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir

    # AUDITORÍA CONTINUA
    logger.info("--- REGRESIÓN PURA (MSE/MAE) ---")
    mse_tr = mean_squared_error(y_train, preds_train)
    mae_tr = mean_absolute_error(y_train, preds_train)
    mse_ts = mean_squared_error(y_test, preds_test)
    mae_ts = mean_absolute_error(y_test, preds_test)
    logger.info(f"Train - MSE: {mse_tr:.3f} | MAE: {mae_tr:.3f}")
    logger.info(f"Test  - MSE: {mse_ts:.3f} | MAE: {mae_ts:.3f}")

    # AUDITORÍA PROBABILIDADES
    logger.info("--- PROBABILIDADES TOTAL MATCH (Over/Under) ---")
    for line in total_lines:
        true_over_ts = (df_test_tmp['true_total_shots'] > line).astype(int)
        prob_ts = df_test_tmp[f'prob_over_{line}_total']
        
        ll_ts = log_loss(true_over_ts, prob_ts)
        bs_ts = brier_score_loss(true_over_ts, prob_ts)
        acc_ts = accuracy_score(true_over_ts, (prob_ts > 0.5).astype(int))
        logger.info(f"OVER {line} Total | Dist: {true_over_ts.mean()*100:.1f}% | LogLoss: {ll_ts:.4f} | Brier: {bs_ts:.4f} | Acc: {acc_ts:.4f}")
        
    logger.info("--- PROBABILIDADES TEAM MATCH (Over/Under) ---")
    for line in team_lines:
        true_over_ts = (df_test_tmp['shots_on_target'] > line).astype(int)
        prob_ts = df_test_tmp[f'prob_over_{line}_team']
        
        ll_ts = log_loss(true_over_ts, prob_ts)
        bs_ts = brier_score_loss(true_over_ts, prob_ts)
        acc_ts = accuracy_score(true_over_ts, (prob_ts > 0.5).astype(int))
        logger.info(f"OVER {line} Team  | Dist: {true_over_ts.mean()*100:.1f}% | LogLoss: {ll_ts:.4f} | Brier: {bs_ts:.4f} | Acc: {acc_ts:.4f}")

    # GUARDADO
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    cols_to_save = ['pred_shots', 'opp_pred_shots', 'lambda_total'] + \
                   [f'prob_over_{L}_team' for L in team_lines] + \
                   [f'prob_over_{L}_total' for L in total_lines]
                   
    oof_train = df_train_tmp[cols_to_save].copy()
    oof_test = df_test_tmp[cols_to_save].copy()
    
    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_shots_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_shots_test.parquet'), engine='fastparquet')
    
    joblib.dump({
        'model_shots': final_stack,
        'features': feature_cols,
        'team_lines': team_lines,
        'total_lines': total_lines,
        'calibrators': calibrators
    }, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO TIROS AL ARCO FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_shots_on_goal()
