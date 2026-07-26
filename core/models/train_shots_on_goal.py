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
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson, nbinom
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

def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"🚀 [CUDA ENABLED] GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ [CUDA DISABLED] Usando CPU. El entrenamiento será más lento.")
    return device

def get_time_weights(dates, half_life_days=365):
    if dates is None or len(dates) == 0:
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
        layers.append(nn.Linear(in_dim, 2)) # Multi-Output: [Home, Away]
        layers.append(nn.Softplus()) # Output > 0 para Poisson
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x) # Shape: (batch_size, 2)
        return torch.clamp(out, min=1e-4) # Evitar ceros exactos para PoissonNLLLoss

class SklearnPyTorchRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.3, weight_decay=1e-2, epochs=100, batch_size=256, lr=1e-3, device='cpu', patience=7):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def _init_model(self):
        self.model = PyTorchMLPRegressor(self.input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        
    def fit(self, X, y, sample_weight=None):
        self._init_model()
        
        val_idx = int(len(X) * 0.85)
        if val_idx < 1:
            val_idx = len(X)
            
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        X_tr = X_scaled[:val_idx]
        X_val = X_scaled[val_idx:] if val_idx < len(X) else X_scaled[:val_idx]
        
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        y_tr = y_arr[:val_idx]
        y_val = y_arr[val_idx:] if val_idx < len(y_arr) else y_arr[:val_idx]
        
        w_arr = sample_weight.values if hasattr(sample_weight, 'values') else np.asarray(sample_weight) if sample_weight is not None else None
        w_tr = w_arr[:val_idx] if w_arr is not None else None
        w_val = w_arr[val_idx:] if w_arr is not None else None
        
        X_tr_t = torch.FloatTensor(np.ascontiguousarray(X_tr)).to(self.device)
        y_tr_t = torch.FloatTensor(np.ascontiguousarray(y_tr)).to(self.device)
        w_tr_t = torch.FloatTensor(np.ascontiguousarray(w_tr)).to(self.device) if w_tr is not None else torch.ones(len(X_tr)).to(self.device)
        
        X_val_t = torch.FloatTensor(np.ascontiguousarray(X_val)).to(self.device)
        y_val_t = torch.FloatTensor(np.ascontiguousarray(y_val)).to(self.device)
        w_val_t = torch.FloatTensor(np.ascontiguousarray(w_val)).to(self.device) if w_val is not None else torch.ones(len(X_val)).to(self.device)
            
        dataset = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, max(len(X_tr_t), 1)), shuffle=True)
        
        criterion = nn.PoissonNLLLoss(log_input=False, reduction='none')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            for bx, by, bw in loader:
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                bw_unsqueeze = bw.unsqueeze(1)
                loss = (loss * bw_unsqueeze).sum() / (bw_unsqueeze.sum() * 2 + 1e-8)
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = criterion(val_preds, y_val_t)
                w_val_unsq = w_val_t.unsqueeze(1)
                val_loss = ((val_loss * w_val_unsq).sum() / (w_val_unsq.sum() * 2 + 1e-8)).item()
                
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
        X_tensor = torch.FloatTensor(np.ascontiguousarray(X_scaled)).to(self.device)
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
    def __init__(self, mlp_params, ridge_alpha=1.0, meta_alpha=1.0, cv_splits=3):
        self.mlp_params = mlp_params
        self.ridge_alpha = ridge_alpha
        self.meta_alpha = meta_alpha
        self.cv_splits = cv_splits
        
        self.mlp_model = SklearnPyTorchRegressorWrapper(**mlp_params)
        self.ridge_imputer = SimpleImputer(strategy='median')
        self.ridge_scaler = StandardScaler()
        self.ridge_model = MultiOutputRegressor(PoissonRegressor(alpha=ridge_alpha))
        self.meta_model = Ridge(alpha=meta_alpha)
        
    def fit(self, X, y, sample_weight=None):
        X_full_scaled = self.ridge_scaler.fit_transform(self.ridge_imputer.fit_transform(X))
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        
        # Predicciones OOF estrictas para AMBOS modelos base (MLP y Ridge)
        mlp_oof = np.zeros_like(y_arr, dtype=float)
        kf = KFold(n_splits=self.cv_splits, shuffle=False)
        
        for tr_idx, val_idx in kf.split(X):
            X_tr = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
            y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
            X_va = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            w_tr = (sample_weight.iloc[tr_idx] if hasattr(sample_weight, 'iloc') else sample_weight[tr_idx]) if sample_weight is not None else None
            
            fold_mlp = SklearnPyTorchRegressorWrapper(**self.mlp_params)
            fold_mlp.fit(X_tr, y_tr, sample_weight=w_tr)
            mlp_oof[val_idx] = fold_mlp.predict(X_va)
            
        ridge_base = MultiOutputRegressor(PoissonRegressor(alpha=self.ridge_alpha))
        ridge_oof = cross_val_predict(ridge_base, X_full_scaled, y_arr, cv=self.cv_splits)
        
        # Meta-modelo entrenado sobre características OOF limpias
        meta_X = np.column_stack([mlp_oof, ridge_oof])
        self.meta_model.fit(meta_X, y_arr, sample_weight=sample_weight)
        
        # Modelos base finales entrenados en 100% de los datos
        self.mlp_model.fit(X, y, sample_weight=sample_weight)
        self.ridge_model.fit(X_full_scaled, y_arr, sample_weight=sample_weight)
        return self
        
    def predict(self, X):
        mlp_preds = self.mlp_model.predict(X)
        X_scaled = self.ridge_scaler.transform(self.ridge_imputer.transform(X))
        ridge_preds = self.ridge_model.predict(X_scaled)
        
        meta_X = np.column_stack([mlp_preds, ridge_preds])
        final_preds = self.meta_model.predict(meta_X)
        return np.maximum(final_preds, 0.1)

def calc_over_probs_bivariate(lambda_home, lambda_away, lines, var_home=None, var_away=None, cov_home_away=0.0):
    probs = {}
    lambda_total = lambda_home + lambda_away
    
    mean_h = np.mean(lambda_home)
    mean_a = np.mean(lambda_away)
    
    disp_h = (var_home / (mean_h + 1e-8)) if (var_home is not None and mean_h > 0) else 1.0
    disp_a = (var_away / (mean_a + 1e-8)) if (var_away is not None and mean_a > 0) else 1.0
    
    var_h_i = np.maximum(lambda_home * disp_h, lambda_home * 1.01)
    var_a_i = np.maximum(lambda_away * disp_a, lambda_away * 1.01)
    
    var_tot_i = np.maximum(var_h_i + var_a_i + 2.0 * cov_home_away, lambda_total * 1.01)
    
    for line in lines:
        k = int(np.floor(line))
        mean_val = np.maximum(lambda_total, 0.01)
        var_val = np.maximum(var_tot_i, mean_val * 1.01)
        
        p = np.clip(mean_val / var_val, 1e-5, 0.999)
        n = np.maximum((mean_val**2) / (var_val - mean_val), 1e-4)
        prob_over = 1.0 - nbinom.cdf(k, n, p)
        probs[f'prob_over_{line}'] = prob_over
    return probs

def calc_over_probs(lambda_vals, lines, empirical_variance=None):
    probs = {}
    mean_val_overall = np.mean(lambda_vals)
    for line in lines:
        k = int(np.floor(line))
        if empirical_variance is not None and empirical_variance > mean_val_overall:
            mean_val = np.maximum(lambda_vals, 0.01)
            var_val = np.maximum(mean_val * (empirical_variance / (mean_val_overall + 1e-8)), mean_val * 1.01)
            
            p = np.clip(mean_val / var_val, 1e-5, 0.999)
            n = np.maximum((mean_val**2) / (var_val - mean_val), 1e-4)
            prob_over = 1.0 - nbinom.cdf(k, n, p)
        else:
            prob_over = 1.0 - poisson.cdf(k, np.maximum(lambda_vals, 1e-4))
            
        probs[f'prob_over_{line}'] = prob_over
    return probs

def train_shots_on_goal():
    active_device = check_cuda()
    
    df = get_base_dataset()
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATASET_PATH = os.path.join(BASE_DIR, 'data/processed/matches_with_referees.parquet')
    FALLBACK_DATASET = os.path.join(BASE_DIR, 'data/processed/matches_with_odds.parquet')
    path_to_load = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    
    df_all = pd.read_parquet(path_to_load, engine='fastparquet')
    
    # 1. Extraer target del visitante (Away Team) emparejado limpiamente por match_id
    away_data = df_all[df_all['is_home'] == 0][['match_id', 'shots_on_target']].copy()
    away_data = away_data.rename(columns={
        'shots_on_target': 'away_shots_on_target',
    }).drop_duplicates(subset=['match_id'])
    
    df = df.merge(away_data, on='match_id', how='left')
    
    # 2. INGENIERÍA DE CARACTERÍSTICAS DE MERCADO (Cuotas de Apuestas Implícitas)
    logger.info("Calculando probabilidades e indicadores predictivos del mercado de cuotas...")
    if 'odds_win' in df.columns and 'odds_loss' in df.columns:
        inv_win = 1.0 / df['odds_win'].clip(lower=1.01)
        inv_draw = (1.0 / df['odds_draw'].clip(lower=1.01)) if 'odds_draw' in df.columns else 0.0
        inv_loss = 1.0 / df['odds_loss'].clip(lower=1.01)
        margin = inv_win + inv_draw + inv_loss
        
        df['implied_prob_home'] = inv_win / margin
        df['implied_prob_away'] = inv_loss / margin
        df['odds_ratio_home_away'] = np.log(df['odds_win'].clip(lower=1.01) / df['odds_loss'].clip(lower=1.01))
        
        if 'open_odds_win' in df.columns and df['open_odds_win'].notna().any():
            df['market_drift_home'] = (df['odds_win'] / df['open_odds_win'].clip(lower=1.01)).fillna(1.0)
        else:
            df['market_drift_home'] = 1.0
    else:
        df['implied_prob_home'] = 0.5
        df['implied_prob_away'] = 0.5
        df['odds_ratio_home_away'] = 0.0
        df['market_drift_home'] = 1.0
    
    # Eliminar partidos sin métrica de target para no distorsionar la distribución
    valid_target_mask = df['shots_on_target'].notna() & df['away_shots_on_target'].notna()
    df = df[valid_target_mask].reset_index(drop=True)
    
    split_idx = get_train_test_split(df)
    
    if 'shots_on_target' not in df.columns:
        logger.error("No se encontró la columna 'shots_on_target' en el dataset.")
        return

    # === CARGAR xG DEL MODELO QUANT ADVANCED ===
    quant_train_path = os.path.join(PROCESSED_DIR, 'oof_quant_train.parquet')
    quant_test_path = os.path.join(PROCESSED_DIR, 'oof_quant_test.parquet')
    
    if os.path.exists(quant_train_path) and os.path.exists(quant_test_path):
        logger.info("Cargando variables xG del modelo Quant Advanced...")
        q_tr = pd.read_parquet(quant_train_path)
        q_ts = pd.read_parquet(quant_test_path)
        q_full = pd.concat([q_tr, q_ts]).reset_index(drop=True)
        
        if len(q_full) == len(df):
            df['predicted_xg_scored_quant'] = q_full['predicted_xg_scored_quant'].values
            df['predicted_xg_conceded_quant'] = q_full['predicted_xg_conceded_quant'].values
        elif 'match_id' in q_full.columns:
            df = df.merge(q_full[['match_id', 'predicted_xg_scored_quant', 'predicted_xg_conceded_quant']], on='match_id', how='left')
            df['predicted_xg_scored_quant'] = df['predicted_xg_scored_quant'].fillna(0.0)
            df['predicted_xg_conceded_quant'] = df['predicted_xg_conceded_quant'].fillna(0.0)
        else:
            logger.warning("No se pudo alinear oof_quant por diferencia de longitud. Usando fallback 0.0.")
            df['predicted_xg_scored_quant'] = 0.0
            df['predicted_xg_conceded_quant'] = 0.0
    else:
        logger.warning("No se encontraron predicciones Quant. Ejecuta train_quant_advanced.py primero.")
        df['predicted_xg_scored_quant'] = 0.0
        df['predicted_xg_conceded_quant'] = 0.0
    
    # Features Multi-Target Clave (incluyendo cuotas de mercado)
    feature_cols = [
        'shots_total_ema3', 'shots_total_ema5',
        'pass_accuracy_ema3', 'pass_accuracy_ema5',
        'team_squad_value', 'opp_squad_value',
        'predicted_xg_scored_quant', 'predicted_xg_conceded_quant',
        'rest_days',
        'possession_pct_ema3', 'possession_pct_ema5',
        'opp_shots_total_ema3', 'opp_shots_total_ema5',
        'opp_pass_accuracy_ema3', 'opp_pass_accuracy_ema5',
        'opp_possession_pct_ema3', 'opp_possession_pct_ema5',
        'implied_prob_home', 'implied_prob_away',
        'odds_ratio_home_away', 'market_drift_home'
    ]
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Faltan variables en Tiros al Arco: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[['shots_on_target', 'away_shots_on_target']].copy() # Target 2D (Multi-Output)
    
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])
        
    logger.info("=== ENTRENANDO MODELO MULTI-OUTPUT DE TIROS AL ARCO (STACKING MLP+RIDGE) ===")
    
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
            'epochs': 50, 
            'batch_size': 256,
            'patience': 4,
            'device': active_device
        }
        
        stack = StackingRegressorEnsemble(mlp_params, ridge_alpha, meta_alpha, cv_splits=3)
        stack.fit(X_opt_train, y_opt_train, sample_weight=w_opt_train)
        preds = stack.predict(X_opt_val)
        
        # Poisson Negative Log-Likelihood para alinear Optuna con la distribución de conteo
        y_val_arr = y_opt_val.values if hasattr(y_opt_val, 'values') else np.asarray(y_opt_val)
        preds_clipped = np.clip(preds, 1e-4, None)
        poisson_nll = np.mean(preds_clipped - y_val_arr * np.log(preds_clipped))
        return poisson_nll

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
        'epochs': 100,
        'batch_size': 256,
        'patience': 7,
        'device': active_device
    }
    
    # OUT OF FOLD PREDICTIONS TEMPORALES (TimeSeriesSplit puro sin shuffle KFold)
    tscv = get_cv_strategy(n_splits=5)
    preds_train = np.zeros((len(X_train), 2))
    preds_train[:] = np.nan
    
    splits = list(tscv.split(X_train))
    
    # Para el primer bloque inicial de entrenamiento, generar predicciones mediante Expanding Split interno sin Data Leakage
    first_train_idx = splits[0][0]
    inner_tscv = TimeSeriesSplit(n_splits=3)
    logger.info(f"  -> Fold Inicial TimeSeriesSplit ({len(first_train_idx)} muestras)...")
    for tr_in, val_in in inner_tscv.split(X_train.iloc[first_train_idx]):
        X_tr_in = X_train.iloc[first_train_idx].iloc[tr_in]
        y_tr_in = y_train.iloc[first_train_idx].iloc[tr_in]
        X_va_in = X_train.iloc[first_train_idx].iloc[val_in]
        
        dates_in_tr = train_dates.iloc[first_train_idx].iloc[tr_in] if train_dates is not None else None
        w_tr = get_time_weights(dates_in_tr) if dates_in_tr is not None else None
        
        stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'], cv_splits=3)
        stack.fit(X_tr_in, y_tr_in, sample_weight=w_tr)
        
        val_idx_orig = first_train_idx[val_in]
        preds_train[val_idx_orig] = stack.predict(X_va_in)

    # Imputar la porción no cubierta del primer split con la media expandida
    first_unpredicted = first_train_idx[np.isnan(preds_train[first_train_idx, 0])]
    if len(first_unpredicted) > 0:
        preds_train[first_unpredicted] = y_train.iloc[first_train_idx].mean().values

    # Expanding Windows CV
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Fold Temporal {i+1}/{len(splits)}...")
        w_tr = get_time_weights(train_dates.iloc[train_idx]) if train_dates is not None else None
        
        stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'], cv_splits=3)
        stack.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], sample_weight=w_tr)
        preds_train[val_idx] = stack.predict(X_train.iloc[val_idx])
        
    logger.info("Entrenando Modelo Final Stacking en todo Train...")
    final_weights = get_time_weights(train_dates)
    final_stack = StackingRegressorEnsemble(best_mlp, best_params['ridge_alpha'], best_params['meta_alpha'], cv_splits=3)
    final_stack.fit(X_train, y_train, sample_weight=final_weights)
    preds_test = final_stack.predict(X_test)
    
    # --- CONVOLUCIÓN BIVARIADA POISSON / NEGATIVE BINOMIAL Y LÍNEAS DE APUESTAS ---
    logger.info("=== CALCULANDO LÍNEAS BIVARIADAS NB Y MÉTRICAS DE APUESTAS ===")
    
    df_train_tmp = df.iloc[:split_idx].copy()
    df_train_tmp['pred_shots'] = preds_train[:, 0]
    df_train_tmp['opp_pred_shots'] = preds_train[:, 1]
    
    df_test_tmp = df.iloc[split_idx:].copy()
    df_test_tmp['pred_shots'] = preds_test[:, 0]
    df_test_tmp['opp_pred_shots'] = preds_test[:, 1]
    
    # Total Shots
    df_train_tmp['lambda_total'] = df_train_tmp['pred_shots'] + df_train_tmp['opp_pred_shots']
    df_test_tmp['lambda_total'] = df_test_tmp['pred_shots'] + df_test_tmp['opp_pred_shots']
    
    df_train_tmp['opp_shots'] = df_train_tmp['away_shots_on_target']
    df_train_tmp['true_total_shots'] = df_train_tmp['shots_on_target'] + df_train_tmp['opp_shots']
    
    df_test_tmp['opp_shots'] = df_test_tmp['away_shots_on_target']
    df_test_tmp['true_total_shots'] = df_test_tmp['shots_on_target'] + df_test_tmp['opp_shots']
    
    # Definición de líneas
    team_lines = [3.5, 4.5, 5.5]
    total_lines = [7.5, 8.5, 9.5]
    
    var_home = np.var(df_train_tmp['shots_on_target'])
    var_away = np.var(df_train_tmp['opp_shots'])
    cov_home_away = np.cov(df_train_tmp['shots_on_target'], df_train_tmp['opp_shots'])[0, 1]
    
    logger.info(f"Estadísticas de Entrenamiento -> Var(Home): {var_home:.3f} | Var(Away): {var_away:.3f} | Cov(Home,Away): {cov_home_away:.3f}")
    
    train_team_probs = calc_over_probs(df_train_tmp['pred_shots'].values, team_lines, var_home)
    test_team_probs = calc_over_probs(df_test_tmp['pred_shots'].values, team_lines, var_home)
    for col, vals in train_team_probs.items(): df_train_tmp[col + '_team'] = vals
    for col, vals in test_team_probs.items(): df_test_tmp[col + '_team'] = vals
        
    train_total_probs = calc_over_probs_bivariate(df_train_tmp['pred_shots'].values, df_train_tmp['opp_pred_shots'].values, total_lines, var_home, var_away, cov_home_away)
    test_total_probs = calc_over_probs_bivariate(df_test_tmp['pred_shots'].values, df_test_tmp['opp_pred_shots'].values, total_lines, var_home, var_away, cov_home_away)
    for col, vals in train_total_probs.items(): df_train_tmp[col + '_total'] = vals
    for col, vals in test_total_probs.items(): df_test_tmp[col + '_total'] = vals
        
    # Calibración Isotónica sobre OOF
    calibrators = {}
    for line in team_lines:
        col = f'prob_over_{line}_team'
        true_over_train = (df_train_tmp['shots_on_target'] > line).astype(int).values
        ir = IsotonicRegression(out_of_bounds='clip')
        df_train_tmp[col] = ir.fit_transform(df_train_tmp[col].values, true_over_train)
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir
        
    for line in total_lines:
        col = f'prob_over_{line}_total'
        true_over_train = (df_train_tmp['true_total_shots'] > line).astype(int).values
        ir = IsotonicRegression(out_of_bounds='clip')
        df_train_tmp[col] = ir.fit_transform(df_train_tmp[col].values, true_over_train)
        df_test_tmp[col] = ir.predict(df_test_tmp[col].values)
        calibrators[col] = ir

    # AUDITORÍA CONTINUA
    logger.info("--- REGRESIÓN PURA (MSE/MAE) - TOTAL MATCH SHOTS ---")
    mse_tr = mean_squared_error(df_train_tmp['true_total_shots'], df_train_tmp['lambda_total'])
    mae_tr = mean_absolute_error(df_train_tmp['true_total_shots'], df_train_tmp['lambda_total'])
    mse_ts = mean_squared_error(df_test_tmp['true_total_shots'], df_test_tmp['lambda_total'])
    mae_ts = mean_absolute_error(df_test_tmp['true_total_shots'], df_test_tmp['lambda_total'])
    logger.info(f"Train - MSE: {mse_tr:.3f} | MAE: {mae_tr:.3f}")
    logger.info(f"Test  - MSE: {mse_ts:.3f} | MAE: {mae_ts:.3f}")

    logger.info("--- PROBABILIDADES TOTAL MATCH BIVARIADAS (Over/Under) ---")
    for line in total_lines:
        true_over_ts = (df_test_tmp['true_total_shots'] > line).astype(int)
        prob_ts = df_test_tmp[f'prob_over_{line}_total']
        ll_ts = log_loss(true_over_ts, prob_ts)
        bs_ts = brier_score_loss(true_over_ts, prob_ts)
        acc_ts = accuracy_score(true_over_ts, (prob_ts > 0.5).astype(int))
        logger.info(f"OVER {line} Total | Dist: {true_over_ts.mean()*100:.1f}% | LogLoss: {ll_ts:.4f} | Brier: {bs_ts:.4f} | Acc: {acc_ts:.4f}")

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
