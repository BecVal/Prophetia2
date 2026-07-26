import os
import json
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import optuna

# Fijar semilla global para reproducibilidad estricta
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Asegurar import de data_splitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.logger_config import get_logger

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA Y DEEP ENSEMBLE)
# ==============================================================================
RUN_OPTUNA = True
OPTUNA_TRIALS = 30
ENSEMBLE_SEEDS = [42, 100, 2024]  # Bagging de 3 semillas por pliegue
# ==============================================================================

OPTUNA_PARAMS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/models_best_parameters/optuna_params_nn.json'))
os.makedirs(os.path.dirname(OPTUNA_PARAMS_FILE), exist_ok=True)
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


def compute_market_logits(df_subset):
    """Calcula los log-odds implícitos de mercado a partir de las cuotas de cierre."""
    cols = df_subset.columns if hasattr(df_subset, 'columns') else []
    has_win = 'odds_win' in cols
    has_draw = 'odds_draw' in cols
    has_loss = 'odds_loss' in cols

    if has_win and has_draw and has_loss:
        ow = df_subset['odds_win'].values
        od = df_subset['odds_draw'].values
        ol = df_subset['odds_loss'].values
        valid = np.isfinite(ow) & np.isfinite(od) & np.isfinite(ol) & (ow > 1.0) & (od > 1.0) & (ol > 1.0)
    else:
        valid = np.zeros(len(df_subset), dtype=bool)

    p_loss_raw = np.where(valid, 1.0 / np.maximum(df_subset['odds_loss'].values if valid.any() else 1.0, 1.01), 0.297)
    p_draw_raw = np.where(valid, 1.0 / np.maximum(df_subset['odds_draw'].values if valid.any() else 1.0, 1.01), 0.260)
    p_win_raw  = np.where(valid, 1.0 / np.maximum(df_subset['odds_win'].values if valid.any() else 1.0, 1.01), 0.443)

    total_p = p_loss_raw + p_draw_raw + p_win_raw
    total_p = np.maximum(total_p, 1e-6)

    p_loss = p_loss_raw / total_p
    p_draw = p_draw_raw / total_p
    p_win  = p_win_raw / total_p

    p_matrix = np.column_stack([p_loss, p_draw, p_win])
    p_matrix = np.clip(p_matrix, 1e-4, 1.0 - 1e-4)
    logits = np.log(p_matrix)
    return logits.astype(np.float32)


class GatedContext(nn.Module):
    """Mecanismo de Selección Adaptativa de Características (Context Gating)"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """Bloque Residual para Datos Tabulares con LayerNorm"""
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class PyTorchGatedResNet10(nn.Module):
    """Arquitectura 10/10: Entity Embeddings + Gated Context + Residual Logits + Market Prior Logits"""
    def __init__(self, num_teams, embed_dim=16, continuous_dim=64, hidden_dim=128, num_blocks=2, dropout_rate=0.2, num_classes=3):
        super().__init__()
        self.team_embed = nn.Embedding(num_teams + 2, embed_dim)
        total_in_dim = continuous_dim + embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.gating = GatedContext(hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_cont, x_team, market_logits):
        team_vec = self.team_embed(x_team)
        x = torch.cat([x_cont, team_vec], dim=1)
        x = self.input_layer(x)
        x = self.gating(x)
        for block in self.blocks:
            x = block(x)
        residual_logits = self.head(x)
        final_logits = residual_logits + market_logits
        return final_logits


class SklearnPyTorchWrapper10(BaseEstimator, ClassifierMixin):
    def __init__(self, num_teams, continuous_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.2, weight_decay=1e-3, num_classes=3, epochs=80, batch_size=256, lr=1e-3, device='cuda', patience=7, seeds=[42, 100, 2024]):
        self.num_teams = num_teams
        self.continuous_dim = continuous_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seeds = seeds
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)

        # Extraer columna de equipos y variables continuas excluyendo odds e idx
        team_indices = X['team_idx'].values.astype(int) if 'team_idx' in X.columns else np.zeros(len(X), dtype=int)
        mkt_logits_arr = compute_market_logits(X)

        exclude_cols = ['team_idx', 'odds_win', 'odds_draw', 'odds_loss']
        cont_cols = [c for c in X.columns if c not in exclude_cols]
        X_cont = X[cont_cols].values if hasattr(X, 'columns') else np.asarray(X)
        self.continuous_dim = X_cont.shape[1]

        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
        w_arr = sample_weight.values if hasattr(sample_weight, 'values') else (np.asarray(sample_weight) if sample_weight is not None else None)

        val_idx = int(len(X_cont) * 0.85)

        X_tr_raw, X_val_raw = X_cont[:val_idx], X_cont[val_idx:]
        team_tr, team_val = team_indices[:val_idx], team_indices[val_idx:]
        mkt_tr, mkt_val = mkt_logits_arr[:val_idx], mkt_logits_arr[val_idx:]

        X_tr_imp = self.imputer.fit_transform(X_tr_raw)
        X_val_imp = self.imputer.transform(X_val_raw)

        X_tr = self.scaler.fit_transform(X_tr_imp)
        X_val = self.scaler.transform(X_val_imp)

        y_tr, y_val = y_arr[:val_idx], y_arr[val_idx:]
        w_tr = w_arr[:val_idx] if w_arr is not None else None
        w_val = w_arr[val_idx:] if w_arr is not None else None

        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        team_tr_t = torch.LongTensor(np.copy(team_tr)).to(self.device)
        mkt_tr_t = torch.FloatTensor(mkt_tr).to(self.device)
        y_tr_t = torch.LongTensor(np.copy(y_tr)).to(self.device)
        w_tr_t = torch.FloatTensor(w_tr).to(self.device) if w_tr is not None else torch.ones(len(X_tr)).to(self.device)

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        team_val_t = torch.LongTensor(np.copy(team_val)).to(self.device)
        mkt_val_t = torch.FloatTensor(mkt_val).to(self.device)
        y_val_t = torch.LongTensor(np.copy(y_val)).to(self.device)
        w_val_t = torch.FloatTensor(w_val).to(self.device) if w_val is not None else torch.ones(len(X_val)).to(self.device)

        dataset = TensorDataset(X_tr_t, team_tr_t, mkt_tr_t, y_tr_t, w_tr_t)
        drop_last = len(dataset) > self.batch_size

        self.models = []
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Deep Ensemble across multiple seeds
        for s in self.seeds:
            set_seed(s)
            model = PyTorchGatedResNet10(
                num_teams=self.num_teams,
                embed_dim=16,
                continuous_dim=self.continuous_dim,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes
            ).to(self.device)

            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(self.epochs):
                model.train()
                for bx, bt, bm, by, bw in loader:
                    optimizer.zero_grad()
                    logits = model(bx, bt, bm)
                    loss = criterion(logits, by)
                    denom = bw.sum()
                    loss = (loss * bw).sum() / (denom + 1e-8)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_t, team_val_t, mkt_val_t)
                    val_loss = criterion(val_logits, y_val_t)
                    val_denom = w_val_t.sum()
                    val_loss = ((val_loss * w_val_t).sum() / (val_denom + 1e-8)).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            self.models.append(model)

        # Refit scaler on full data
        X_full_imp = self.imputer.fit_transform(X_cont)
        self.scaler.fit(X_full_imp)

        return self

    def predict_proba(self, X):
        team_indices = X['team_idx'].values.astype(int) if 'team_idx' in X.columns else np.zeros(len(X), dtype=int)
        mkt_logits_arr = compute_market_logits(X)

        exclude_cols = ['team_idx', 'odds_win', 'odds_draw', 'odds_loss']
        cont_cols = [c for c in X.columns if c not in exclude_cols]
        X_cont = X[cont_cols].values if hasattr(X, 'columns') else np.asarray(X)

        X_imp = self.imputer.transform(X_cont)
        X_scaled = self.scaler.transform(X_imp)

        X_t = torch.FloatTensor(X_scaled).to(self.device)
        team_t = torch.LongTensor(np.copy(team_indices)).to(self.device)
        mkt_t = torch.FloatTensor(mkt_logits_arr).to(self.device)

        ensemble_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X_t, team_t, mkt_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                ensemble_probs.append(probs)

        avg_probs = np.mean(ensemble_probs, axis=0)
        return avg_probs

    def get_params(self, deep=True):
        return {
            'num_teams': self.num_teams,
            'continuous_dim': self.continuous_dim,
            'hidden_dim': self.hidden_dim,
            'num_blocks': self.num_blocks,
            'dropout_rate': self.dropout_rate,
            'weight_decay': self.weight_decay,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'device': self.device,
            'patience': self.patience,
            'seeds': self.seeds
        }


def train_nn():
    set_seed(42)
    df = get_base_dataset()

    # Mapeo de Entity Embeddings para Equipos
    teams = df['team'].astype(str).unique() if 'team' in df.columns else []
    team2idx = {t: i + 1 for i, t in enumerate(teams)}
    df['team_idx'] = df['team'].astype(str).map(team2idx).fillna(0).astype(int) if 'team' in df.columns else 0
    num_teams = len(teams)

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

    # Incluir odds si existen en df para pasar al wrapper
    cols_to_extract = feature_cols + ['team_idx']
    for odd_col in ['odds_win', 'odds_draw', 'odds_loss']:
        if odd_col in df.columns:
            cols_to_extract.append(odd_col)

    X = df[cols_to_extract].copy()
    y = df['outcome'].replace({-1: 0, 0: 1, 1: 2})

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]

    train_dates = None
    if 'match_date' in df.columns:
        train_dates = pd.to_datetime(df['match_date'].iloc[:split_idx])

    cv_strategy = get_cv_strategy(n_splits=5)

    logger.info("Configurando Red Neuronal 10/10 (Entity Embeddings + Gated ResNet + Market Prior Logits + Deep Ensemble)...")

    continuous_dim = len([c for c in feature_cols if c != 'team_idx'])

    # === OPTUNA HYPERPARAMETER TUNING ===
    def objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1)
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        kf = TimeSeriesSplit(n_splits=3)
        fold_losses = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
            w_tr = get_time_weights(dates_tr)

            estimator = SklearnPyTorchWrapper10(
                num_teams=num_teams,
                continuous_dim=continuous_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                epochs=40,
                batch_size=batch_size,
                lr=lr,
                patience=5,
                seeds=[42] # Usar 1 semilla en Optuna para velocidad
            )
            estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
            preds = estimator.predict_proba(X_val)
            fold_loss = log_loss(y_val, preds)
            fold_losses.append(fold_loss)

        return np.mean(fold_losses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if RUN_OPTUNA:
        logger.info(f"Iniciando Optuna con {OPTUNA_TRIALS} trials (Model 10/10)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
        best_params = study.best_params
        logger.info(f"Mejores hiperparámetros encontrados por Optuna: {best_params}")
        logger.info(f"Mejor Log-Loss en validación (3 folds): {study.best_value:.4f}")
        with open(OPTUNA_PARAMS_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        logger.info("Cargando mejores parámetros de Optuna guardados...")
        if os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE, 'r') as f:
                best_params = json.load(f)
            logger.info(f"Mejores hiperparámetros cargados: {best_params}")
        else:
            logger.warning(f"Archivo de parámetros {OPTUNA_PARAMS_FILE} no encontrado. Ejecutando Optuna como fallback.")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
            best_params = study.best_params
            logger.info(f"Mejores hiperparámetros encontrados por Optuna: {best_params}")
            logger.info(f"Mejor Log-Loss en validación (3 folds): {study.best_value:.4f}")
            with open(OPTUNA_PARAMS_FILE, 'w') as f:
                json.dump(best_params, f, indent=4)

    best_hidden_dim = best_params['hidden_dim']
    best_num_blocks = best_params['num_blocks']
    best_dropout = best_params['dropout_rate']
    best_lr = best_params['lr']
    best_wd = best_params['weight_decay']
    best_bs = best_params['batch_size']

    nn_best = SklearnPyTorchWrapper10(
        num_teams=num_teams,
        continuous_dim=continuous_dim,
        hidden_dim=best_hidden_dim,
        num_blocks=best_num_blocks,
        dropout_rate=best_dropout,
        weight_decay=best_wd,
        epochs=80,
        batch_size=best_bs,
        lr=best_lr,
        patience=7,
        seeds=ENSEMBLE_SEEDS
    )

    logger.info("Calculando predicciones OOF para Train (NN 10/10) con Deep Ensemble...")
    pred_probs_train = np.zeros((len(X_train), 3))
    pred_probs_train[:] = np.nan

    splits = list(cv_strategy.split(X_train, y_train))

    # 1. Resolver OOF del Fold Inicial SIN Data Leakage
    first_train_idx = splits[0][0]
    X_first = X_train.iloc[first_train_idx]
    y_first = y_train.iloc[first_train_idx]
    dates_first = train_dates.iloc[first_train_idx] if train_dates is not None else None

    logger.info(f"  -> Procesando Primer Fold Inicial ({len(first_train_idx)} muestras) con TimeSeriesSplit(5)...")
    ts_init = TimeSeriesSplit(n_splits=5)

    prior_probs = y_first.value_counts(normalize=True).sort_index().values
    if len(prior_probs) == 3:
        for idx_pos in range(len(first_train_idx)):
            pred_probs_train[first_train_idx[idx_pos]] = prior_probs

    for ts_tr, ts_val in ts_init.split(X_first):
        X_ts_tr, y_ts_tr = X_first.iloc[ts_tr], y_first.iloc[ts_tr]
        X_ts_val = X_first.iloc[ts_val]

        dates_ts_tr = dates_first.iloc[ts_tr] if dates_first is not None else None
        w_tr = get_time_weights(dates_ts_tr)

        ts_estimator_base = SklearnPyTorchWrapper10(**nn_best.get_params())
        ts_estimator = CalibratedClassifierCV(estimator=ts_estimator_base, method='sigmoid', cv=3)
        ts_estimator.fit(X_ts_tr, y_ts_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)

        val_indices_in_original = first_train_idx[ts_val]
        pred_probs_train[val_indices_in_original] = ts_estimator.predict_proba(X_ts_val)

    # 2. Expanding Windows estándar con Calibración Sigmoide y Deep Ensemble
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  -> Procesando Fold Temporal {i+1}/{len(splits)} (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        dates_tr = train_dates.iloc[train_idx] if train_dates is not None else None
        w_tr = get_time_weights(dates_tr)

        fold_estimator_base = SklearnPyTorchWrapper10(**nn_best.get_params())
        fold_estimator = CalibratedClassifierCV(estimator=fold_estimator_base, method='sigmoid', cv=3)
        fold_estimator.fit(X_tr, y_tr, sample_weight=w_tr.values if isinstance(w_tr, pd.Series) else w_tr)
        pred_probs_train[val_idx] = fold_estimator.predict_proba(X_val)

    logger.info("Entrenando Modelo NN 10/10 final y prediciendo Test con Deep Ensemble...")
    final_w_tr = get_time_weights(train_dates)
    base_final = SklearnPyTorchWrapper10(**nn_best.get_params())
    final_estimator = CalibratedClassifierCV(estimator=base_final, method='sigmoid', cv=3)
    final_estimator.fit(X_train, y_train, sample_weight=final_w_tr.values if isinstance(final_w_tr, pd.Series) else final_w_tr)
    pred_probs_test = final_estimator.predict_proba(X_test)

    # LOGS: Verificación de calibración y métricas
    logger.info("=== ESTADÍSTICAS Y AUDITORÍA DEL MODELO NN 10/10 ===")

    valid_idx = ~np.isnan(pred_probs_train[:, 0])
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

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    oof_train = pd.DataFrame(pred_probs_train, columns=['prob_loss_nn', 'prob_draw_nn', 'prob_win_nn'], index=X_train.index)
    oof_test = pd.DataFrame(pred_probs_test, columns=['prob_loss_nn', 'prob_draw_nn', 'prob_win_nn'], index=X_test.index)

    oof_train.to_parquet(os.path.join(PROCESSED_DIR, 'oof_nn_train.parquet'), engine='fastparquet')
    oof_test.to_parquet(os.path.join(PROCESSED_DIR, 'oof_nn_test.parquet'), engine='fastparquet')

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    joblib.dump({'model': final_estimator, 'features': feature_cols, 'team2idx': team2idx}, MODEL_SAVE_PATH)
    logger.info(f"=== MODELO NN 10/10 FINALIZADO === Guardado en {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_nn()

