import os
import sys
import pandas as pd
import numpy as np
import optuna
import joblib
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from core.logger_config import get_logger
from core.models.data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logger = get_logger(__name__, 'train_clv_model')

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SAVE_MODELS_DIR = os.path.join(BASE_DIR, 'core', 'save_models')

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, 'X_train.parquet')
X_TEST_PATH = os.path.join(PROCESSED_DIR, 'X_test.parquet')
TRAIN_PREDS_PATH = os.path.join(PROCESSED_DIR, 'train_predictions.parquet')
PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, 'test_predictions.parquet')

TAX_RETENTION_RATE = 0.0075

def get_time_weights(dates, half_life_days=365):
    """ Calcula pesos de vida media exponencial basados en fechas reales. """
    if dates is None or pd.isna(dates).all():
        return None
    dates_dt = pd.to_datetime(dates)
    max_date = dates_dt.max()
    days_diff = (max_date - dates_dt).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_diff / half_life_days).values

def train_clv_model():
    if not all(os.path.exists(p) for p in [X_TRAIN_PATH, X_TEST_PATH, PREDICTIONS_PATH, TRAIN_PREDS_PATH]):
        logger.error("Faltan archivos parquet generados por el stacker. Ejecuta train_stacker.py primero.")
        return

    logger.info("Cargando dataset base para alinear índices...")
    df = get_base_dataset()
    split_idx = get_train_test_split(df)
    
    # Codificación de categorías unificada antes del split para evitar mismatch entre train y test
    comp_type = None
    if 'competition' in df.columns:
        comp_type = pd.CategoricalDtype(categories=df['competition'].unique(), ordered=False)

    # Target log-drift simétrico: log(open_odds / closing_odds)
    # Evita el sesgo asimétrico y estabiliza la varianza residual para la función de pérdida RMSE de XGBoost.
    open_loss_df = df['open_odds_loss'].fillna(df['odds_loss'])
    open_draw_df = df['open_odds_draw'].fillna(df['odds_draw'])
    open_win_df = df['open_odds_win'].fillna(df['odds_win'])

    target_loss_log = np.log(np.clip(open_loss_df / df['odds_loss'], 1e-4, 1e4)).fillna(0)
    target_draw_log = np.log(np.clip(open_draw_df / df['odds_draw'], 1e-4, 1e4)).fillna(0)
    target_win_log = np.log(np.clip(open_win_df / df['odds_win'], 1e-4, 1e4)).fillna(0)
    
    # Guardar también target lineal para métricas financieras reales
    target_loss_lin = (open_loss_df / df['odds_loss'] - 1).fillna(0)
    target_draw_lin = (open_draw_df / df['odds_draw'] - 1).fillna(0)
    target_win_lin = (open_win_df / df['odds_win'] - 1).fillna(0)

    y_drift_loss_train = target_loss_log.iloc[:split_idx]
    y_drift_draw_train = target_draw_log.iloc[:split_idx]
    y_drift_win_train = target_win_log.iloc[:split_idx]
    
    y_drift_loss_test_log = target_loss_log.iloc[split_idx:]
    y_drift_draw_test_log = target_draw_log.iloc[split_idx:]
    y_drift_win_test_log = target_win_log.iloc[split_idx:]
    
    y_drift_loss_test_lin = target_loss_lin.iloc[split_idx:]
    y_drift_draw_test_lin = target_draw_lin.iloc[split_idx:]
    y_drift_win_test_lin = target_win_lin.iloc[split_idx:]

    logger.info("Cargando predicciones del Stacker (Nivel 2) y features...")
    X_train_full = pd.read_parquet(X_TRAIN_PATH, engine='fastparquet')
    X_test_full = pd.read_parquet(X_TEST_PATH, engine='fastparquet')
    
    df_train_preds = pd.read_parquet(TRAIN_PREDS_PATH, engine='fastparquet')
    df_test_preds = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    
    df_train_orig = df.iloc[:split_idx].reset_index(drop=True)
    df_test_orig = df.iloc[split_idx:].reset_index(drop=True)
    
    dates_train = df_train_orig['match_date'] if 'match_date' in df_train_orig.columns else None
    
    def prepare_clv_features(X_stacker, df_preds, df_orig):
        X_clv = pd.DataFrame(index=X_stacker.index)
        
        # 1. Probabilidades del ensamblado final Nivel 2
        X_clv['stacker_prob_loss'] = df_preds['prob_loss'].values
        X_clv['stacker_prob_draw'] = df_preds['prob_draw'].values
        X_clv['stacker_prob_win'] = df_preds['prob_win'].values
        
        # Probabilidades fundamentales (Nivel 1) si existen
        if 'fund_prob_loss' in X_stacker.columns:
            X_clv['fund_prob_loss'] = X_stacker['fund_prob_loss'].values
            X_clv['fund_prob_draw'] = X_stacker['fund_prob_draw'].values
            X_clv['fund_prob_win'] = X_stacker['fund_prob_win'].values
            
        # 2. Cuotas y probabilidades implícitas de apertura del mercado (No-Vig)
        open_loss_col = df_orig['open_odds_loss'] if 'open_odds_loss' in df_orig.columns else df_orig['odds_loss']
        open_draw_col = df_orig['open_odds_draw'] if 'open_odds_draw' in df_orig.columns else df_orig['odds_draw']
        open_win_col = df_orig['open_odds_win'] if 'open_odds_win' in df_orig.columns else df_orig['odds_win']

        open_loss = open_loss_col.fillna(df_orig['odds_loss']).clip(lower=1.01).values
        open_draw = open_draw_col.fillna(df_orig['odds_draw']).clip(lower=1.01).values
        open_win = open_win_col.fillna(df_orig['odds_win']).clip(lower=1.01).values
        
        implied_loss = 1.0 / open_loss
        implied_draw = 1.0 / open_draw
        implied_win = 1.0 / open_win
        vig_open = implied_loss + implied_draw + implied_win
        
        open_fair_loss = implied_loss / vig_open
        open_fair_draw = implied_draw / vig_open
        open_fair_win = implied_win / vig_open
        
        # 3. Features de Sesgo Favorito-Underdog (Cuotas Absolutas y Términos Polinomiales)
        X_clv['log_open_win'] = np.log(open_win)
        X_clv['log_open_draw'] = np.log(open_draw)
        X_clv['log_open_loss'] = np.log(open_loss)
        
        X_clv['log_open_win_sq'] = X_clv['log_open_win'] ** 2
        X_clv['log_open_loss_sq'] = X_clv['log_open_loss'] ** 2

        # 4. Ratios Implícitos del Mercado
        X_clv['implied_ratio_win_loss'] = open_fair_win / (open_fair_loss + 1e-6)
        X_clv['implied_ratio_win_draw'] = open_fair_win / (open_fair_draw + 1e-6)

        # 5. Features Matemáticos: Edge Lineal, Edge Logarítmico y EV de Apertura
        X_clv['edge_diff_loss'] = X_clv['stacker_prob_loss'] - open_fair_loss
        X_clv['edge_diff_draw'] = X_clv['stacker_prob_draw'] - open_fair_draw
        X_clv['edge_diff_win'] = X_clv['stacker_prob_win'] - open_fair_win
        
        X_clv['edge_log_loss'] = np.log(np.clip(X_clv['stacker_prob_loss'] / open_fair_loss, 1e-6, 1e6))
        X_clv['edge_log_draw'] = np.log(np.clip(X_clv['stacker_prob_draw'] / open_fair_draw, 1e-6, 1e6))
        X_clv['edge_log_win'] = np.log(np.clip(X_clv['stacker_prob_win'] / open_fair_win, 1e-6, 1e6))
        
        # Ratios de Edge Cruzados
        X_clv['edge_win_vs_loss'] = X_clv['edge_diff_win'] - X_clv['edge_diff_loss']
        X_clv['edge_win_vs_draw'] = X_clv['edge_diff_win'] - X_clv['edge_diff_draw']

        # Entropía de Shannon (Mercado vs Stacker)
        p_mkt = np.column_stack([open_fair_loss, open_fair_draw, open_fair_win])
        p_stk = np.column_stack([X_clv['stacker_prob_loss'], X_clv['stacker_prob_draw'], X_clv['stacker_prob_win']])
        
        X_clv['market_entropy'] = -np.sum(p_mkt * np.log(np.clip(p_mkt, 1e-6, 1.0)), axis=1)
        X_clv['stacker_entropy'] = -np.sum(p_stk * np.log(np.clip(p_stk, 1e-6, 1.0)), axis=1)
        X_clv['entropy_diff'] = X_clv['stacker_entropy'] - X_clv['market_entropy']

        # EV Esperado a la Apertura
        net_odds_win = 1 + (open_win - 1) * (1 - TAX_RETENTION_RATE)
        net_odds_draw = 1 + (open_draw - 1) * (1 - TAX_RETENTION_RATE)
        net_odds_loss = 1 + (open_loss - 1) * (1 - TAX_RETENTION_RATE)
        
        X_clv['ev_open_win'] = (X_clv['stacker_prob_win'] * net_odds_win) - 1
        X_clv['ev_open_draw'] = (X_clv['stacker_prob_draw'] * net_odds_draw) - 1
        X_clv['ev_open_loss'] = (X_clv['stacker_prob_loss'] * net_odds_loss) - 1

        # EV Dominance & EV / Vig Ratio
        X_clv['ev_dominance_win'] = X_clv['ev_open_win'] - np.maximum(X_clv['ev_open_draw'], X_clv['ev_open_loss'])
        X_clv['ev_dominance_loss'] = X_clv['ev_open_loss'] - np.maximum(X_clv['ev_open_win'], X_clv['ev_open_draw'])
        X_clv['ev_over_vig_win'] = X_clv['ev_open_win'] / (vig_open - 1 + 1e-4)
        X_clv['ev_over_vig_loss'] = X_clv['ev_open_loss'] / (vig_open - 1 + 1e-4)

        X_clv['vig_open'] = vig_open - 1
        X_clv['open_implied_loss'] = implied_loss
        X_clv['open_implied_draw'] = implied_draw
        X_clv['open_implied_win'] = implied_win
        
        # Categorización unificada de liga
        if comp_type is not None and 'competition' in df_orig.columns:
            X_clv['competition_id'] = df_orig['competition'].astype(comp_type)
        elif 'meta_competition_id' in X_stacker.columns:
            X_clv['competition_id'] = X_stacker['meta_competition_id'].astype('category')
            
        return X_clv
        
    logger.info("Construyendo matriz de características para CLV 10/10 basada en Nivel 2...")
    X_train = prepare_clv_features(X_train_full, df_train_preds, df_train_orig)
    X_test = prepare_clv_features(X_test_full, df_test_preds, df_test_orig)
    
    if X_train is None:
        return
        
    w_train_all = get_time_weights(dates_train)
    
    def optimize_xgb(X, y, dates):
        cv_strategy = get_cv_strategy(n_splits=5)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 350),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'device': 'cuda',
                'enable_categorical': True,
                'tree_method': 'hist',
                'objective': 'reg:squarederror'
            }
            
            rmse_scores = []
            for train_idx, val_idx in cv_strategy.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                dates_tr = dates.iloc[train_idx] if dates is not None else None
                weights_tr = get_time_weights(dates_tr)
                
                model = XGBRegressor(**params)
                if weights_tr is not None:
                    model.fit(X_tr, y_tr, sample_weight=weights_tr)
                else:
                    model.fit(X_tr, y_tr)
                
                preds = model.predict(X_val)
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
                
            return np.mean(rmse_scores)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['device'] = 'cuda'
        best_params['enable_categorical'] = True
        best_params['tree_method'] = 'hist'
        best_params['objective'] = 'reg:squarederror'
        
        final_model = XGBRegressor(**best_params, eval_metric='rmse')
        if w_train_all is not None:
            final_model.fit(X, y, sample_weight=w_train_all)
        else:
            final_model.fit(X, y)
        return final_model

    logger.info("Entrenando y Optimizando Meta-Modelos XGBoost CLV (Target Log-Drift + CV Purged)...")
    
    logger.info("Optimizando modelo para Drift Win...")
    model_win = optimize_xgb(X_train, y_drift_win_train, dates_train)
    pred_drift_win_log = model_win.predict(X_test)
    
    logger.info("Optimizando modelo para Drift Draw...")
    model_draw = optimize_xgb(X_train, y_drift_draw_train, dates_train)
    pred_drift_draw_log = model_draw.predict(X_test)
    
    logger.info("Optimizando modelo para Drift Loss...")
    model_loss = optimize_xgb(X_train, y_drift_loss_train, dates_train)
    pred_drift_loss_log = model_loss.predict(X_test)
    
    # --------------------------------------------------------------------------
    # ACOPLAMIENTO COHERENTE QUANT DE PROBABILIDADES (SOFTMAX PROBABILITY NORMALIZATION)
    # --------------------------------------------------------------------------
    open_loss_col_test = df_test_orig['open_odds_loss'] if 'open_odds_loss' in df_test_orig.columns else df_test_orig['odds_loss']
    open_draw_col_test = df_test_orig['open_odds_draw'] if 'open_odds_draw' in df_test_orig.columns else df_test_orig['odds_draw']
    open_win_col_test = df_test_orig['open_odds_win'] if 'open_odds_win' in df_test_orig.columns else df_test_orig['odds_win']

    open_loss_test = open_loss_col_test.fillna(df_test_orig['odds_loss']).clip(lower=1.01).values
    open_draw_test = open_draw_col_test.fillna(df_test_orig['odds_draw']).clip(lower=1.01).values
    open_win_test = open_win_col_test.fillna(df_test_orig['odds_win']).clip(lower=1.01).values
    
    vig_open_test = (1.0 / open_loss_test) + (1.0 / open_draw_test) + (1.0 / open_win_test)
    open_fair_loss_test = (1.0 / open_loss_test) / vig_open_test
    open_fair_draw_test = (1.0 / open_draw_test) / vig_open_test
    open_fair_win_test = (1.0 / open_win_test) / vig_open_test
    
    # Estimación de probabilidades implícitas de cierre
    p_close_win_est = open_fair_win_test * np.exp(pred_drift_win_log)
    p_close_draw_est = open_fair_draw_test * np.exp(pred_drift_draw_log)
    p_close_loss_est = open_fair_loss_test * np.exp(pred_drift_loss_log)
    
    total_p_close = p_close_win_est + p_close_draw_est + p_close_loss_est
    total_p_close = np.where(total_p_close <= 0, 1.0, total_p_close)

    # Normalización estricta 1X2 (Softmax Normalization)
    p_fair_close_win = p_close_win_est / total_p_close
    p_fair_close_draw = p_close_draw_est / total_p_close
    p_fair_close_loss = p_close_loss_est / total_p_close
    
    # CLV Lineal Acoplado Coherente Definitivo
    pred_drift_win = np.nan_to_num((p_fair_close_win / open_fair_win_test) - 1.0, nan=0.0)
    pred_drift_draw = np.nan_to_num((p_fair_close_draw / open_fair_draw_test) - 1.0, nan=0.0)
    pred_drift_loss = np.nan_to_num((p_fair_close_loss / open_fair_loss_test) - 1.0, nan=0.0)
    
    def print_metrics(y_true_lin, y_pred_lin, name):
        mae = mean_absolute_error(y_true_lin, y_pred_lin)
        rmse = np.sqrt(mean_squared_error(y_true_lin, y_pred_lin))
        
        r_val, _ = pearsonr(y_pred_lin, y_true_lin) if len(np.unique(y_pred_lin)) > 1 else (0.0, 0.0)
        rho_val, _ = spearmanr(y_pred_lin, y_true_lin) if len(np.unique(y_pred_lin)) > 1 else (0.0, 0.0)
        
        # Direccionalidad binaria (1 si Drift > 0 "Odds bajan (CLV a favor)")
        y_true_bin = (y_true_lin > 0).astype(int)
        y_pred_bin = (y_pred_lin > 0).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        
        # Precision con Filtro CLV > +1.5% (Umbral Financiero Real)
        y_true_filt = (y_true_lin > 0.015).astype(int)
        y_pred_filt = (y_pred_lin > 0.015).astype(int)
        prec_filt = precision_score(y_true_filt, y_pred_filt, zero_division=0)

        # Precision Top 10% Decil de Predicción
        top10_cutoff = np.percentile(y_pred_lin, 90)
        top10_mask = y_pred_lin >= top10_cutoff
        prec_top10 = (y_true_lin[top10_mask] > 0).mean() if top10_mask.sum() > 0 else 0.0

        logger.info(f"--- Métricas para {name} ---")
        logger.info(f"RMSE (Lineal %): {rmse:.5f} | MAE: {mae:.5f}")
        logger.info(f"Correlación Pearson r: {r_val:.4f} | Spearman rho: {rho_val:.4f}")
        logger.info(f"Accuracy Direccional: {acc*100:.2f}% | Precision (>0): {prec*100:.2f}% | Recall (>0): {rec*100:.2f}%")
        logger.info(f"Precision @ CLV > +1.5%: {prec_filt*100:.2f}% | Precision Top 10% Decil: {prec_top10*100:.2f}%")
        logger.info(f"-----------------------------")

    logger.info("=== EVALUACIÓN DEL META-MODELO CLV DEFINITIVO (10/10 PERFECCIÓN QUANT) EN TEST ===")
    print_metrics(y_drift_win_test_lin, pred_drift_win, "Win (Local)")
    print_metrics(y_drift_draw_test_lin, pred_drift_draw, "Draw (Empate)")
    print_metrics(y_drift_loss_test_lin, pred_drift_loss, "Loss (Visitante)")
    
    logger.info("Guardando los modelos entrenados...")
    os.makedirs(SAVE_MODELS_DIR, exist_ok=True)
    clv_features = list(X_train.columns)
    
    joblib.dump({'model': model_win, 'features': clv_features}, os.path.join(SAVE_MODELS_DIR, 'clv_model_win.pkl'))
    joblib.dump({'model': model_draw, 'features': clv_features}, os.path.join(SAVE_MODELS_DIR, 'clv_model_draw.pkl'))
    joblib.dump({'model': model_loss, 'features': clv_features}, os.path.join(SAVE_MODELS_DIR, 'clv_model_loss.pkl'))
    
    logger.info(f"Modelos CLV guardados en {SAVE_MODELS_DIR}")

    # Guardar predicciones pred_clv en escala lineal directamente en test_predictions.parquet
    df_test_preds = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    
    df_test_preds['pred_clv_loss'] = pred_drift_loss
    df_test_preds['pred_clv_draw'] = pred_drift_draw
    df_test_preds['pred_clv_win'] = pred_drift_win
    
    df_test_preds.to_parquet(PREDICTIONS_PATH, engine='fastparquet')
    logger.info(f"Predicciones de CLV Lineal % acopladas exitosamente añadidas a {PREDICTIONS_PATH}")
    logger.info("Ejecuta 'python core/simulate_bankroll.py' a continuación para la evaluación financiera independiente.")

if __name__ == "__main__":
    train_clv_model()


if __name__ == "__main__":
    train_clv_model()
