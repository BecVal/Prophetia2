import os
import pandas as pd
import numpy as np
import logging
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = '../data/processed/matches_with_odds.parquet'
PREDICTIONS_PATH = '../data/processed/test_predictions.parquet'
X_TRAIN_PATH = '../data/processed/X_train.parquet'
X_TEST_PATH = '../data/processed/X_test.parquet'
TRAIN_PREDS_PATH = '../data/processed/train_predictions.parquet'

def train_clv_model():
    if not all(os.path.exists(p) for p in [DATASET_PATH, PREDICTIONS_PATH, X_TRAIN_PATH, X_TEST_PATH, TRAIN_PREDS_PATH]):
        logger.error("Faltan archivos requeridos. Asegúrate de haber ejecutado train.py primero.")
        return

    logger.info("Cargando datasets base y cuotas...")
    df = pd.read_parquet(DATASET_PATH, engine='fastparquet')
    if 'match_date' in df.columns:
        df = df.sort_values('match_date').reset_index(drop=True)
    df = df[df['is_home'] == 1].reset_index(drop=True)

    # Cargar X_train y X_test
    X_train = pd.read_parquet(X_TRAIN_PATH, engine='fastparquet')
    X_test = pd.read_parquet(X_TEST_PATH, engine='fastparquet')
    
    # Cargar probabilidades base
    df_train_preds = pd.read_parquet(TRAIN_PREDS_PATH, engine='fastparquet')
    df_test_preds = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    
    # Calcular implied probs de apertura y cuotas justas (sin vig)
    df['open_implied_loss'] = 1 / df['open_odds_loss']
    df['open_implied_draw'] = 1 / df['open_odds_draw']
    df['open_implied_win'] = 1 / df['open_odds_win']
    open_vig = df['open_implied_loss'] + df['open_implied_draw'] + df['open_implied_win']
    
    df['open_fair_loss'] = df['open_implied_loss'] / open_vig
    df['open_fair_draw'] = df['open_implied_draw'] / open_vig
    df['open_fair_win'] = df['open_implied_win'] / open_vig
    
    # Calcular implied probs de cierre y cuotas justas (sin vig)
    df['implied_loss'] = 1 / df['odds_loss']
    df['implied_draw'] = 1 / df['odds_draw']
    df['implied_win'] = 1 / df['odds_win']
    vig = df['implied_loss'] + df['implied_draw'] + df['implied_win']
    
    df['fair_loss'] = df['implied_loss'] / vig
    df['fair_draw'] = df['implied_draw'] / vig
    df['fair_win'] = df['implied_win'] / vig

    # Drift histórico vs apertura (como proxy continuo) logarítmico (positivo = odds dropean = CLV a favor)
    df['target_loss'] = np.log(df['open_odds_loss'] / df['odds_loss'])
    df['target_draw'] = np.log(df['open_odds_draw'] / df['odds_draw'])
    df['target_win'] = np.log(df['open_odds_win'] / df['odds_win'])
    
    y_drift_loss = df['target_loss'].fillna(0)
    y_drift_draw = df['target_draw'].fillna(0)
    y_drift_win = df['target_win'].fillna(0)
    
    split_idx = int(len(df) * 0.8)
    
    y_drift_loss_train = y_drift_loss.iloc[:split_idx]
    y_drift_draw_train = y_drift_draw.iloc[:split_idx]
    y_drift_win_train = y_drift_win.iloc[:split_idx]
    
    y_drift_loss_test = y_drift_loss.iloc[split_idx:]
    y_drift_draw_test = y_drift_draw.iloc[split_idx:]
    y_drift_win_test = y_drift_win.iloc[split_idx:]
    
    logger.info("Inyectando Probabilidades y Características de Divergencia...")
    # Inyectar para Train (usando las probs finales del Stacker)
    X_train['prob_loss'] = df_train_preds['prob_loss'].values
    X_train['prob_draw'] = df_train_preds['prob_draw'].values
    X_train['prob_win'] = df_train_preds['prob_win'].values
    
    # REMOVED: divergence_loss using fair_loss (closing odds) causes data leakage because closing odds are not available at inference time.
    
    X_train['open_divergence_loss'] = np.log(np.clip(X_train['prob_loss'] / df['open_fair_loss'].iloc[:split_idx].values, 1e-6, 1e6))
    X_train['open_divergence_draw'] = np.log(np.clip(X_train['prob_draw'] / df['open_fair_draw'].iloc[:split_idx].values, 1e-6, 1e6))
    X_train['open_divergence_win'] = np.log(np.clip(X_train['prob_win'] / df['open_fair_win'].iloc[:split_idx].values, 1e-6, 1e6))
    
    # Inyectar para Test (usando las probs finales del Stacker)
    X_test['prob_loss'] = df_test_preds['prob_loss'].values
    X_test['prob_draw'] = df_test_preds['prob_draw'].values
    X_test['prob_win'] = df_test_preds['prob_win'].values
    
    # REMOVED: divergence_loss leakage
    
    X_test['open_divergence_loss'] = np.log(np.clip(X_test['prob_loss'] / df['open_fair_loss'].iloc[split_idx:].values, 1e-6, 1e6))
    X_test['open_divergence_draw'] = np.log(np.clip(X_test['prob_draw'] / df['open_fair_draw'].iloc[split_idx:].values, 1e-6, 1e6))
    X_test['open_divergence_win'] = np.log(np.clip(X_test['prob_win'] / df['open_fair_win'].iloc[split_idx:].values, 1e-6, 1e6))
    
    def optimize_xgb(X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'device': 'cuda'
            }
            model = XGBRegressor(**params)
            
            # Simple Time-based Train/Val split
            val_split = int(len(X) * 0.8)
            X_tr, y_tr = X.iloc[:val_split], y.iloc[:val_split]
            X_va, y_va = X.iloc[val_split:], y.iloc[val_split:]
            
            # Time Decay Weights para el subconjunto de entrenamiento temporal
            weights_tr = np.exp(np.linspace(-2, 0, len(X_tr)))
            
            model.fit(X_tr, y_tr, sample_weight=weights_tr)
            preds = model.predict(X_va)
            return mean_absolute_error(y_va, preds)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['device'] = 'cuda'
        
        final_model = XGBRegressor(**best_params, eval_metric='rmse')
        weights_all = np.exp(np.linspace(-2, 0, len(X)))
        final_model.fit(X, y, sample_weight=weights_all)
        return final_model

    logger.info("Entrenando y Optimizando Meta-Modelos XGBoost (Odds Drift) con Optuna...")
    
    logger.info("Optimizando modelo para Drift Win...")
    model_win = optimize_xgb(X_train, y_drift_win_train)
    pred_drift_win = model_win.predict(X_test)
    
    logger.info("Optimizando modelo para Drift Draw...")
    model_draw = optimize_xgb(X_train, y_drift_draw_train)
    pred_drift_draw = model_draw.predict(X_test)
    
    logger.info("Optimizando modelo para Drift Loss...")
    model_loss = optimize_xgb(X_train, y_drift_loss_train)
    pred_drift_loss = model_loss.predict(X_test)
    
    def print_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        
        # Direccionalidad binaria (1 si Drift > 0 "Mueve a favor (odds bajan)", 0 si Drift <= 0 "Mueve en contra o estable")
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        
        logger.info(f"--- Métricas para {name} ---")
        logger.info(f"MAE: {mae:.5f}")
        logger.info(f"Accuracy Direccional: {acc*100:.2f}%")
        logger.info(f"Precision (Detectar Drift > 0): {prec*100:.2f}%")
        logger.info(f"Recall (Detectar Drift > 0): {rec*100:.2f}%")
        logger.info(f"-----------------------------")

    logger.info("=== EVALUACIÓN DEL META-MODELO CLV ===")
    print_metrics(y_drift_win_test, pred_drift_win, "Win (Local)")
    print_metrics(y_drift_draw_test, pred_drift_draw, "Draw (Empate)")
    print_metrics(y_drift_loss_test, pred_drift_loss, "Loss (Visitante)")
    
    logger.info("Actualizando test_predictions.parquet con el meta-modelo optimizado...")
    
    # Convertimos de Log-Ratio (pred_clv) a % matemático lineal (como requiere el simulador)
    df_test_preds['pred_clv_loss'] = np.exp(pred_drift_loss) - 1
    df_test_preds['pred_clv_draw'] = np.exp(pred_drift_draw) - 1
    df_test_preds['pred_clv_win'] = np.exp(pred_drift_win) - 1
    
    df_test_preds.to_parquet(PREDICTIONS_PATH, engine='fastparquet')
    logger.info(f"Meta-predicciones añadidas exitosamente a {PREDICTIONS_PATH}")
    logger.info("Ejecuta 'python core/simulate_bankroll.py' a continuación para la evaluación financiera independiente.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_clv_model()
