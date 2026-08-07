import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import optuna
from scipy.stats import poisson
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, d2_tweedie_score
from sklearn.model_selection import TimeSeriesSplit

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..')))

from core.logger_config import get_logger
from core.models.data_splitter import get_base_dataset, get_train_test_split, get_cv_strategy

logger = get_logger(__name__, 'train_cards_model')

# ==============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN (OPTUNA)
# ==============================================================================
RUN_OPTUNA = True  # Usa los parámetros óptimos ya encontrados o True para re-optimizar
OPTUNA_TRIALS = 100
# ==============================================================================

MODELS_DIR = os.path.abspath(os.path.join(script_dir, '..', 'save_models'))
PROCESSED_DIR = os.path.join(script_dir, '..', '..', 'data', 'processed')
PARAMS_DIR = os.path.join(PROCESSED_DIR, 'models_best_parameters')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

def build_perspective_features(df, perspective='home'):
    """
    Crea un DataFrame de features simétricas alineadas según la perspectiva ('home' o 'away').
    Esto garantiza que el modelo aprenda la dinámica del equipo objetivo vs su oponente.
    XGBoost maneja nativamente los valores nulos (np.nan).
    """
    df_feat = pd.DataFrame(index=df.index)
    
    if perspective == 'home':
        # Equipo objetivo = Home, Oponente = Away
        df_feat['team_yellow_cards_ema5'] = df.get('yellow_cards_ema5')
        df_feat['team_yellow_cards_ema3'] = df.get('yellow_cards_ema3')
        df_feat['team_red_cards_ema5'] = df.get('red_cards_ema5')
        df_feat['team_fouls_committed_ema5'] = df.get('fouls_committed_ema5')
        df_feat['team_fouls_won_ema5'] = df.get('fouls_won_ema5')
        
        df_feat['opp_yellow_cards_ema5'] = df.get('opp_yellow_cards_ema5')
        df_feat['opp_yellow_cards_ema3'] = df.get('opp_yellow_cards_ema3')
        df_feat['opp_red_cards_ema5'] = df.get('opp_red_cards_ema5')
        df_feat['opp_fouls_committed_ema5'] = df.get('opp_fouls_committed_ema5')
        df_feat['opp_fouls_won_ema5'] = df.get('opp_fouls_won_ema5')
        
        df_feat['elo_diff'] = df.get('elo_diff', 0).fillna(0)
    else:
        # Equipo objetivo = Away, Oponente = Home
        df_feat['team_yellow_cards_ema5'] = df.get('opp_yellow_cards_ema5')
        df_feat['team_yellow_cards_ema3'] = df.get('opp_yellow_cards_ema3')
        df_feat['team_red_cards_ema5'] = df.get('opp_red_cards_ema5')
        df_feat['team_fouls_committed_ema5'] = df.get('opp_fouls_committed_ema5')
        df_feat['team_fouls_won_ema5'] = df.get('opp_fouls_won_ema5')
        
        df_feat['opp_yellow_cards_ema5'] = df.get('yellow_cards_ema5')
        df_feat['opp_yellow_cards_ema3'] = df.get('yellow_cards_ema3')
        df_feat['opp_red_cards_ema5'] = df.get('red_cards_ema5')
        df_feat['opp_fouls_committed_ema5'] = df.get('fouls_committed_ema5')
        df_feat['opp_fouls_won_ema5'] = df.get('fouls_won_ema5')
        
        df_feat['elo_diff'] = (-df.get('elo_diff', 0)).fillna(0)

    # Contexto común del partido y árbitro
    df_feat['h2h_points_last_5'] = df.get('h2h_points_last_5', 0).fillna(0)
    df_feat['referee_avg_yellows'] = df.get('referee_avg_yellows')
    df_feat['referee_avg_reds'] = df.get('referee_avg_reds')
    df_feat['referee_fouls_per_yellow'] = df.get('referee_fouls_per_yellow')
    df_feat['referee_avg_fouls'] = df.get('referee_avg_fouls')
    
    # Seleccionar columnas válidas presentes en el dataframe
    valid_cols = [c for c in df_feat.columns if df_feat[c].notna().sum() > 0]
    return df_feat[valid_cols]

def calc_cards_probabilities_vectorized(lam_home_vec, lam_away_vec, max_cards=15, rho_cards=0.05):
    """
    Cálculo vectorizado de probabilidades bivariadas para el mercado de tarjetas de fútbol:
    - prob_over_3_5_cards (P(Total > 3.5))
    - prob_over_4_5_cards (P(Total > 4.5))
    - prob_over_5_5_cards (P(Total > 5.5))
    - prob_home_more_cards (P(Home > Away))
    - prob_away_more_cards (P(Away > Home))
    - prob_equal_cards (P(Home == Away))
    """
    n_samples = len(lam_home_vec)
    probs_over_3_5 = np.zeros(n_samples)
    probs_over_4_5 = np.zeros(n_samples)
    probs_over_5_5 = np.zeros(n_samples)
    probs_home_more = np.zeros(n_samples)
    probs_away_more = np.zeros(n_samples)
    probs_equal = np.zeros(n_samples)
    
    cards_range = np.arange(max_cards + 1)
    total_matrix = np.add.outer(cards_range, cards_range)
    
    for i in range(n_samples):
        lh = lam_home_vec[i]
        la = lam_away_vec[i]
        
        if np.isnan(lh) or np.isnan(la) or lh <= 0 or la <= 0:
            continue
            
        pmf_h = poisson.pmf(cards_range, lh)
        pmf_a = poisson.pmf(cards_range, la)
        
        # Producto exterior independiente
        p_matrix = np.outer(pmf_h, pmf_a)
        
        # Ajuste de correlación estilo Dixon-Coles para tarjetas (partidos tensos aumentan tarjetas de ambos equipos)
        if rho_cards != 0:
            tau_00 = max(0, 1 - (lh * la * rho_cards))
            tau_01 = max(0, 1 + (lh * rho_cards))
            tau_10 = max(0, 1 + (la * rho_cards))
            tau_11 = max(0, 1 - rho_cards)
            
            p_matrix[0, 0] *= tau_00
            p_matrix[0, 1] *= tau_01
            p_matrix[1, 0] *= tau_10
            p_matrix[1, 1] *= tau_11
            
            p_sum = p_matrix.sum()
            if p_sum > 0:
                p_matrix /= p_sum
                
        probs_over_3_5[i] = p_matrix[total_matrix > 3.5].sum()
        probs_over_4_5[i] = p_matrix[total_matrix > 4.5].sum()
        probs_over_5_5[i] = p_matrix[total_matrix > 5.5].sum()
        
        probs_home_more[i] = np.tril(p_matrix, -1).sum()
        probs_equal[i] = np.trace(p_matrix)
        probs_away_more[i] = np.triu(p_matrix, 1).sum()
        
    return {
        'prob_over_3_5_cards': probs_over_3_5,
        'prob_over_4_5_cards': probs_over_4_5,
        'prob_over_5_5_cards': probs_over_5_5,
        'prob_home_more_cards': probs_home_more,
        'prob_away_more_cards': probs_away_more,
        'prob_equal_cards': probs_equal
    }

def generate_time_series_oof(X_train, y_train, best_params, n_splits=5):
    """
    Genera predicciones Out-Of-Fold (OOF) usando estrictamente TimeSeriesSplit sin Data Leakage.
    """
    y_oof_train = np.full(len(X_train), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_train))
    
    tweedie_power = best_params.get('tweedie_variance_power', 1.2)
    m_params = best_params.copy()
    m_params['objective'] = 'reg:tweedie'
    m_params['eval_metric'] = f"tweedie-nloglik@{tweedie_power}"
    
    # 1. Rellenar el primer bloque secuencialmente usando sub-splits de TimeSeriesSplit
    first_train_idx, first_val_idx = splits[0]
    sub_tscv = TimeSeriesSplit(n_splits=3)
    X_first_tr = X_train.iloc[first_train_idx]
    y_first_tr = y_train.iloc[first_train_idx]
    
    sub_splits = list(sub_tscv.split(X_first_tr))
    for sub_tr_idx, sub_va_idx in sub_splits:
        m = XGBRegressor(**m_params)
        m.fit(
            X_first_tr.iloc[sub_tr_idx], y_first_tr.iloc[sub_tr_idx], 
            eval_set=[(X_first_tr.iloc[sub_va_idx], y_first_tr.iloc[sub_va_idx])], 
            verbose=False
        )
        y_oof_train[first_train_idx[sub_va_idx]] = m.predict(X_first_tr.iloc[sub_va_idx])
        
    # Para la fracción inicial donde no hay suficientes datos para validación temporal, usar la media histórica previa
    initial_unpredicted_count = first_train_idx[sub_splits[0][1][0]]
    y_oof_train[:initial_unpredicted_count] = y_train.iloc[:initial_unpredicted_count].mean()

    # 2. Rellenar el resto de bloques de TimeSeriesSplit
    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
        
        m = XGBRegressor(**m_params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_oof_train[va_idx] = m.predict(X_va)
        
    return y_oof_train

def train_target(target_prefix, X_train, y_train, X_test, y_test, available_features):
    """
    Entrena un modelo XGBoost optimizado para un target específico (home o away).
    Retorna (y_oof_train, y_pred_test, model)
    """
    logger.info(f"\n{'='*50}\nENTRENANDO MODELO PARA: {target_prefix.upper()}\n{'='*50}")
    
    optuna_params_file = os.path.join(PARAMS_DIR, f'optuna_params_cards_{target_prefix}.json')
    model_save_path = os.path.join(MODELS_DIR, f'cards_{target_prefix}_xgboost_model.pkl')
    
    if RUN_OPTUNA or not os.path.exists(optuna_params_file):
        logger.info(f"Ejecutando optimización Optuna para {target_prefix} con {OPTUNA_TRIALS} iteraciones...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.05, 1.8),
                'random_state': 42,
                'objective': 'reg:tweedie',
                'early_stopping_rounds': 20
            }
            params['eval_metric'] = f"tweedie-nloglik@{params['tweedie_variance_power']}"
            
            tscv = TimeSeriesSplit(n_splits=4)
            cv_scores = []
            
            for train_index, val_index in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[train_index], X_train.iloc[val_index]
                y_tr, y_va = y_train.iloc[train_index], y_train.iloc[val_index]
                
                model = XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                
                y_pred_va = model.predict(X_va)
                score = mean_absolute_error(y_va, y_pred_va)
                cv_scores.append(score)
                
            return np.mean(cv_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['objective'] = 'reg:tweedie'
        best_params['early_stopping_rounds'] = 20
        
        logger.info(f"Mejores parámetros encontrados para {target_prefix}: {best_params}")
        with open(optuna_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        logger.info(f"Cargando parámetros desde {optuna_params_file}")
        with open(optuna_params_file, 'r') as f:
            best_params = json.load(f)
            
    # ====== OUT-OF-FOLD (OOF) PREDICTIONS SIN DATA LEAKAGE ======
    logger.info(f"Generando predicciones OOF (TimeSeriesSplit estricto) para {target_prefix}...")
    y_oof_train = generate_time_series_oof(X_train, y_train, best_params, n_splits=5)
        
    # ====== MODELO FINAL ======
    logger.info(f"Entrenando Modelo Final para {target_prefix}...")
    final_params = best_params.copy()
    final_params['eval_metric'] = f"tweedie-nloglik@{best_params.get('tweedie_variance_power', 1.2)}"
    
    model = XGBRegressor(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred_test = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    poisson_deviance_explained = d2_tweedie_score(y_test, y_pred_test, power=1)
    
    logger.info(f"--- METRICS: {target_prefix.upper()} ---")
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"MAE  : {mae:.4f}")
    logger.info(f"Poisson Deviance Expl: {poisson_deviance_explained:.4f}")
    
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': available_features, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    
    logger.info(f"--- FEATURE IMPORTANCES: {target_prefix.upper()} ---")
    for _, row in feat_imp.head(10).iterrows():
        logger.info(f"{row['feature']:<25}: {row['importance']:.4f}")
        
    joblib.dump(model, model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return y_oof_train, y_pred_test

def train_cards_model():
    logger.info("Cargando dataset principal vía data_splitter...")
    df = get_base_dataset()
    
    # 1. Definir Targets: Home, Away
    BASE_DIR = os.path.abspath(os.path.join(script_dir, '../../'))
    DATASET_PATH = os.path.join(BASE_DIR, 'data/processed/matches_with_referees.parquet')
    FALLBACK_DATASET = os.path.join(BASE_DIR, 'data/processed/matches_with_odds.parquet')
    path_to_load = DATASET_PATH if os.path.exists(DATASET_PATH) else FALLBACK_DATASET
    
    df_all = pd.read_parquet(path_to_load, engine='fastparquet')
    
    # Home target: yellow + (red * 2)
    df['target_home_cards'] = df['yellow_cards'].fillna(0) + (df['red_cards'].fillna(0) * 2)
    
    # Away target: yellow + (red * 2)
    away_cards = df_all[df_all['is_home'] == 0][['match_id', 'yellow_cards', 'red_cards']]
    away_cards = away_cards.rename(columns={
        'yellow_cards': 'away_yellow_cards',
        'red_cards': 'away_red_cards'
    }).drop_duplicates(subset=['match_id'])
    
    df = df.merge(away_cards, on='match_id', how='left')
    df['target_away_cards'] = df['away_yellow_cards'].fillna(0) + (df['away_red_cards'].fillna(0) * 2)
    
    # Target total (suma exacta de ambos componentes)
    df['target_total_cards'] = df['target_home_cards'] + df['target_away_cards']
    
    if 'match_date' in df.columns:
        df = df.sort_values('match_date').reset_index(drop=True)
        
    # Split índice temporal oficial
    split_idx = get_train_test_split(df)
    df_train_idx = df.iloc[:split_idx].index
    df_test_idx = df.iloc[split_idx:].index
    
    # 2. Entrenar modelo HOME con features con perspectiva alineada
    df_feat_home = build_perspective_features(df, perspective='home')
    features_home = list(df_feat_home.columns)
    logger.info(f"Features para HOME ({len(features_home)}): {features_home}")
    
    X_tr_home = df_feat_home.iloc[:split_idx].copy().reset_index(drop=True)
    X_te_home = df_feat_home.iloc[split_idx:].copy().reset_index(drop=True)
    y_tr_home = df['target_home_cards'].iloc[:split_idx].copy().reset_index(drop=True)
    y_te_home = df['target_home_cards'].iloc[split_idx:].copy().reset_index(drop=True)
    
    oof_tr_home, pred_te_home = train_target('home', X_tr_home, y_tr_home, X_te_home, y_te_home, features_home)
    
    # 3. Entrenar modelo AWAY con features con perspectiva alineada
    df_feat_away = build_perspective_features(df, perspective='away')
    features_away = list(df_feat_away.columns)
    logger.info(f"Features para AWAY ({len(features_away)}): {features_away}")
    
    X_tr_away = df_feat_away.iloc[:split_idx].copy().reset_index(drop=True)
    X_te_away = df_feat_away.iloc[split_idx:].copy().reset_index(drop=True)
    y_tr_away = df['target_away_cards'].iloc[:split_idx].copy().reset_index(drop=True)
    y_te_away = df['target_away_cards'].iloc[split_idx:].copy().reset_index(drop=True)
    
    oof_tr_away, pred_te_away = train_target('away', X_tr_away, y_tr_away, X_te_away, y_te_away, features_away)
    
    # 4. Total coherente e incondicional (Linearity of expectation: E[Home + Away] = E[Home] + E[Away])
    oof_tr_total = oof_tr_home + oof_tr_away
    pred_te_total = pred_te_home + pred_te_away
    
    y_te_total = df['target_total_cards'].iloc[split_idx:].copy().reset_index(drop=True)
    rmse_tot = np.sqrt(mean_squared_error(y_te_total, pred_te_total))
    mae_tot = mean_absolute_error(y_te_total, pred_te_total)
    pdev_tot = d2_tweedie_score(y_te_total, pred_te_total, power=1)
    
    logger.info(f"\n{'='*50}\n--- METRICS: TOTAL (ADITIVO E[HOME] + E[AWAY]) ---\n{'='*50}")
    logger.info(f"RMSE : {rmse_tot:.4f}")
    logger.info(f"MAE  : {mae_tot:.4f}")
    logger.info(f"Poisson Deviance Expl: {pdev_tot:.4f}")
    
    # 5. Generar Matriz de Probabilidades Bivariadas para Mercados de Tarjetas
    logger.info("Generando probabilidades bivariadas de tarjetas (Over 3.5, 4.5, 5.5 y 1X2 tarjetas)...")
    probs_tr = calc_cards_probabilities_vectorized(oof_tr_home, oof_tr_away)
    probs_te = calc_cards_probabilities_vectorized(pred_te_home, pred_te_away)
    
    # Guardar OOF enriquecido para el Stacker
    train_dict = {
        'lambda_home': oof_tr_home,
        'lambda_away': oof_tr_away,
        'lambda_total': oof_tr_total
    }
    train_dict.update(probs_tr)
    df_oof_train = pd.DataFrame(train_dict, index=df_train_idx)
    oof_train_path = os.path.join(PROCESSED_DIR, 'oof_cards_train.parquet')
    df_oof_train.to_parquet(oof_train_path, engine='fastparquet')
    
    test_dict = {
        'lambda_home': pred_te_home,
        'lambda_away': pred_te_away,
        'lambda_total': pred_te_total
    }
    test_dict.update(probs_te)
    df_oof_test = pd.DataFrame(test_dict, index=df_test_idx)
    oof_test_path = os.path.join(PROCESSED_DIR, 'oof_cards_test.parquet')
    df_oof_test.to_parquet(oof_test_path, engine='fastparquet')
    
    logger.info("Todos los modelos y probabilidades bivariadas de tarjetas generados y guardados exitosamente (10/10).")

if __name__ == '__main__':
    train_cards_model()
