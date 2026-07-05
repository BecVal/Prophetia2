import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson, entropy
import logging

logger = logging.getLogger('polymarket_bot.adapters.predictor')

def safe_logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

class ProphetiaPredictor:
    """
    Pipiline Quant Completo de Prophetia2.
    Integra modelos base (Poisson, Context), submodelos de mercado (Market, GBM),
    ensamble final (Stacker Final) y el modelo dinámico CLV.
    """
    def __init__(self, core_dir):
        self.core_dir = core_dir
        self.core_save_dir = os.path.join(core_dir, 'save_models')
        self.root_save_dir = os.path.join(core_dir, '..', 'save_models')
            
        # 1. Modelos Base (Fundamentales)
        self.poisson_data = joblib.load(os.path.join(self.core_save_dir, 'poisson_model.pkl'))
        self.context_data = joblib.load(os.path.join(self.core_save_dir, 'context_model.pkl'))
        self.stacker_old_data = joblib.load(os.path.join(self.core_save_dir, 'stacker_model.pkl')) # Usado como proxy para Fundamental Stacker
        
        # 2. Submodelos Quant de Mercado
        self.market_data = joblib.load(os.path.join(self.core_save_dir, 'market_model.pkl'))
        self.gbm_data = joblib.load(os.path.join(self.core_save_dir, 'gbm_model.pkl'))
        
        # 3. Meta-Modelos
        self.stacker_final_data = joblib.load(os.path.join(self.core_save_dir, 'stacker_final_model.pkl'))
        
        # 4. CLV Models (guardados en root/save_models por train_clv_model.py)
        self.clv_win_data = joblib.load(os.path.join(self.root_save_dir, 'clv_model_win.pkl'))
        self.clv_draw_data = joblib.load(os.path.join(self.root_save_dir, 'clv_model_draw.pkl'))
        self.clv_loss_data = joblib.load(os.path.join(self.root_save_dir, 'clv_model_loss.pkl'))
        
        self.dataset_path = os.path.join(core_dir, '..', 'data', 'processed', 'matches_with_odds.parquet')
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset no encontrado en {self.dataset_path}")
        self.df = pd.read_parquet(self.dataset_path)
        if 'match_date' in self.df.columns:
            self.df['match_date'] = pd.to_datetime(self.df['match_date'])
        
    def _calc_match_probabilities(self, lam_scored, lam_conceded, rho=-0.15, max_goals=10):
        x = np.arange(max_goals + 1)
        y = np.arange(max_goals + 1)
        
        pmf_scored = poisson.pmf(x, lam_scored)
        pmf_conceded = poisson.pmf(y, lam_conceded)
        
        prob_matrix = np.outer(pmf_scored, pmf_conceded)
        
        tau_00 = max(0, 1 - (lam_scored * lam_conceded * rho))
        tau_01 = max(0, 1 + (lam_scored * rho))
        tau_10 = max(0, 1 + (lam_conceded * rho))
        tau_11 = max(0, 1 - rho)
        
        prob_matrix[0, 0] *= tau_00
        prob_matrix[0, 1] *= tau_01
        prob_matrix[1, 0] *= tau_10
        prob_matrix[1, 1] *= tau_11
        
        win_prob = np.tril(prob_matrix, -1).sum()
        draw_prob = np.trace(prob_matrix)
        loss_prob = np.triu(prob_matrix, 1).sum()
        
        total = win_prob + draw_prob + loss_prob
        if total > 0:
            return win_prob / total, draw_prob / total, loss_prob / total
        return 0.0, 0.0, 0.0

    def _get_latest_team_stats(self, df, team_name):
        team_df = df[(df['team'] == team_name)].sort_values('match_date')
        if team_df.empty:
            return None
        return team_df.iloc[-1]
        
    def _get_gbm_historical_stats(self, df, team_name, current_drift_win, current_drift_draw, current_drift_loss):
        """Calcula la volatilidad histórica de los drifts (últimos 10 partidos)"""
        team_df = df[(df['team'] == team_name)].sort_values('match_date').copy()
        
        if len(team_df) < 3:
            return {
                'gbm_sigma_win': 0.1, 'gbm_sigma_draw': 0.1, 'gbm_sigma_loss': 0.1,
                'gbm_z_win': 0.0, 'gbm_z_draw': 0.0, 'gbm_z_loss': 0.0
            }
            
        # Recrear el drift de los últimos partidos
        inv_w = 1 / team_df['odds_win']
        inv_d = 1 / team_df['odds_draw']
        inv_l = 1 / team_df['odds_loss']
        vig = inv_w + inv_d + inv_l
        p_w = inv_w / vig
        p_d = inv_d / vig
        p_l = inv_l / vig
        
        inv_ow = 1 / team_df['open_odds_win']
        inv_od = 1 / team_df['open_odds_draw']
        inv_ol = 1 / team_df['open_odds_loss']
        vig_o = inv_ow + inv_od + inv_ol
        po_w = inv_ow / vig_o
        po_d = inv_od / vig_o
        po_l = inv_ol / vig_o
        
        team_df['mu_win'] = safe_logit(p_w) - safe_logit(po_w)
        team_df['mu_draw'] = safe_logit(p_d) - safe_logit(po_d)
        team_df['mu_loss'] = safe_logit(p_l) - safe_logit(po_l)
        
        # Tomar últimos 10
        last_10 = team_df.tail(10)
        
        sigma_win = max(last_10['mu_win'].std(), 1e-5)
        sigma_draw = max(last_10['mu_draw'].std(), 1e-5)
        sigma_loss = max(last_10['mu_loss'].std(), 1e-5)
        
        mean_win = last_10['mu_win'].mean()
        mean_draw = last_10['mu_draw'].mean()
        mean_loss = last_10['mu_loss'].mean()
        
        z_win = (current_drift_win - mean_win) / sigma_win
        z_draw = (current_drift_draw - mean_draw) / sigma_draw
        z_loss = (current_drift_loss - mean_loss) / sigma_loss
        
        if np.isnan(sigma_win): sigma_win = 0.1
        if np.isnan(sigma_draw): sigma_draw = 0.1
        if np.isnan(sigma_loss): sigma_loss = 0.1
        
        return {
            'gbm_sigma_win': sigma_win, 'gbm_sigma_draw': sigma_draw, 'gbm_sigma_loss': sigma_loss,
            'gbm_z_win': z_win, 'gbm_z_draw': z_draw, 'gbm_z_loss': z_loss
        }

    def predict_match(self, home_team, away_team, odds_1, odds_X, odds_2, open_odds_1=None, open_odds_X=None, open_odds_2=None):
        home_stats = self._get_latest_team_stats(self.df, home_team)
        away_stats = self._get_latest_team_stats(self.df, away_team)
        
        if home_stats is None or away_stats is None:
            logger.warning(f"No hay suficientes stats para {home_team} vs {away_team}")
            return None
            
        competition = home_stats.get('competition', 'DEFAULT')
        
        if open_odds_1 is None: open_odds_1 = odds_1
        if open_odds_X is None: open_odds_X = odds_X
        if open_odds_2 is None: open_odds_2 = odds_2
            
        input_data = {}
        all_base_features = set(self.poisson_data['features'] + self.context_data['features'])
        
        for f in all_base_features:
            input_data[f] = home_stats.get(f, 0.0)
                
        input_data['is_home'] = 1
        input_data['team_elo'] = home_stats.get('team_elo', 1500)
        input_data['opp_elo'] = away_stats.get('team_elo', 1500)
        input_data['elo_diff'] = input_data['team_elo'] - input_data['opp_elo']
        
        # --- MARKET FEATURES ---
        inv_w = 1 / max(odds_1, 1.01)
        inv_d = 1 / max(odds_X, 1.01)
        inv_l = 1 / max(odds_2, 1.01)
        vig = inv_w + inv_d + inv_l
        p_w = inv_w / vig
        p_d = inv_d / vig
        p_l = inv_l / vig
        
        inv_ow = 1 / max(open_odds_1, 1.01)
        inv_od = 1 / max(open_odds_X, 1.01)
        inv_ol = 1 / max(open_odds_2, 1.01)
        vig_o = inv_ow + inv_od + inv_ol
        po_w = inv_ow / vig_o
        po_d = inv_od / vig_o
        po_l = inv_ol / vig_o
        
        input_data['open_prob_win'] = po_w
        input_data['open_prob_draw'] = po_d
        input_data['open_prob_loss'] = po_l
        
        df_input = pd.DataFrame([input_data])
        
        # --- LAYER 1: FUNDAMENTAL PROBS (via Poisson, Context, and Old Stacker) ---
        df_poisson = df_input[self.poisson_data['features']].fillna(0)
        lam_scored = self.poisson_data['model_scored'].predict(df_poisson)[0]
        lam_conceded = self.poisson_data['model_conceded'].predict(df_poisson)[0]
        poisson_win, poisson_draw, poisson_loss = self._calc_match_probabilities(lam_scored, lam_conceded)
        
        df_context = df_input[self.context_data['features']].fillna(0)
        ctx_probs = self.context_data['model'].predict_proba(df_context)[0]
        
        old_stacker_input = {
            'predicted_xg_scored': lam_scored, 'predicted_xg_conceded': lam_conceded,
            'poisson_win_prob': poisson_win, 'poisson_draw_prob': poisson_draw, 'poisson_loss_prob': poisson_loss,
            'prob_loss_ctx': ctx_probs[0], 'prob_draw_ctx': ctx_probs[1], 'prob_win_ctx': ctx_probs[2],
            'open_prob_loss': po_l, 'open_prob_draw': po_d, 'open_prob_win': po_w
        }
        df_old_stacker = pd.DataFrame([old_stacker_input])[self.stacker_old_data['features']]
        fund_probs = self.stacker_old_data['model'].predict_proba(df_old_stacker)[0]
        fund_prob_loss, fund_prob_draw, fund_prob_win = fund_probs / np.sum(fund_probs)
        
        # --- LAYER 2: MARKET MODEL ---
        market_input = {
            'open_prob_win': po_w, 'open_prob_draw': po_d, 'open_prob_loss': po_l,
            'prob_win_implied': p_w, 'prob_draw_implied': p_d, 'prob_loss_implied': p_l,
            'steam_win': p_w - po_w, 'steam_draw': p_d - po_d, 'steam_loss': p_l - po_l,
            'vig_open': vig_o - 1, 'vig_close': vig - 1
        }
        df_market = pd.DataFrame([market_input])[self.market_data['features']]
        mkt_probs = self.market_data['model'].predict_proba(df_market)[0]
        prob_loss_mkt, prob_draw_mkt, prob_win_mkt = mkt_probs
        
        # --- LAYER 3: GBM MODEL ---
        gbm_mu_win = safe_logit(p_w) - safe_logit(po_w)
        gbm_mu_draw = safe_logit(p_d) - safe_logit(po_d)
        gbm_mu_loss = safe_logit(p_l) - safe_logit(po_l)
        
        gbm_stats = self._get_gbm_historical_stats(self.df, home_team, gbm_mu_win, gbm_mu_draw, gbm_mu_loss)
        
        gbm_input = {
            'gbm_mu_win': gbm_mu_win, 'gbm_mu_draw': gbm_mu_draw, 'gbm_mu_loss': gbm_mu_loss,
            'gbm_base_prob_win': p_w, 'gbm_base_prob_draw': p_d, 'gbm_base_prob_loss': p_l,
        }
        gbm_input.update(gbm_stats)
        
        df_gbm = pd.DataFrame([gbm_input])[self.gbm_data['features']]
        gbm_probs = self.gbm_data['model'].predict_proba(df_gbm)[0]
        prob_loss_gbm, prob_draw_gbm, prob_win_gbm = gbm_probs
        
        # --- LAYER 4: META FEATURES & FINAL STACKER ---
        loss_probs = [fund_prob_loss, prob_loss_mkt, prob_loss_gbm]
        draw_probs = [fund_prob_draw, prob_draw_mkt, prob_draw_gbm]
        win_probs  = [fund_prob_win, prob_win_mkt, prob_win_gbm]
        
        meta_std_loss = np.std(loss_probs)
        meta_std_draw = np.std(draw_probs)
        meta_std_win = np.std(win_probs)
        
        mean_loss = np.mean(loss_probs)
        mean_draw = np.mean(draw_probs)
        mean_win = np.mean(win_probs)
        meta_entropy = entropy([mean_loss, mean_draw, mean_win])
        
        # Manejar competition_id (se factoriza normalmente en train_stacker, usamos 0 si no lo conocemos)
        comps = self.df['competition'].unique().tolist()
        comp_id = comps.index(competition) if competition in comps else 0
        
        final_input = {
            'fund_prob_loss': fund_prob_loss, 'fund_prob_draw': fund_prob_draw, 'fund_prob_win': fund_prob_win,
            'prob_loss_mkt': prob_loss_mkt, 'prob_draw_mkt': prob_draw_mkt, 'prob_win_mkt': prob_win_mkt,
            'prob_loss_gbm': prob_loss_gbm, 'prob_draw_gbm': prob_draw_gbm, 'prob_win_gbm': prob_win_gbm,
            'meta_std_loss': meta_std_loss, 'meta_std_draw': meta_std_draw, 'meta_std_win': meta_std_win,
            'meta_entropy': meta_entropy,
            'implied_open_loss': 1 / max(open_odds_2, 1.01),
            'implied_open_draw': 1 / max(open_odds_X, 1.01),
            'implied_open_win': 1 / max(open_odds_1, 1.01),
            'competition_id': comp_id
        }
        
        df_final = pd.DataFrame([final_input])[self.stacker_final_data['features']]
        final_probs = self.stacker_final_data['model'].predict_proba(df_final)[0]
        prob_loss, prob_draw, prob_win = final_probs / np.sum(final_probs)
        
        # --- LAYER 5: CLV MODEL ---
        # Features: Todas las de df_final + prob_loss/draw/win + open_divergence_loss/draw/win
        clv_input = final_input.copy()
        clv_input.update({
            'prob_loss': prob_loss,
            'prob_draw': prob_draw,
            'prob_win': prob_win,
            'open_divergence_loss': np.log(np.clip(prob_loss / po_l, 1e-6, 1e6)),
            'open_divergence_draw': np.log(np.clip(prob_draw / po_d, 1e-6, 1e6)),
            'open_divergence_win': np.log(np.clip(prob_win / po_w, 1e-6, 1e6))
        })
        
        df_clv = pd.DataFrame([clv_input])[self.clv_win_data['features']]
        
        pred_drift_win = self.clv_win_data['model'].predict(df_clv)[0]
        pred_drift_draw = self.clv_draw_data['model'].predict(df_clv)[0]
        pred_drift_loss = self.clv_loss_data['model'].predict(df_clv)[0]
        
        pred_clv_win = np.exp(pred_drift_win) - 1
        pred_clv_draw = np.exp(pred_drift_draw) - 1
        pred_clv_loss = np.exp(pred_drift_loss) - 1
        
        return {
            'competition': competition,
            'home_prob': prob_win,
            'draw_prob': prob_draw,
            'away_prob': prob_loss,
            'pred_clv_win': pred_clv_win,
            'pred_clv_draw': pred_clv_draw,
            'pred_clv_loss': pred_clv_loss
        }
