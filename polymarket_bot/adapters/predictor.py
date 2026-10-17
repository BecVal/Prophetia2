import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.cli_predictor import CLIQuantPredictor

logger = logging.getLogger('polymarket_bot.adapters.predictor')


class ProphetiaPredictor:
    """
    Adaptador del motor cuantitativo de Prophetia2 para Polymarket.
    Delega en CLIQuantPredictor para garantizar paridad exacta de features
    con los 9 modelos base, Sub-Stacker Fundamental, Meta-Stacker Final y CLV.
    """
    def __init__(self, core_dir):
        self.core_dir = core_dir
        self._engine = CLIQuantPredictor()

    def predict_match(self, home_team, away_team, odds_1, odds_X, odds_2, open_odds_1=None, open_odds_X=None, open_odds_2=None):
        o1 = open_odds_1 if open_odds_1 is not None else odds_1
        oX = open_odds_X if open_odds_X is not None else odds_X
        o2 = open_odds_2 if open_odds_2 is not None else odds_2

        res = self._engine.predict_match(
            home_name=home_team,
            away_name=away_team,
            odds_1=o1,
            odds_X=oX,
            odds_2=o2,
        )
        if not res:
            logger.warning(f"No hay suficientes stats para {home_team} vs {away_team}")
            return None

        return {
            'competition': res['competition'],
            'home_prob': float(res['final_probs'][2]),
            'draw_prob': float(res['final_probs'][1]),
            'away_prob': float(res['final_probs'][0]),
            'pred_clv_win': float(res['clv_preds'][2]),
            'pred_clv_draw': float(res['clv_preds'][1]),
            'pred_clv_loss': float(res['clv_preds'][0]),
        }
