"""
Módulo de Optimización de Portafolio de Harry Markowitz (1952) - Mean-Variance Optimization (MVO)
Diseñado para la gestión cuantitativa de apuestas deportivas en Prophetia2.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

class MarkowitzPortfolioOptimizer:
    """
    Optimizador de Portafolio Cuantitativo basado en Markowitz (1952).
    Resuelve el problema de asignación de capital asignando posturas (stakes)
    óptimas entre N apuestas simultáneas considerando retornos esperados,
    varianzas, matriz de covarianza y restricciones operativas.
    """

    def __init__(self, risk_aversion: float = 2.0, max_stake_pct: float = 0.03, max_daily_portfolio_pct: float = 0.15):
        """
        :param risk_aversion: Parámetro lambda de aversión al riesgo (lambda >= 0).
                              1.0 equivale a la utilidad logarítmica (Kelly),
                              2.0 es conservador institucional (reducción de drawdown).
        :param max_stake_pct: Apuesta máxima por activo/partido (ej. 3% del bankroll).
        :param max_daily_portfolio_pct: Presupuesto total de riesgo diario (ej. 15% del bankroll).
        """
        self.risk_aversion = float(risk_aversion)
        self.max_stake_pct = float(max_stake_pct)
        self.max_daily_portfolio_pct = float(max_daily_portfolio_pct)

    @staticmethod
    def build_bernoulli_correlation_matrix(candidates: List[Dict]) -> np.ndarray:
        """
        Construye la matriz de correlación analítica exacta entre activos compuestos
        de Bernoulli (Singles, Dutching y Parlays de 2 o 3 patas) cuando comparten partidos
        en la misma jornada:
            Cov(R_A, R_B) = O_A * O_B * P_A * P_B * (1 / P_{A cap B} - 1)
            rho_{A,B} = sqrt((P_A * P_B) / ((1 - P_A) * (1 - P_B))) * (1 / P_{A cap B} - 1)
        """
        n = len(candidates)
        corr_matrix = np.eye(n, dtype=np.float64)
        if n <= 1:
            return corr_matrix

        for i in range(n):
            set_a = candidates[i].get('match_indices')
            if not set_a:
                continue
            p_a = float(np.clip(candidates[i].get('prob', 0.5), 1e-4, 1.0 - 1e-4))
            leg_probs_a = candidates[i].get('leg_probs', {})
            for j in range(i + 1, n):
                set_b = candidates[j].get('match_indices')
                if not set_b:
                    continue
                shared = set_a.intersection(set_b)
                if not shared:
                    continue
                p_b = float(np.clip(candidates[j].get('prob', 0.5), 1e-4, 1.0 - 1e-4))
                leg_probs_b = candidates[j].get('leg_probs', {})
                p_shared = 1.0
                for m_idx in shared:
                    p_m = leg_probs_a.get(m_idx, leg_probs_b.get(m_idx, max(p_a, p_b)))
                    p_shared *= float(np.clip(p_m, 1e-4, 1.0 - 1e-4))
                if p_shared < 1.0:
                    denom = np.sqrt(max((1.0 - p_a) * (1.0 - p_b), 1e-8))
                    num = np.sqrt(p_a * p_b) * ((1.0 / p_shared) - 1.0)
                    raw_rho = float(np.clip(num / denom, 0.0, 0.90))
                    # Ponderación estructural de solapamiento de patas (permite cobertura escalonada Dutching + Parlay)
                    overlap_cap = 0.70 * (len(shared) / np.sqrt(len(set_a) * len(set_b)))
                    rho = min(raw_rho, float(overlap_cap))
                else:
                    rho = 0.0
                corr_matrix[i, j] = corr_matrix[j, i] = rho

        return corr_matrix

    @staticmethod
    def compute_asset_covariance(prob_vec: np.ndarray, odds_vec: np.ndarray, corr_matrix: Optional[np.ndarray] = None, shrinkage: float = 0.1) -> np.ndarray:
        """
        Calcula la matriz de covarianza de los retornos de las apuestas.
        Var(R_i) = p_i * (1 - p_i) * (odds_i)^2
        Cov(R_i, R_j) = corr_ij * std_i * std_j
        Aplica regularización/shrinkage (Ledoit-Wolf style) para evitar inestabilidad.
        """
        n = len(prob_vec)
        prob_arr = np.clip(np.asarray(prob_vec, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        odds_arr = np.asarray(odds_vec, dtype=np.float64)

        # Desviación estándar del retorno por unidad apostada
        std_vec = np.sqrt(prob_arr * (1.0 - prob_arr)) * odds_arr

        if corr_matrix is None or corr_matrix.shape != (n, n):
            corr_matrix = np.eye(n)

        cov = np.outer(std_vec, std_vec) * corr_matrix

        # Regularización Ridge / Shrinkage de covarianza
        diag_cov = np.diag(np.diag(cov))
        cov_reg = (1.0 - shrinkage) * cov + shrinkage * diag_cov + np.eye(n) * 1e-4
        return cov_reg

    def optimize_mean_variance(
        self,
        ev_vec: List[float],
        odds_vec: List[float],
        prob_vec: List[float],
        corr_matrix: Optional[np.ndarray] = None,
        bankroll: float = 10000.0,
        uncertainty_penalties: Optional[List[float]] = None,
        liquidity_caps: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Resuelve la Optimización Media-Varianza de Markowitz (1952):
        Max U(w) = w^T * mu - (lambda / 2) * w^T * Sigma * w
        sujeto a:
          0 <= w_i <= min(max_stake_pct, liquidity_cap_i / bankroll)
          sum(w_i) <= max_daily_portfolio_pct
        """
        n = len(ev_vec)
        if n == 0:
            return {
                'weights': np.array([]),
                'stakes': np.array([]),
                'expected_return': 0.0,
                'variance': 0.0,
                'std_dev': 0.0,
                'sharpe': 0.0,
                'converged': True
            }

        ev_arr = np.maximum(np.asarray(ev_vec, dtype=np.float64), 0.0)
        odds_arr = np.asarray(odds_vec, dtype=np.float64)
        prob_arr = np.asarray(prob_vec, dtype=np.float64)

        # Ajuste bayesiano de EV por penalizaciones de incertidumbre
        if uncertainty_penalties is not None:
            penalties = np.clip(np.asarray(uncertainty_penalties, dtype=np.float64), 0.0, 1.0)
            mu = ev_arr * penalties
        else:
            mu = ev_arr

        cov_matrix = self.compute_asset_covariance(prob_arr, odds_arr, corr_matrix)

        # Definir límites por activo (Box Constraints sin topes arbitrarios contra favoritos/Dutching)
        bounds = []
        for i in range(n):
            upper_bound = self.max_stake_pct
            if liquidity_caps is not None and bankroll > 0:
                upper_bound = min(upper_bound, liquidity_caps[i] / bankroll)
            bounds.append((0.0, max(0.0, upper_bound)))

        # Restricción de presupuesto global de portafolio diario: sum(w_i) <= max_daily_portfolio_pct
        constraints = [
            {'type': 'ineq', 'fun': lambda w: self.max_daily_portfolio_pct - np.sum(w)}
        ]

        # Función objetivo a minimizar: - (w^T * mu - 0.5 * lambda * w^T * Sigma * w)
        def objective(w):
            port_return = np.dot(w, mu)
            port_variance = np.dot(w, np.dot(cov_matrix, w))
            utility = port_return - 0.5 * self.risk_aversion * port_variance
            return -utility

        # Punto de partida uniforme
        w0 = np.full(n, min(self.max_stake_pct, self.max_daily_portfolio_pct / max(n, 1)))

        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-7, 'maxiter': 500}
        )

        weights = np.maximum(res.x if res.success else np.zeros(n), 0.0)
        # Limpieza de pequeños errores numéricos
        weights[weights < 1e-4] = 0.0

        stakes = weights * bankroll
        port_return = float(np.dot(weights, mu))
        port_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
        port_std = float(np.sqrt(max(port_var, 1e-8)))
        sharpe = float(port_return / port_std) if port_std > 1e-6 else 0.0

        return {
            'weights': weights,
            'stakes': stakes,
            'expected_return': port_return,
            'variance': port_var,
            'std_dev': port_std,
            'sharpe': sharpe,
            'converged': res.success
        }
