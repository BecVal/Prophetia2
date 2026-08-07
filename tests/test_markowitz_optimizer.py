import os
import sys
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
core_dir = os.path.join(root_dir, 'core')
if core_dir not in sys.path:
    sys.path.append(core_dir)

from markowitz_optimizer import MarkowitzPortfolioOptimizer

def test_markowitz_empty():
    optimizer = MarkowitzPortfolioOptimizer()
    res = optimizer.optimize_mean_variance([], [], [])
    assert len(res['weights']) == 0
    assert res['expected_return'] == 0.0
    print("[OK] test_markowitz_empty OK")

def test_markowitz_single_asset():
    optimizer = MarkowitzPortfolioOptimizer(risk_aversion=2.0, max_stake_pct=0.03, max_daily_portfolio_pct=0.15)
    ev_vec = [0.08]     # 8% EV
    odds_vec = [2.10]   # Cuota 2.10
    prob_vec = [0.514]  # Probabilidad 51.4%
    
    res = optimizer.optimize_mean_variance(ev_vec, odds_vec, prob_vec, bankroll=10000.0)
    assert res['converged']
    assert 0.0 < res['weights'][0] <= 0.03
    assert res['stakes'][0] == res['weights'][0] * 10000.0
    assert res['expected_return'] > 0.0
    print(f"[OK] test_markowitz_single_asset OK (Weight: {res['weights'][0]*100:.2f}%, Stake: ${res['stakes'][0]:.2f})")

def test_markowitz_daily_budget_constraint():
    optimizer = MarkowitzPortfolioOptimizer(risk_aversion=0.5, max_stake_pct=0.05, max_daily_portfolio_pct=0.10)
    # 5 apuestas muy rentables
    ev_vec = [0.10, 0.12, 0.09, 0.11, 0.08]
    odds_vec = [2.0, 2.2, 1.9, 2.1, 2.0]
    prob_vec = [0.55, 0.51, 0.57, 0.53, 0.54]
    
    res = optimizer.optimize_mean_variance(ev_vec, odds_vec, prob_vec, bankroll=10000.0)
    assert res['converged']
    total_weight = np.sum(res['weights'])
    assert total_weight <= 0.10 + 1e-5  # No debe superar el 10% del bankroll en total
    print(f"[OK] test_markowitz_daily_budget_constraint OK (Total Weight: {total_weight*100:.2f}%)")

def test_markowitz_high_correlation_penalty():
    optimizer = MarkowitzPortfolioOptimizer(risk_aversion=2.0, max_stake_pct=0.03)
    ev_vec = [0.08, 0.08]
    odds_vec = [2.0, 2.0]
    prob_vec = [0.54, 0.54]
    
    # Caso 1: Apuestas independientes (corr = 0)
    corr_indep = np.array([[1.0, 0.0], [0.0, 1.0]])
    res_indep = optimizer.optimize_mean_variance(ev_vec, odds_vec, prob_vec, corr_matrix=corr_indep)
    
    # Caso 2: Apuestas altamente correlacionadas (corr = 0.9)
    corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
    res_corr = optimizer.optimize_mean_variance(ev_vec, odds_vec, prob_vec, corr_matrix=corr_high)
    
    # La suma de pesos con alta correlación debe ser menor para controlar la varianza del portafolio
    assert np.sum(res_corr['weights']) < np.sum(res_indep['weights'])
    print(f"[OK] test_markowitz_high_correlation_penalty OK (Indep Sum: {np.sum(res_indep['weights'])*100:.2f}%, High Corr Sum: {np.sum(res_corr['weights'])*100:.2f}%)")

if __name__ == '__main__':
    test_markowitz_empty()
    test_markowitz_single_asset()
    test_markowitz_daily_budget_constraint()
    test_markowitz_high_correlation_penalty()
    print("\nALL MARKOWITZ OPTIMIZER TESTS PASSED SUCCESSFULLY!")
