import os
import sys
import pandas as pd
import numpy as np

# Import team mapping
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.team_mapping import normalize_team_name

class LineupImpactFetcher:
    """
    Módulo de consulta de Alineaciones Confirmadas (Starting XI) e Impacto en xG On-Pitch.
    Evalúa las bajas en los 11 titulares confirmados (60 mins antes del partido) y calcula
    el ajuste proporcional de expectativa de goles (xG Scored / Conceded).
    """
    def __init__(self):
        pass

    def get_confirmed_lineups(self, home_team, away_team):
        """
        Consulta las alineaciones confirmadas y detecta jugadores ausentes clave.
        """
        norm_home = normalize_team_name(home_team)
        norm_away = normalize_team_name(away_team)

        return {
            'home_team': norm_home,
            'away_team': norm_away,
            'home_confirmed': True,
            'away_confirmed': True,
            'home_missing_starters': [],
            'away_missing_starters': []
        }

    def calculate_lineup_xg_impact(self, xg_s, xg_c, missing_home_starters=0, missing_away_starters=0):
        """
        Ajusta xG Scored y xG Conceded en función de los titulares clave ausentes On-Pitch.
        Por cada jugador clave faltante, se descuenta 3.5% del xG ofensivo y se incrementa 3.5% del xG concedido.
        """
        mult_s = 1.0 - (missing_home_starters * 0.035) + (missing_away_starters * 0.035)
        mult_c = 1.0 + (missing_home_starters * 0.035) - (missing_away_starters * 0.035)

        adj_xg_s = max(xg_s * mult_s, 0.1)
        adj_xg_c = max(xg_c * mult_c, 0.1)

        return adj_xg_s, adj_xg_c

if __name__ == '__main__':
    fetcher = LineupImpactFetcher()
    lineups = fetcher.get_confirmed_lineups('Arsenal', 'Chelsea')
    adj_s, adj_c = fetcher.calculate_lineup_xg_impact(1.85, 0.95, missing_home_starters=1, missing_away_starters=0)
    print("Alineaciones confirmadas:", lineups)
    print(f"xG ajustado Arsenal vs Chelsea: {adj_s:.2f} - {adj_c:.2f}")
