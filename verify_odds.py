import pandas as pd

df = pd.read_parquet('data/processed/matches_with_odds.parquet', engine='fastparquet')
leagues = ['MLS', 'Liga MX', 'J1', 'Swiss Super League', 'Eliteserien', 'Allsvenskan', 'Austrian Bundesliga', 'Superligaen', 'SP1', 'E0', 'D1', 'I1']
print(f"{'Competicion':<22} | {'Total':<6} | {'Open Odds Valid':<16} | {'Close Odds Valid':<16}")
print("-" * 70)
for comp in leagues:
    df_c = df[df['competition'] == comp]
    total = len(df_c)
    open_v = (df_c['open_odds_win'] > 1.01).sum()
    close_v = (df_c['odds_win'] > 1.01).sum()
    print(f"{comp:<22} | {total:<6} | {open_v:<16} | {close_v:<16}")
