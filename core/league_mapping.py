# core/league_mapping.py

# Definición canónica de ligas y sus alias
CANONICAL_LEAGUES = {
    # Ligas Europeas Principales
    'E0': ['Premier League'],
    'SP1': ['La Liga'],
    'D1': ['1. Bundesliga'],
    'I1': ['Serie A'],
    'F1': ['Ligue 1'],
    'E1': ['Championship'],
    'SP2': ['La Liga 2'],
    'D2': ['2. Bundesliga'],
    'F2': ['Ligue 2'],
    'I2': ['Serie B'],
    'B1': ['Jupiler Pro League'],
    'N1': ['Eredivisie'],
    'P1': ['Primeira Liga'],
    'SC0': ['Scottish Premiership'],
    'T1': ['Süper Lig'],
    'G1': ['Super League'],
    'E2': ['League One'],
    # América & Internacionales
    'MLS': ['Major League Soccer', 'USA'],
    'MEX': ['Liga MX', 'MEX1'],
    'J1': ['J-League 1', 'JPN'],
    'SWE': ['Allsvenskan'],
    'NOR': ['Eliteserien'],
    'DNK': ['Superligaen'],
    'SWZ': ['Swiss Super League'],
    'AUT': ['Austrian Bundesliga'],
    # Torneos Internacionales UEFA / FIFA
    'CL': ['Champions League'],
    'EL': ['Europa League'],
    'WC': ['FIFA World Cup']
}

# Diccionario invertido para búsqueda O(1) de cualquier alias al código canónico
COMPETITION_MAPPING = {}
for canonical, aliases in CANONICAL_LEAGUES.items():
    COMPETITION_MAPPING[canonical] = canonical
    for alias in aliases:
        COMPETITION_MAPPING[alias] = canonical

WHITELIST_LEAGUES = list(CANONICAL_LEAGUES.keys())

def normalize_league(comp):
    """Devuelve el ID canónico de la liga si existe en el mapeo, de lo contrario devuelve el nombre original."""
    if not isinstance(comp, str):
        return comp
    return COMPETITION_MAPPING.get(comp, comp)
