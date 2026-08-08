# core/league_mapping.py

# Definición canónica de ligas y sus alias
CANONICAL_LEAGUES = {
    # Ligas Europeas Principales
    'E0': ['Premier League'],
    'SP1': ['La Liga', 'La Liga EA Sports'],
    'D1': ['1. Bundesliga', 'Bundesliga'],
    'I1': ['Serie A', 'Serie A Enilive'],
    'F1': ['Ligue 1', 'Ligue 1 McDonald\'s'],
    'E1': ['Championship', 'EFL Championship'],
    'SP2': ['La Liga 2', 'La Liga Hypermotion'],
    'D2': ['2. Bundesliga'],
    'F2': ['Ligue 2'],
    'I2': ['Serie B', 'Serie BKT'],
    'B1': ['Jupiler Pro League'],
    'N1': ['Eredivisie'],
    'P1': ['Primeira Liga', 'Liga Portugal Betclic'],
    'SC0': ['Scottish Premiership'],
    'T1': ['Süper Lig', 'Trendyol Süper Lig'],
    'G1': ['Super League', 'Stoiximan Super League'],
    'E2': ['League One', 'EFL League One'],
    # América & Internacionales
    'MLS': ['Major League Soccer', 'USA'],
    'MEX': ['Liga MX', 'MEX1'],
    'J1': ['J-League 1', 'Meiji Yasuda J1 League', 'JPN'],
    'SWE': ['Allsvenskan'],
    'NOR': ['Eliteserien'],
    'DNK': ['Superligaen', '3F Superliga'],
    'SWZ': ['Swiss Super League', 'Credit Suisse Super League'],
    'AUT': ['Austrian Bundesliga', 'Admiral Bundesliga'],
    # Torneos Internacionales UEFA / FIFA
    'CL': ['Champions League', 'UEFA Champions League'],
    'EL': ['Europa League', 'UEFA Europa League'],
    'WC': ['FIFA World Cup']
}

# Diccionario invertido para búsqueda O(1) de cualquier alias al código canónico
COMPETITION_MAPPING = {}
for canonical, aliases in CANONICAL_LEAGUES.items():
    COMPETITION_MAPPING[canonical] = canonical
    for alias in aliases:
        COMPETITION_MAPPING[alias] = canonical

WHITELIST_LEAGUES = ['MEX']

def normalize_league(comp):
    """Devuelve el ID canónico de la liga si existe en el mapeo, de lo contrario devuelve el nombre original."""
    if not isinstance(comp, str):
        return comp
    return COMPETITION_MAPPING.get(comp, comp)

def get_param_by_comp(param_dict, comp, default_val=0.015):
    """
    Búsqueda exhaustiva y bidireccional de parámetros por competición:
    1. Coincidencia exacta con comp.
    2. Coincidencia con el código canónico normalize_league(comp).
    3. Búsqueda en todos los alias asociados a la liga.
    4. Búsqueda por mapeo inverso.
    5. Fallback a 'DEFAULT' o default_val.
    """
    if not param_dict or not isinstance(param_dict, dict):
        return default_val

    # 1. Búsqueda directa
    if comp in param_dict:
        return param_dict[comp]

    # 2. Búsqueda por código canónico
    canonical = normalize_league(comp)
    if canonical in param_dict:
        return param_dict[canonical]

    # 3. Búsqueda en los alias del código canónico
    aliases = CANONICAL_LEAGUES.get(canonical, [])
    for alias in aliases:
        if alias in param_dict:
            return param_dict[alias]

    # 4. Búsqueda inversa general por si comp era un alias o canónico
    for c_key, c_aliases in CANONICAL_LEAGUES.items():
        if comp == c_key or comp in c_aliases or canonical == c_key:
            if c_key in param_dict:
                return param_dict[c_key]
            for a in c_aliases:
                if a in param_dict:
                    return param_dict[a]

    # 5. Fallback
    return param_dict.get('DEFAULT', default_val)
