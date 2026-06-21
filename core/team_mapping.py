# Diccionario para normalizar nombres de equipos de diferentes fuentes
# Llave: Nombre en Football-Data u otra fuente. Valor: Nombre estándar (usualmente el de StatsBomb o el más común)

TEAM_NAME_MAPPING = {
    # Premier League
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Aston Villa": "Aston Villa",
    "Nott'm Forest": "Nottingham Forest",
    "Luton": "Luton Town",
    "Sheffield United": "Sheffield United",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Burnley": "Burnley",
    
    # La Liga
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Ath Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Girona": "Girona",
    "Ath Bilbao": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    "Sociedad": "Real Sociedad",
    "Real Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano",
    "Villarreal": "Villarreal",
    "Valencia": "Valencia",
    "Getafe": "Getafe",
    "Alaves": "Deportivo Alaves",
    "Osasuna": "CA Osasuna",
    "Las Palmas": "UD Las Palmas",
    "Mallorca": "RCD Mallorca",
    "Celta": "Celta Vigo",
    "Sevilla": "Sevilla",
    "Cadiz": "Cadiz",
    "Granada": "Granada",
    "Almeria": "Almeria",

    # Serie A
    "Inter": "Internazionale",
    "Milan": "AC Milan",
    "Juventus": "Juventus",
    "Bologna": "Bologna",
    "Roma": "AS Roma",
    "Atalanta": "Atalanta",
    "Napoli": "Napoli",
    "Fiorentina": "Fiorentina",
    "Lazio": "Lazio",
    "Torino": "Torino",
    "Monza": "Monza",
    "Genoa": "Genoa",
    "Lecce": "Lecce",
    "Empoli": "Empoli",
    "Udinese": "Udinese",
    "Verona": "Hellas Verona",
    "Cagliari": "Cagliari",
    "Frosinone": "Frosinone",
    "Sassuolo": "Sassuolo",
    "Salernitana": "Salernitana",

    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Leverkusen": "Bayer Leverkusen",
    "Stuttgart": "VfB Stuttgart",
    "RB Leipzig": "RB Leipzig",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG Hoffenheim",
    "Augsburg": "FC Augsburg",
    "Heidenheim": "1. FC Heidenheim 1846",
    "Werder Bremen": "Werder Bremen",
    "M'gladbach": "Borussia Monchengladbach",
    "Bochum": "VfL Bochum",
    "Wolfsburg": "VfL Wolfsburg",
    "Union Berlin": "Union Berlin",
    "Mainz": "FSV Mainz 05",
    "FC Koln": "FC Cologne",
    "Darmstadt": "SV Darmstadt 98",
    
    # Ligue 1
    "Paris SG": "Paris Saint Germain",
    "PSG": "Paris Saint Germain",
    "Brest": "Brest",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Nice": "Nice",
    "Lens": "Lens",
    "Rennes": "Rennes",
    "Marseille": "Marseille",
    "Reims": "Reims",
    "Lyon": "Lyon",
    "Toulouse": "Toulouse",
    "Montpellier": "Montpellier",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Le Havre": "Le Havre",
    "Metz": "Metz",
    "Lorient": "Lorient",
    "Clermont": "Clermont Foot"
}

def normalize_team_name(team_name):
    """
    Normaliza el nombre de un equipo usando el diccionario TEAM_NAME_MAPPING.
    Si el equipo no está en el diccionario, se devuelve el nombre original (con espacios en los extremos quitados).
    """
    if not isinstance(team_name, str):
        return team_name
    
    team_name = team_name.strip()
    return TEAM_NAME_MAPPING.get(team_name, team_name)
