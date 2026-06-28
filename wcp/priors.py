"""Seed Elo priors for 2026 World Cup participants."""

from wcp.config import WORLD_CUP_2026_GROUPS

# Approximate pre-tournament Elo (derived from FIFA rankings + recent form, Jun 2026)
TEAM_ELO_PRIORS: dict[str, float] = {
    "France": 2050, "Spain": 2000, "Argentina": 1990, "England": 1980,
    "Brazil": 1970, "Portugal": 1950, "Netherlands": 1940, "Germany": 1930,
    "Belgium": 1900, "Croatia": 1880, "Italy": 1870, "Colombia": 1860,
    "Uruguay": 1850, "Morocco": 1840, "Switzerland": 1830, "Japan": 1820,
    "Mexico": 1810, "United States": 1800, "Senegal": 1790, "Denmark": 1780,
    "Austria": 1770, "Turkey": 1760, "Ecuador": 1750, "Norway": 1740,
    "Paraguay": 1730, "Egypt": 1720, "Scotland": 1710, "South Korea": 1700,
    "Australia": 1690, "Iran": 1680, "Algeria": 1670, "Ivory Coast": 1660,
    "Tunisia": 1650, "Sweden": 1640, "Czech Republic": 1630, "Poland": 1620,
    "Ukraine": 1610, "Serbia": 1600, "Canada": 1590, "Panama": 1580,
    "Ghana": 1570, "Saudi Arabia": 1560, "Qatar": 1550, "Jordan": 1540,
    "Iraq": 1530, "South Africa": 1520, "DR Congo": 1510, "Uzbekistan": 1500,
    "Cape Verde": 1490, "Bosnia and Herzegovina": 1480, "New Zealand": 1470,
    "Haiti": 1400, "Curaçao": 1380,
}

# Approximate squad market value (millions EUR, Transfermarkt-style estimates)
TEAM_SQUAD_VALUE: dict[str, float] = {
    "France": 1250, "England": 1180, "Spain": 1100, "Brazil": 1050,
    "Argentina": 980, "Germany": 920, "Portugal": 880, "Netherlands": 850,
    "Belgium": 780, "Italy": 750, "Colombia": 420, "Uruguay": 400,
    "Croatia": 380, "Morocco": 350, "Mexico": 340, "United States": 320,
    "Switzerland": 310, "Japan": 300, "Senegal": 280, "Austria": 270,
    "Turkey": 260, "Ecuador": 250, "Norway": 240, "Denmark": 230,
    "Paraguay": 200, "Egypt": 190, "Scotland": 180, "South Korea": 175,
    "Australia": 170, "Iran": 160, "Algeria": 155, "Ivory Coast": 150,
    "Tunisia": 140, "Sweden": 135, "Czech Republic": 130, "Canada": 125,
    "Panama": 90, "Ghana": 85, "Saudi Arabia": 80, "Qatar": 75,
    "Jordan": 70, "Iraq": 65, "South Africa": 60, "DR Congo": 55,
    "Uzbekistan": 50, "Cape Verde": 45, "Bosnia and Herzegovina": 42,
    "New Zealand": 35, "Haiti": 25, "Curaçao": 20,
}


def all_wc_teams() -> list[str]:
    teams = []
    for group in WORLD_CUP_2026_GROUPS.values():
        teams.extend(group)
    return teams


def apply_elo_priors(elo_ratings) -> None:
    """Overlay trained Elo with priors for teams that have sparse match history."""
    for team, prior in TEAM_ELO_PRIORS.items():
        trained = elo_ratings.ratings.get(team, elo_ratings.initial)
        # Blend: trust training where we have data, otherwise use prior
        n_implied = max(0, (trained - elo_ratings.initial) / 5)  # rough match count proxy
        weight = min(n_implied / 20, 1.0)
        elo_ratings.ratings[team] = weight * trained + (1 - weight) * prior

    for team in all_wc_teams():
        if team not in elo_ratings.ratings:
            elo_ratings.ratings[team] = TEAM_ELO_PRIORS.get(team, elo_ratings.initial)
