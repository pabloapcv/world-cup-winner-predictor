"""Load and normalize historical international match data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wcp.config import DATA_DIR

# Canonical team name mapping
TEAM_ALIASES: dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Korea Rep": "South Korea",
    "Korea Republic (South Korea)": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cote D'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Czechia": "Czech Republic",
    "Czech Rep": "Czech Republic",
    "Congo DR": "DR Congo",
    "Congo, DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Curacao": "Curaçao",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Ivory Coast",
    "Holland": "Netherlands",
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "USSR": "Russia",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
}


def normalize_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def load_matches(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_world_cup_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tournament"] == "World Cup"].copy()


def get_all_teams(df: pd.DataFrame) -> set[str]:
    return set(df["home_team"]) | set(df["away_team"])
