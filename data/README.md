# Data Dictionary

## `matches.csv`

Historical international football matches used to train and evaluate the prediction models.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Match date (ISO 8601) |
| `home_team` | string | Home team name (canonical) |
| `away_team` | string | Away team name (canonical) |
| `home_goals` | int | Goals scored by home team |
| `away_goals` | int | Goals scored by away team |
| `tournament` | string | Competition (`World Cup`, `Euro`, `Nations League`, `International`) |
| `neutral` | int | `1` if neutral venue, `0` if home advantage applies |

## Coverage

- **World Cup knockout & key group matches** (1990–2022)
- **UEFA Euro 2024** (full knockout stage)
- **UEFA Nations League 2025** (recent competitive fixtures)
- **International friendlies** (2023–2025, top nations)

## Regenerating

```bash
python scripts/generate_matches.py
```

## Notes

- Team names are normalized in `wcp/data.py` (e.g. `USA` → `United States`, `Czechia` → `Czech Republic`)
- All World Cup matches are played at neutral venues (`neutral = 1`)
- Dataset is intentionally compact for portfolio reproducibility; production use would ingest from [Kaggle International Football](https://www.kaggle.com/datasets/patateriedata/international-football-results) or similar APIs
