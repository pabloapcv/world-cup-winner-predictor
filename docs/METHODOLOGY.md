# Methodology

## Problem Formulation

Given historical international match results, estimate the probability that each of the 48 qualified nations wins the 2026 FIFA World Cup.

This decomposes into two sub-problems:

1. **Match outcome modeling** — P(home win), P(draw), P(away win) for any team pairing
2. **Tournament simulation** — propagate match-level probabilities through the full bracket via Monte Carlo methods

---

## Data

### Sources
- FIFA World Cup matches (1990–2022): knockout rounds and representative group fixtures
- UEFA Euro 2024: full knockout stage
- UEFA Nations League 2025 & recent internationals (2023–2025)

### Preprocessing
- Team name canonicalization (`USA` → `United States`, `Czechia` → `Czech Republic`)
- Chronological ordering for point-in-time feature computation (no data leakage)
- Neutral venue flag for World Cup / tournament matches

---

## Model 1: Elo Rating System

Dynamic ratings updated after each match:

```
E_new = E_old + K × (S - E_expected)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 40 | Learning rate |
| Home advantage | 65 Elo points | Applied when `neutral = 0` |
| Initial rating | 1500 | For unseen teams |

**Expected score:**
```
E_expected = 1 / (1 + 10^(-Δ/400))
```

**Draw modeling:** Base draw rate of 26%, decaying exponentially with rating gap:
```
P(draw) = 0.26 × exp(-|Δ| / 600)
```

---

## Model 2: Dixon-Coles Poisson Regression

Models goals as independent Poisson processes with team-specific attack (α) and defense (δ) parameters:

```
λ_home = exp(α_home - δ_away + γ)
λ_away = exp(α_away - δ_home)
```

### Low-score correction (τ)

Accounts for under-dispersion in 0-0, 0-1, 1-0, 1-1 results:

| Score | τ factor |
|-------|----------|
| 0-0 | 1 - λ_h × λ_a × ρ |
| 0-1 | 1 + λ_h × ρ |
| 1-0 | 1 + λ_a × ρ |
| 1-1 | 1 - ρ |

Default ρ = -0.13 (negative correlation between low scores).

### Time decay

Recent matches weighted more heavily:
```
w(t) = exp(-ξ × (t_max - t))
```
with ξ = 0.0018.

Parameters estimated via maximum likelihood (L-BFGS-B).

---

## Model 3: HistGradientBoosting Classifier

**Features:**
- `home_elo`, `away_elo`, `elo_diff`
- `neutral` (binary)
- `is_world_cup` (binary)

**Target:** 3-class outcome (home win / draw / away win)

**Training:**
- 100 boosting iterations, max depth 4
- Time-series cross-validation (3 folds)
- Isotonic calibration for probability outputs

---

## Ensemble

Weighted average of probability vectors:

```
P_ensemble = 0.25 × P_elo + 0.35 × P_dixon_coles + 0.40 × P_gbm
```

Weights reflect each model's empirical contribution; Dixon-Coles gets higher weight for goal-based simulation, GBM for non-linear patterns.

---

## Evaluation

### Out-of-time backtest
- **Train:** All matches before 2018
- **Test:** Matches from 2018 onward (World Cup 2018, 2022, Euro 2024, recent internationals)

### Metrics
- **Log loss** (primary): penalizes confident wrong predictions
- **Accuracy**: fraction of correct outcome class

---

## Tournament Simulation

### Group stage
- Round-robin within each of 12 groups (6 matches per group)
- Top 2 per group + 8 best third-place teams advance (32 total)

### Knockout
- Teams seeded by Elo rating
- Matches simulated via Dixon-Coles Poisson sampling
- Draws resolved by Elo-weighted penalty shootout

### Output
After N simulations (default 10,000):
```
P(team wins World Cup) = (times team won) / N
```

---

## Elo Priors for 2026 Teams

Teams with sparse match history receive blended ratings from FIFA ranking-derived priors (see `wcp/priors.py`), preventing weak teams from defaulting to the 1500 baseline.

---

## References

- Dixon & Coles (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*
- FIFA World Cup 2026 format: [fifa.com](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026)
