"""Model evaluation and backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from wcp.data import load_matches
from wcp.features import FEATURE_COLS, build_training_frame, match_features, match_outcome
from wcp.models.elo import EloRatings
from wcp.models.ensemble import EnsemblePredictor


def evaluate_models(matches: pd.DataFrame | None = None, test_from_year: int = 2018) -> pd.DataFrame:
    matches = matches if matches is not None else load_matches()
    train = matches[matches["date"].dt.year < test_from_year].copy()
    test = matches[matches["date"].dt.year >= test_from_year].copy()

    if test.empty:
        raise ValueError(f"No test matches found from year {test_from_year}")

    elo_feat = EloRatings()
    train_df, team_store = build_training_frame(train, elo_feat)

    predictor = EnsemblePredictor(team_store=team_store)
    predictor.elo.fit(train)
    predictor.dixon_coles.fit(train)

    gbm = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
    gbm.fit(train_df[FEATURE_COLS], train_df["outcome"])
    predictor.lightgbm.model = gbm  # type: ignore[assignment]

    def _gbm_probs(h, a, n):
        feats = match_features(h, a, predictor.elo, team_store, neutral=n, is_wc=True)
        probs = gbm.predict_proba(feats)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    rows = []
    for name, prob_fn in [
        ("Elo", lambda h, a, n: predictor.elo.win_prob(h, a, neutral=n)),
        ("Dixon-Coles", lambda h, a, n: predictor.dixon_coles.win_prob(h, a, neutral=n)),
        ("Gradient Boosting", _gbm_probs),
        ("Ensemble", lambda h, a, n: predictor.match_prob(h, a, neutral=n, is_wc=True)),
    ]:
        y_true, y_prob, y_pred = [], [], []
        for _, m in test.iterrows():
            neutral = bool(m.get("neutral", 0))
            outcome = match_outcome(m["home_goals"], m["away_goals"])
            probs = prob_fn(m["home_team"], m["away_team"], neutral)
            y_true.append(outcome)
            y_prob.append(probs)
            y_pred.append(int(np.argmax(probs)))

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        rows.append({
            "model": name,
            "n_matches": len(test),
            "accuracy": accuracy_score(y_true_arr, y_pred),
            "log_loss": log_loss(y_true_arr, y_prob_arr, labels=[0, 1, 2]),
        })

    return pd.DataFrame(rows).sort_values("log_loss")


def dataset_summary(matches: pd.DataFrame | None = None) -> dict:
    matches = matches if matches is not None else load_matches()
    return {
        "total_matches": len(matches),
        "date_range": f"{matches['date'].min():%Y-%m-%d} → {matches['date'].max():%Y-%m-%d}",
        "teams": len(set(matches["home_team"]) | set(matches["away_team"])),
        "tournaments": matches["tournament"].value_counts().to_dict(),
        "avg_goals_per_match": round((matches["home_goals"] + matches["away_goals"]).mean(), 2),
        "draw_rate": round((matches["home_goals"] == matches["away_goals"]).mean(), 3),
    }
