#!/usr/bin/env python3
"""Generate portfolio artifacts: figures, evaluation metrics, and prediction CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from rich.console import Console

from wcp.config import DEFAULT_SIMULATIONS, FIGURES_DIR, RESULTS_DIR
from wcp.data import load_matches
from wcp.evaluate import dataset_summary, evaluate_models
from wcp.models.ensemble import EnsemblePredictor
from wcp.priors import apply_elo_priors
from wcp.simulation import simulate_tournament
from wcp.train import train
from wcp import viz

console = Console()


def generate_report(n_sims: int = DEFAULT_SIMULATIONS, retrain: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]1/4[/bold] Loading data & training models...")
    matches = load_matches()
    predictor = train(verbose=False) if retrain else EnsemblePredictor.load()
    if not retrain:
        apply_elo_priors(predictor.elo)

    console.print("[bold]2/4[/bold] Running model evaluation...")
    eval_df = evaluate_models(matches)
    eval_df.to_csv(RESULTS_DIR / "model_evaluation.csv", index=False)

    summary = dataset_summary(matches)
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print("[bold]3/4[/bold] Simulating tournament...")
    results = simulate_tournament(predictor, n_sims=n_sims, seed=42)
    pred_df = pd.DataFrame([
        {"team": t, "win_probability": p, "elo": predictor.elo.get(t)}
        for t, p in results.items()
    ]).sort_values("win_probability", ascending=False)
    pred_df.to_csv(RESULTS_DIR / "predictions_2026.csv", index=False)

    console.print("[bold]4/4[/bold] Generating figures...")
    paths = [
        viz.plot_win_probabilities(results, predictor),
        viz.plot_elo_rankings(predictor),
        viz.plot_attack_defense(predictor),
        viz.plot_goals_distribution(matches),
        viz.plot_model_comparison(eval_df),
        viz.plot_group_strength(predictor),
    ]

    console.print("\n[green bold]Portfolio artifacts generated:[/green bold]")
    for p in paths:
        console.print(f"  📊 {p}")
    console.print(f"  📄 {RESULTS_DIR / 'predictions_2026.csv'}")
    console.print(f"  📄 {RESULTS_DIR / 'model_evaluation.csv'}")

    best = eval_df.iloc[0]
    top_team = pred_df.iloc[0]
    console.print(Panel_summary(eval_df, top_team, n_sims))


def Panel_summary(eval_df, top_team, n_sims):
    from rich.panel import Panel
    return Panel(
        f"Best model: [cyan]{eval_df.loc[eval_df['log_loss'].idxmin(), 'model']}[/cyan] "
        f"(log loss: {eval_df['log_loss'].min():.3f})\n"
        f"Predicted champion: [green]{top_team['team']}[/green] "
        f"({top_team['win_probability']*100:.1f}% over {n_sims:,} sims)",
        title="Summary",
        border_style="green",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--no-retrain", action="store_true")
    args = parser.parse_args()
    generate_report(n_sims=args.sims, retrain=not args.no_retrain)
