"""Train all models on historical match data."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from wcp.data import load_matches
from wcp.features import build_training_frame
from wcp.models.ensemble import EnsemblePredictor
from wcp.priors import apply_elo_priors

console = Console()


def train(verbose: bool = True) -> EnsemblePredictor:
    matches = load_matches()
    elo_for_features = __import__("wcp.models.elo", fromlist=["EloRatings"]).EloRatings()
    train_df = build_training_frame(matches, elo_for_features)

    predictor = EnsemblePredictor()
    predictor.fit(matches, train_df)
    apply_elo_priors(predictor.elo)
    predictor.save()

    if verbose:
        console.print(f"[green]Trained on {len(matches)} matches[/green]")
        table = Table(title="Top 15 Teams by Elo Rating")
        table.add_column("Rank", style="dim")
        table.add_column("Team")
        table.add_column("Elo", justify="right")
        table.add_column("Attack", justify="right")
        table.add_column("Defense", justify="right")

        for i, (team, elo, att, deff) in enumerate(predictor.ratings_table()[:15], 1):
            table.add_row(str(i), team, f"{elo:.0f}", f"{att:.3f}", f"{deff:.3f}")
        console.print(table)

    return predictor


def main():
    parser = argparse.ArgumentParser(description="Train World Cup predictor models")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    train(verbose=not args.quiet)


if __name__ == "__main__":
    main()
