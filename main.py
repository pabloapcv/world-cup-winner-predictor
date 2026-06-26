#!/usr/bin/env python3
"""World Cup Winner Predictor — CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from wcp.priors import apply_elo_priors
from wcp.config import DEFAULT_SIMULATIONS, MODELS_DIR, WORLD_CUP_2026_GROUPS
from wcp.models.ensemble import EnsemblePredictor
from wcp.simulation import simulate_tournament
from wcp.train import train

console = Console()


def predict(n_sims: int = DEFAULT_SIMULATIONS, retrain: bool = False) -> dict[str, float]:
    model_path = MODELS_DIR / "elo.joblib"
    if retrain or not model_path.exists():
        console.print("[yellow]Training models...[/yellow]")
        predictor = train(verbose=False)
    else:
        predictor = EnsemblePredictor.load()
        apply_elo_priors(predictor.elo)

    console.print(Panel(
        f"[bold]FIFA World Cup 2026[/bold] — Monte Carlo Simulation\n"
        f"Simulations: [cyan]{n_sims:,}[/cyan] | "
        f"Teams: [cyan]48[/cyan] | "
        f"Format: 12 groups → R32 knockout",
        title="🏆 World Cup Winner Predictor",
        border_style="green",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Simulating {n_sims:,} tournaments...", total=None)
        results = simulate_tournament(predictor, n_sims=n_sims, seed=42)
    _display_results(results, predictor)
    return results


def _display_results(results: dict[str, float], predictor: EnsemblePredictor):
    sorted_results = sorted(results.items(), key=lambda x: -x[1])

    table = Table(title="🏆 World Cup 2026 Win Probabilities")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Team", style="bold")
    table.add_column("Win %", justify="right")
    table.add_column("Bar", width=30)
    table.add_column("Elo", justify="right")

    for i, (team, prob) in enumerate(sorted_results[:20], 1):
        bar_len = int(prob * 300)
        bar = "█" * bar_len
        elo = predictor.elo.get(team)
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        table.add_row(f"{medal}{i}", team, f"{prob*100:.2f}%", bar, f"{elo:.0f}")

    console.print(table)

    groups_table = Table(title="Group Stage Draw")
    groups_table.add_column("Group")
    groups_table.add_column("Teams")
    for g, teams in sorted(WORLD_CUP_2026_GROUPS.items()):
        groups_table.add_row(f"Group {g}", " · ".join(teams))
    console.print(groups_table)

    winner = sorted_results[0]
    console.print(Panel(
        f"[bold green]{winner[0]}[/bold green] is the predicted favorite "
        f"with a [cyan]{winner[1]*100:.1f}%[/cyan] chance of winning the 2026 World Cup.",
        title="Prediction",
        border_style="gold1",
    ))


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Winner Predictor")
    sub = parser.add_subparsers(dest="command", required=False)

    train_p = sub.add_parser("train", help="Train models only")
    train_p.set_defaults(command="train")

    pred_p = sub.add_parser("predict", help="Run tournament simulation")
    pred_p.add_argument("--train", action="store_true", help="Retrain before predicting")
    pred_p.add_argument("--sims", type=int, default=DEFAULT_SIMULATIONS, help="Number of simulations")
    pred_p.set_defaults(command="predict")

    args = parser.parse_args()
    command = getattr(args, "command", None) or "predict"

    if command == "train":
        train()
    else:
        predict(n_sims=getattr(args, "sims", DEFAULT_SIMULATIONS), retrain=getattr(args, "train", False))


if __name__ == "__main__":
    main()
