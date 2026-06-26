"""Visualization utilities for portfolio outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wcp.config import FIGURES_DIR, WORLD_CUP_2026_GROUPS
from wcp.models.ensemble import EnsemblePredictor

# Portfolio-friendly style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#1a535c", "#4ecdc4", "#ff6b6b", "#ffe66d", "#95e1d3"]


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_win_probabilities(
    results: dict[str, float],
    predictor: EnsemblePredictor,
    top_n: int = 15,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "win_probabilities.png")
    sorted_results = sorted(results.items(), key=lambda x: -x[1])[:top_n]
    teams = [t for t, _ in sorted_results]
    probs = [p * 100 for _, p in sorted_results]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[0] if i == 0 else PALETTE[1] if i < 3 else "#7f8c8d" for i in range(len(teams))]
    bars = ax.barh(teams[::-1], probs[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Win Probability (%)", fontsize=12)
    ax.set_title("2026 FIFA World Cup — Simulated Title Probabilities", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(probs) * 1.25)

    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_elo_rankings(
    predictor: EnsemblePredictor,
    top_n: int = 20,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "elo_rankings.png")
    ratings = predictor.ratings_table()[:top_n]
    teams = [r[0] for r in ratings]
    elos = [r[1] for r in ratings]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(teams[::-1], elos[::-1], color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Elo Rating", fontsize=12)
    ax.set_title("Team Strength Rankings (Elo)", fontsize=14, fontweight="bold", pad=12)
    ax.axvline(1500, color="#e74c3c", linestyle="--", alpha=0.6, label="Baseline (1500)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_attack_defense(
    predictor: EnsemblePredictor,
    top_n: int = 20,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "attack_defense.png")
    ratings = predictor.ratings_table()[:top_n]
    df = pd.DataFrame(ratings, columns=["team", "elo", "attack", "defense"])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["attack"], df["defense"],
        s=df["elo"] / 5, c=df["elo"], cmap="viridis", alpha=0.75, edgecolors="white", linewidth=0.5
    )
    for _, row in df.iterrows():
        ax.annotate(row["team"], (row["attack"], row["defense"]), fontsize=8, alpha=0.85)
    ax.set_xlabel("Attack Strength (Dixon-Coles)", fontsize=12)
    ax.set_ylabel("Defense Strength (Dixon-Coles)", fontsize=12)
    ax.set_title("Attack vs Defense Profile", fontsize=14, fontweight="bold", pad=12)
    plt.colorbar(scatter, label="Elo Rating")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_goals_distribution(
    matches: pd.DataFrame,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "goals_distribution.png")
    total_goals = matches["home_goals"] + matches["away_goals"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(total_goals, bins=range(0, 13), ax=axes[0], color=PALETTE[0], edgecolor="white")
    axes[0].set_title("Total Goals per Match")
    axes[0].set_xlabel("Goals")

    wc = matches[matches["tournament"] == "World Cup"]
    by_year = wc.groupby(wc["date"].dt.year).apply(
        lambda g: (g["home_goals"] + g["away_goals"]).mean(), include_groups=False
    )
    axes[1].plot(by_year.index, by_year.values, marker="o", color=PALETTE[2], linewidth=2)
    axes[1].set_title("World Cup Avg Goals by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Avg Goals")

    fig.suptitle("Exploratory Analysis — Scoring Patterns", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_model_comparison(
    eval_df: pd.DataFrame,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "model_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = [PALETTE[3] if m == "Ensemble" else PALETTE[1] for m in eval_df["model"]]
    axes[0].barh(eval_df["model"], eval_df["accuracy"], color=colors, edgecolor="white")
    axes[0].set_title("Out-of-Time Accuracy (2018+)")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_xlim(0, 1)

    axes[1].barh(eval_df["model"], eval_df["log_loss"], color=colors, edgecolor="white")
    axes[1].set_title("Log Loss (lower is better)")
    axes[1].invert_xaxis()

    fig.suptitle("Model Backtesting Results", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_group_strength(
    predictor: EnsemblePredictor,
    save_path: Path | None = None,
) -> Path:
    save_path = _ensure_dir(save_path or FIGURES_DIR / "group_strength.png")
    rows = []
    for group, teams in WORLD_CUP_2026_GROUPS.items():
        avg_elo = np.mean([predictor.elo.get(t) for t in teams])
        rows.append({"group": f"Group {group}", "avg_elo": avg_elo, "teams": ", ".join(teams)})

    df = pd.DataFrame(rows).sort_values("avg_elo", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["group"], df["avg_elo"], color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Average Elo Rating")
    ax.set_title("2026 World Cup — Group Strength (by avg Elo)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
