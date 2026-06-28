"""Streamlit dashboard for World Cup 2026 predictions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from wcp.config import RESULTS_DIR, WORLD_CUP_2026_GROUPS
from wcp.models.ensemble import EnsemblePredictor
from wcp.path_features import build_path_table
from wcp.priors import apply_elo_priors
from wcp.simulation import simulate_tournament_detailed, stage_probs_to_dataframe
from wcp.train import train

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA World Cup 2026 Winner Predictor")
st.caption(
    "Probabilistic tournament simulator using Elo, Dixon-Coles, gradient boosting, "
    "rolling form, squad value, and bracket-path features."
)


@st.cache_resource
def load_predictor():
    model_path = ROOT / "models" / "elo.joblib"
    if model_path.exists():
        predictor = EnsemblePredictor.load()
        apply_elo_priors(predictor.elo)
        return predictor
    return train(verbose=False)


predictor = load_predictor()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Champion Odds",
    "📊 Team Deep Dive",
    "⚔️ Match Calculator",
    "🗺️ Path Difficulty",
])

with tab1:
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)
    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Simulating {n_sims:,} tournaments..."):
            result = simulate_tournament_detailed(predictor, n_sims=n_sims)
            df = stage_probs_to_dataframe(result)
            df.to_csv(RESULTS_DIR / "predictions_2026.csv", index=False)
            st.session_state["sim_df"] = df
            st.session_state["n_sims"] = n_sims

    if "sim_df" in st.session_state:
        df = st.session_state["sim_df"]
        st.subheader("Title Probabilities")
        top = df.head(15)[["team", "win_probability"]].copy()
        top["win_probability"] = (top["win_probability"] * 100).round(2)
        top.columns = ["Team", "Win %"]
        st.dataframe(top, use_container_width=True, hide_index=True)
        st.bar_chart(top.set_index("Team"))

        st.subheader("Reach Each Stage")
        stage_cols = ["team", "r32", "r16", "quarter", "semi", "final", "champion"]
        stage_df = df[stage_cols].head(10).copy()
        for c in stage_cols[1:]:
            stage_df[c] = (stage_df[c] * 100).round(1)
        stage_df.columns = ["Team", "R32%", "R16%", "QF%", "SF%", "Final%", "Win%"]
        st.dataframe(stage_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run Simulation** to generate champion probabilities.")

with tab2:
    teams = sorted({t for g in WORLD_CUP_2026_GROUPS.values() for t in g})
    team = st.selectbox("Select team", teams)
    snap = predictor.team_store.get(team)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elo", f"{predictor.elo.get(team):.0f}")
    c2.metric("Form (last 10)", f"{snap.points_last10:.2f} pts/match")
    c3.metric("Squad Value", f"€{snap.squad_value:.0f}M")
    c4.metric("Win Rate (10)", f"{snap.win_rate_last10:.0%}")

    if "sim_df" in st.session_state and team in st.session_state["sim_df"]["team"].values:
        row = st.session_state["sim_df"][st.session_state["sim_df"]["team"] == team].iloc[0]
        st.subheader(f"{team} — Tournament Path Probabilities")
        stages = {
            "Round of 32": row["r32"],
            "Round of 16": row["r16"],
            "Quarterfinal": row["quarter"],
            "Semifinal": row["semi"],
            "Final": row["final"],
            "Champion": row["champion"],
        }
        path_df = pd.DataFrame({"Stage": list(stages.keys()), "Probability": list(stages.values())})
        path_df["Probability"] = (path_df["Probability"] * 100).round(1)
        st.bar_chart(path_df.set_index("Stage"))
    else:
        st.warning("Run a simulation on the Champion Odds tab first.")

with tab3:
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A (home)", teams, index=teams.index("France") if "France" in teams else 0)
    team_b = col2.selectbox("Team B (away)", teams, index=teams.index("Germany") if "Germany" in teams else 1)

    p_h, p_d, p_a = predictor.match_prob(team_a, team_b, neutral=True, is_wc=True)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{team_a} win", f"{p_h:.1%}")
    m2.metric("Draw", f"{p_d:.1%}")
    m3.metric(f"{team_b} win", f"{p_a:.1%}")

    st.subheader("Feature Comparison")
    from wcp.team_features import TeamFeatureStore
    store = predictor.team_store
    ha = store.get(team_a)
    hb = store.get(team_b)
    cmp_df = pd.DataFrame({
        "Metric": ["Elo", "Form (10)", "Goals scored (10)", "Goals conceded (10)", "Squad €M"],
        team_a: [predictor.elo.get(team_a), ha.points_last10, ha.goals_for_last10, ha.goals_against_last10, ha.squad_value],
        team_b: [predictor.elo.get(team_b), hb.points_last10, hb.goals_for_last10, hb.goals_against_last10, hb.squad_value],
    })
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Bracket & Group Path Difficulty")
    path_df = pd.DataFrame(build_path_table(predictor.elo.get))
    st.dataframe(
        path_df[["team", "group", "elo", "group_difficulty", "path_difficulty", "easy_path_index", "host"]],
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(path_df.set_index("team")["easy_path_index"].head(15))
    st.caption(
        "**Easy path index** = team Elo minus expected cumulative opponent strength. "
        "Higher means an easier route to the final."
    )
