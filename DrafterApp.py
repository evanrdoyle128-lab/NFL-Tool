"""
Drafter App -- NFL Draft Prospect Comparison Tool

Run with:  streamlit run app.py

Enter a prospect's combine measurables and position, and see:
  - How they stack up (percentile) against every historical player at that
    position, weighted by which drills matter most for that position.
  - The historical players they compare most closely to, and where those
    players actually got drafted.
  - An estimated draft round range based on how similarly-scored historical
    prospects were actually drafted.
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_CSV = SCRIPT_DIR / "master_players.csv"

METRICS = [
    "height_inches", "weight_lbs", "forty_yard", "vertical",
    "bench_press", "broad_jump", "three_cone", "shuttle",
]

METRIC_LABELS = {
    "height_inches": "Height (in)",
    "weight_lbs": "Weight (lbs)",
    "forty_yard": "40-Yard Dash (sec)",
    "vertical": "Vertical Jump (in)",
    "bench_press": "Bench Press (reps)",
    "broad_jump": "Broad Jump (in)",
    "three_cone": "3-Cone Drill (sec)",
    "shuttle": "20-Yard Shuttle (sec)",
}

# Lower raw value = better performance for these (times).
LOWER_IS_BETTER = {"forty_yard", "three_cone", "shuttle"}

# Scouting-style default weights per position group. Weights don't need to
# sum to 1 -- they're normalized at scoring time using only the metrics the
# user actually filled in.
DEFAULT_WEIGHTS = {
    "QB":  {"height_inches": 0.20, "weight_lbs": 0.15, "forty_yard": 0.15,
            "vertical": 0.10, "bench_press": 0.00, "broad_jump": 0.10,
            "three_cone": 0.15, "shuttle": 0.15},
    "RB":  {"height_inches": 0.05, "weight_lbs": 0.10, "forty_yard": 0.30,
            "vertical": 0.10, "bench_press": 0.05, "broad_jump": 0.15,
            "three_cone": 0.15, "shuttle": 0.15},
    "FB":  {"height_inches": 0.05, "weight_lbs": 0.15, "forty_yard": 0.20,
            "vertical": 0.10, "bench_press": 0.15, "broad_jump": 0.10,
            "three_cone": 0.10, "shuttle": 0.10},
    "WR":  {"height_inches": 0.05, "weight_lbs": 0.05, "forty_yard": 0.30,
            "vertical": 0.15, "bench_press": 0.05, "broad_jump": 0.15,
            "three_cone": 0.15, "shuttle": 0.15},
    "TE":  {"height_inches": 0.10, "weight_lbs": 0.15, "forty_yard": 0.20,
            "vertical": 0.10, "bench_press": 0.15, "broad_jump": 0.10,
            "three_cone": 0.10, "shuttle": 0.10},
    "OT":  {"height_inches": 0.10, "weight_lbs": 0.20, "forty_yard": 0.10,
            "vertical": 0.00, "bench_press": 0.30, "broad_jump": 0.00,
            "three_cone": 0.15, "shuttle": 0.15},
    "OG":  {"height_inches": 0.05, "weight_lbs": 0.20, "forty_yard": 0.10,
            "vertical": 0.00, "bench_press": 0.35, "broad_jump": 0.00,
            "three_cone": 0.15, "shuttle": 0.15},
    "C":   {"height_inches": 0.05, "weight_lbs": 0.20, "forty_yard": 0.10,
            "vertical": 0.00, "bench_press": 0.35, "broad_jump": 0.00,
            "three_cone": 0.15, "shuttle": 0.15},
    "OL":  {"height_inches": 0.10, "weight_lbs": 0.20, "forty_yard": 0.10,
            "vertical": 0.00, "bench_press": 0.30, "broad_jump": 0.00,
            "three_cone": 0.15, "shuttle": 0.15},
    "EDGE": {"height_inches": 0.05, "weight_lbs": 0.10, "forty_yard": 0.25,
             "vertical": 0.10, "bench_press": 0.15, "broad_jump": 0.15,
             "three_cone": 0.10, "shuttle": 0.10},
    "DT":  {"height_inches": 0.05, "weight_lbs": 0.20, "forty_yard": 0.15,
            "vertical": 0.05, "bench_press": 0.25, "broad_jump": 0.15,
            "three_cone": 0.10, "shuttle": 0.05},
    "LB":  {"height_inches": 0.05, "weight_lbs": 0.05, "forty_yard": 0.20,
            "vertical": 0.15, "bench_press": 0.15, "broad_jump": 0.15,
            "three_cone": 0.20, "shuttle": 0.05},
    "CB":  {"height_inches": 0.05, "weight_lbs": 0.05, "forty_yard": 0.30,
            "vertical": 0.15, "bench_press": 0.05, "broad_jump": 0.15,
            "three_cone": 0.15, "shuttle": 0.10},
    "S":   {"height_inches": 0.05, "weight_lbs": 0.05, "forty_yard": 0.25,
            "vertical": 0.15, "bench_press": 0.10, "broad_jump": 0.15,
            "three_cone": 0.15, "shuttle": 0.10},
    "DB":  {"height_inches": 0.05, "weight_lbs": 0.05, "forty_yard": 0.28,
            "vertical": 0.15, "bench_press": 0.07, "broad_jump": 0.15,
            "three_cone": 0.15, "shuttle": 0.10},
    "K":   {"height_inches": 0.10, "weight_lbs": 0.10, "forty_yard": 0.20,
            "vertical": 0.20, "bench_press": 0.00, "broad_jump": 0.20,
            "three_cone": 0.10, "shuttle": 0.10},
    "P":   {"height_inches": 0.10, "weight_lbs": 0.10, "forty_yard": 0.20,
            "vertical": 0.20, "bench_press": 0.00, "broad_jump": 0.20,
            "three_cone": 0.10, "shuttle": 0.10},
    "LS":  {"height_inches": 0.15, "weight_lbs": 0.20, "forty_yard": 0.15,
            "vertical": 0.05, "bench_press": 0.20, "broad_jump": 0.05,
            "three_cone": 0.10, "shuttle": 0.10},
}


@st.cache_data
def load_data():
    if not MASTER_CSV.exists():
        subprocess.run(["python3", str(SCRIPT_DIR / "data_prep.py")], check=True)
    df = pd.read_csv(MASTER_CSV)
    return df


def percentile_of(value, series, lower_is_better):
    """% of historical players this value beats. series should already be NA-free."""
    if len(series) == 0 or pd.isna(value):
        return None
    if lower_is_better:
        pct = (series > value).mean() * 100
    else:
        pct = (series < value).mean() * 100
    return round(pct, 1)


def round_label(rnd):
    rnd = int(round(rnd))
    return f"Round {rnd}" if rnd <= 7 else "Undrafted / FA"


def main():
    st.set_page_config(page_title="Drafter App", layout="wide")
    st.title("🏈 Drafter App")
    st.caption(
        "Enter a prospect's measurables to see how they stack up against "
        "historical performers at their position, and where players who "
        "tested similarly actually got drafted."
    )

    df = load_data()
    position_groups = sorted(df["position_group"].dropna().unique())

    st.subheader("Prospect Inputs")
    st.caption("Uncheck a field if unknown — it will be excluded from scoring.")

    inputs = {}

    def num_input(container, key, label, min_v, max_v, step, default):
        with container:
            use = st.checkbox(label, value=True, key=f"chk_{key}")
            if use:
                inputs[key] = st.number_input(label, min_value=min_v, max_value=max_v,
                                               step=step, value=default, key=key,
                                               label_visibility="collapsed")
            else:
                inputs[key] = None

    row1 = st.columns(9)
    with row1[0]:
        position = st.selectbox("Position", position_groups,
                                 index=position_groups.index("WR") if "WR" in position_groups else 0)
    num_input(row1[1], "height_inches", "Height (in)", 60.0, 84.0, 0.5, 72.0)
    num_input(row1[2], "weight_lbs", "Weight (lbs)", 150.0, 380.0, 1.0, 210.0)
    num_input(row1[3], "forty_yard", "40-Yd (sec)", 4.20, 5.80, 0.01, 4.60)
    num_input(row1[4], "vertical", "Vertical (in)", 20.0, 46.0, 0.5, 34.0)
    num_input(row1[5], "bench_press", "Bench (reps)", 0.0, 50.0, 1.0, 15.0)
    num_input(row1[6], "broad_jump", "Broad Jump (in)", 90.0, 145.0, 1.0, 118.0)
    num_input(row1[7], "three_cone", "3-Cone (sec)", 6.30, 8.50, 0.01, 7.00)
    num_input(row1[8], "shuttle", "Shuttle (sec)", 3.80, 5.20, 0.01, 4.30)

    run = st.button("Evaluate Prospect", type="primary")
    st.divider()

    if not run:
        st.info("Fill in the prospect's measurables above and click **Evaluate Prospect**.")
        return

    pool = df[df["position_group"] == position].copy()
    weights = DEFAULT_WEIGHTS.get(position, {m: 1 / len(METRICS) for m in METRICS})

    percentiles = {}
    for m in METRICS:
        series = pool[m].dropna()
        percentiles[m] = percentile_of(inputs.get(m), series, m in LOWER_IS_BETTER)

    used_metrics = {m: p for m, p in percentiles.items() if p is not None}
    if not used_metrics:
        st.warning("Enter at least one measurable to evaluate.")
        return

    weight_sum = sum(weights.get(m, 0) for m in used_metrics)
    if weight_sum == 0:
        weight_sum = len(used_metrics)
        composite = np.mean(list(used_metrics.values()))
    else:
        composite = sum(used_metrics[m] * weights.get(m, 0) for m in used_metrics) / weight_sum

    st.subheader(f"Composite Score: {composite:.1f}th percentile among {position} prospects (n={len(pool)})")

    # --- percentile bar chart ---
    chart_df = pd.DataFrame({
        "Metric": [METRIC_LABELS[m] for m in used_metrics],
        "Percentile": [used_metrics[m] for m in used_metrics],
        "Weight": [f"{weights.get(m, 0) * 100 / weight_sum:.0f}%" for m in used_metrics],
    })
    st.bar_chart(chart_df.set_index("Metric")["Percentile"])
    st.dataframe(chart_df, hide_index=True, use_container_width=True)

    # --- draft round estimate ---
    st.subheader("Estimated Draft Range")
    pool_scored = pool.dropna(subset=["draft_round"]).copy()
    for m in METRICS:
        series = pool_scored[m].dropna()
        if m in used_metrics and len(series) > 0:
            pool_scored[f"pct_{m}"] = pool_scored[m].apply(
                lambda v: percentile_of(v, series, m in LOWER_IS_BETTER)
            )
    pct_cols = [f"pct_{m}" for m in used_metrics if f"pct_{m}" in pool_scored.columns]
    if pct_cols:
        pool_scored["composite"] = pool_scored[pct_cols].apply(
            lambda row: sum(row[f"pct_{m}"] * weights.get(m, 0) for m in used_metrics
                             if not pd.isna(row.get(f"pct_{m}"))) / weight_sum,
            axis=1,
        )
        similar_band = pool_scored[
            (pool_scored["composite"] >= composite - 10) &
            (pool_scored["composite"] <= composite + 10)
        ]
        if len(similar_band) >= 5:
            rounds = similar_band["draft_round"].dropna()
            lo, hi = rounds.quantile(0.25), rounds.quantile(0.75)
            median_rnd = rounds.median()
            col1, col2, col3 = st.columns(3)
            col1.metric("Typical Round", round_label(median_rnd))
            col2.metric("Likely Range", f"{round_label(lo)} – {round_label(hi)}")
            col3.metric("Comparable Players", len(similar_band))
        else:
            st.write("Not enough historically similar players (±10 percentile) to estimate a range reliably.")

        # --- nearest comps ---
        st.subheader("Closest Historical Comparisons")
        pool_scored["dist"] = (pool_scored["composite"] - composite).abs()
        comps = pool_scored.sort_values("dist").head(10)
        display_cols = ["player", "draft_year", "school", "draft_round", "draft_pick", "team", "composite"] \
            if "school" in comps.columns else ["player", "draft_year", "draft_round", "draft_pick", "team", "composite"]
        comps_display = comps[[c for c in display_cols if c in comps.columns]].rename(columns={
            "player": "Player", "draft_year": "Year", "school": "School",
            "draft_round": "Round", "draft_pick": "Pick", "team": "Team", "composite": "Composite %ile",
        })
        comps_display["Composite %ile"] = comps_display["Composite %ile"].round(1)
        st.dataframe(comps_display, hide_index=True, use_container_width=True)

    with st.expander("Position weight profile used"):
        st.write({METRIC_LABELS[m]: f"{w*100:.0f}%" for m, w in weights.items()})


if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli

    # Lets you launch this with `python app.py` (or VS Code's Run button)
    # instead of typing `streamlit run app.py` in a terminal.
    if st.runtime.exists():
        # Already running under `streamlit run` -- just execute normally.
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())