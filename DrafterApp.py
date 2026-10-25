"""
Drafter App v2 -- NFL Draft Prospect Comparison Tool

Run with:  streamlit run DrafterApp.py   (or the VS Code Run button)

What changed from v1
  1. Input bounds come from the data for the selected position, so impossible
     profiles can't be entered.
  2. Height/weight are scored as "size fit" (how close to a normal body for the
     position), not "bigger = better".
  3. NFL career outcomes stored in the DB (columns starting with "pro_", e.g. career
     value, games, yards) are shown for the closest comps -- "what did similar
     players become" -- but are NOT used to score the prospect.
  4. Draft estimate uses nearest-neighbor matching on the actual numbers
     (not a +/-10 percentile band), includes undrafted players, and refuses to
     give a number when there are no close historical matches.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_CSV = SCRIPT_DIR / "master_players.csv"

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
K_NEIGHBORS = 25          # how many historical comps drive the round estimate
MIN_COVERAGE = 0.60       # comp must have >=60% of the weighted metrics filled in
MAX_COMP_DISTANCE = 1.00  # median comp distance (in std devs) above this = "no reliable match"
MIN_POOL_N = 10           # minimum players with a metric before it can be used
MIN_OUTCOME_SHARE = 0.25  # a pro_ column must be filled in for >=25% of drafted players at the position
PRO_PREFIX = "pro_"       # prefix data_prep.py puts on NFL career-outcome columns
SIZE_Z_LIMIT = 3.0        # height/weight beyond this many std devs = implausible
UNDRAFTED_ROUND = 8

CORE_METRICS = [
    "height_inches", "weight_lbs", "forty_yard", "vertical",
    "bench_press", "broad_jump", "three_cone", "shuttle",
]
SIZE_METRICS = {"height_inches", "weight_lbs"}

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
STEPS = {
    "height_inches": 0.5, "weight_lbs": 1.0, "forty_yard": 0.01, "vertical": 0.5,
    "bench_press": 1.0, "broad_jump": 1.0, "three_cone": 0.01, "shuttle": 0.01,
}
FORMATS = {
    "height_inches": "%.1f", "weight_lbs": "%.0f", "forty_yard": "%.2f", "vertical": "%.1f",
    "bench_press": "%.0f", "broad_jump": "%.0f", "three_cone": "%.2f", "shuttle": "%.2f",
}

# Lower raw value = better performance for these (times).
LOWER_IS_BETTER = {"forty_yard", "three_cone", "shuttle"}

# Scouting-style default weights per position group. Normalized at scoring time
# using only the metrics the user actually filled in. For height/weight the
# weight applies to the SIZE FIT score, not to "bigger is better".
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not MASTER_CSV.exists():
        subprocess.run([sys.executable, str(SCRIPT_DIR / "data_prep.py")], check=True)
    return pd.read_csv(MASTER_CSV)


def pretty(col):
    if col.startswith(PRO_PREFIX):
        col = col[len(PRO_PREFIX):]
    return col.replace("_", " ").title()


def label_for(key):
    return METRIC_LABELS.get(key, pretty(key))


def percentile_of(value, series, lower_is_better):
    """% of historical players this value beats (ties count half)."""
    s = series.dropna()
    if len(s) == 0 or value is None or pd.isna(value):
        return None
    if lower_is_better:
        beat = (s > value).mean() + 0.5 * (s == value).mean()
    else:
        beat = (s < value).mean() + 0.5 * (s == value).mean()
    return round(beat * 100, 1)


def size_fit(value, series):
    """
    0-100 score for how normal a body measurement is for the position.
    100 = exactly the position average, 0 = 3+ standard deviations away
    (in either direction). Returns (score, z).
    """
    s = series.dropna()
    sd = s.std()
    if len(s) == 0 or value is None or pd.isna(value):
        return None, None
    if not sd or np.isnan(sd):
        return 100.0, 0.0
    z = (value - s.mean()) / sd
    return round(100 * max(0.0, 1 - abs(z) / SIZE_Z_LIMIT), 1), z


def round_label(rnd):
    rnd = int(round(rnd))
    return f"Round {rnd}" if rnd < UNDRAFTED_ROUND else "Undrafted / FA"


def round_cell(r):
    return "UDFA" if pd.isna(r) or r >= UNDRAFTED_ROUND else str(int(r))


def match_quality(median_dist):
    if median_dist < 0.5:
        return "Strong"
    if median_dist < 0.8:
        return "Fair"
    return "Weak"


def outcome_columns(pool):
    """NFL career-outcome columns (pro_*) that are populated enough at this position."""
    drafted = pool[pool["draft_round"].notna()]
    cols = []
    for c in pool.columns:
        if not c.startswith(PRO_PREFIX) or not pd.api.types.is_numeric_dtype(pool[c]):
            continue
        if len(drafted) and drafted[c].notna().mean() >= MIN_OUTCOME_SHARE and drafted[c].nunique() >= 3:
            cols.append(c)
    return cols


def outcome_summary(nbrs, pool, cols):
    """What the closest comps went on to do vs. the typical player at the position.
    Blank career stats on a drafted player are treated as 0 (never recorded that stat)."""
    comps = nbrs[nbrs["draft_round"].notna()][cols].fillna(0)
    base = pool[pool["draft_round"].notna()][cols].fillna(0)
    return pd.DataFrame({
        "Career outcome": [pretty(c) for c in cols],
        "Comps median": comps.median().values,
        "Comps 75th pct": comps.quantile(0.75).values,
        "Comps best": comps.max().values,
        "All drafted at position (median)": base.median().values,
    }).round(1)


# ---------------------------------------------------------------------------
# Input widgets
# ---------------------------------------------------------------------------
def render_core_inputs(pool, position):
    """Combine inputs with bounds taken from the selected position's data."""
    inputs = {}
    cols = st.columns(4)
    for i, m in enumerate(CORE_METRICS):
        with cols[i % 4]:
            series = pool[m].dropna()
            if len(series) < MIN_POOL_N:
                st.caption(f"{METRIC_LABELS[m]}: not enough {position} data")
                inputs[m] = None
                continue
            lo, hi = float(series.min()), float(series.max())
            p5, p95 = series.quantile([0.05, 0.95])
            step = STEPS[m]
            default = float(np.clip(round(series.median() / step) * step, lo, hi))
            use = st.checkbox(
                METRIC_LABELS[m], value=True, key=f"chk_{position}_{m}",
                help=f"{position} data range: {lo:g} to {hi:g} (middle 90%: {p5:g} to {p95:g}; n={len(series)})",
            )
            if use:
                inputs[m] = st.number_input(
                    METRIC_LABELS[m], min_value=lo, max_value=hi, step=step,
                    value=default, format=FORMATS[m], key=f"in_{position}_{m}",
                    label_visibility="collapsed",
                )
            else:
                inputs[m] = None
    return inputs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def build_weights(position, used):
    base = DEFAULT_WEIGHTS.get(position, {m: 1 / len(CORE_METRICS) for m in CORE_METRICS})
    w = {m: base.get(m, 0.0) for m in used}
    if used and sum(w.values()) == 0:
        w = {m: 1.0 for m in used}
    total = sum(w.values())
    return {m: v / total for m, v in w.items()} if total else {}


def estimate_draft(pool, values, weights):
    """
    Nearest-neighbor match on standardized actual values (weighted).
    Returns (neighbors_df, reliable_bool, metrics_used_for_matching).
    """
    keys = [k for k, w in weights.items() if w > 0 and k in pool.columns]
    w = pd.Series({k: weights[k] for k in keys})
    mu, sd = pool[keys].mean(), pool[keys].std().replace(0, np.nan)

    z_pool = (pool[keys] - mu) / sd
    z_pros = (pd.Series({k: values[k] for k in keys}) - mu) / sd

    d2 = (z_pool - z_pros) ** 2
    valid = z_pool.notna()
    wsum_valid = valid.mul(w, axis=1).sum(axis=1)
    num = d2.mul(w, axis=1).sum(axis=1)

    scored = pool.copy()
    scored["_dist"] = np.sqrt(num / wsum_valid.replace(0, np.nan))
    scored["_cov"] = wsum_valid / w.sum()

    cand = scored[scored["_cov"] >= MIN_COVERAGE].dropna(subset=["_dist"])
    nbrs = cand.nsmallest(K_NEIGHBORS, "_dist")
    reliable = len(nbrs) >= 10 and nbrs["_dist"].median() <= MAX_COMP_DISTANCE
    return nbrs, reliable, keys


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Drafter App", layout="wide")
    st.title("🏈 Drafter App")
    st.caption(
        "Enter a prospect's combine measurables to see how they stack up against historical "
        "players at their position, where players with similar profiles actually got drafted, "
        "and what those players went on to do in the NFL."
    )

    df = load_data()
    position_groups = sorted(df["position_group"].dropna().unique())

    st.subheader("Prospect Inputs")
    position = st.selectbox(
        "Position", position_groups,
        index=position_groups.index("WR") if "WR" in position_groups else 0,
    )
    pool = df[df["position_group"] == position].copy()
    st.caption(
        f"{len(pool)} historical {position} players in the database. Input limits are the "
        "min/max seen at this position. Uncheck a field if unknown — it is excluded from scoring."
    )

    core_inputs = render_core_inputs(pool, position)

    run = st.button("Evaluate Prospect", type="primary")
    st.divider()
    if not run:
        st.info("Fill in the prospect's numbers above and click **Evaluate Prospect**.")
        return

    values = {k: v for k, v in core_inputs.items() if v is not None}
    if not values:
        st.warning("Enter at least one measurable to evaluate.")
        return

    # ---- per-metric scores -------------------------------------------------
    scores, zscores = {}, {}
    for k in values:
        if k in SIZE_METRICS:
            scores[k], zscores[k] = size_fit(values[k], pool[k])
        else:
            scores[k] = percentile_of(values[k], pool[k], k in LOWER_IS_BETTER)
    scores = {k: v for k, v in scores.items() if v is not None}

    weights = build_weights(position, list(scores))
    weights = {k: w for k, w in weights.items() if w > 0}
    if not weights:
        st.warning("None of the entered metrics carry weight for this position. Enter more measurables.")
        return
    wsum = sum(weights.values())
    composite = sum(scores[k] * w for k, w in weights.items()) / wsum

    # ---- plausibility checks ----------------------------------------------
    implausible = False
    for k in SIZE_METRICS:
        z = zscores.get(k)
        if z is not None and abs(z) > SIZE_Z_LIMIT:
            implausible = True
            st.error(
                f"{METRIC_LABELS[k]} of {values[k]:g} is {abs(z):.1f} standard deviations "
                f"{'above' if z > 0 else 'below'} the {position} average. "
                "That is outside any realistic profile, so no draft estimate will be given."
            )
    for k in values:
        s = pool[k].dropna()
        if k in SIZE_METRICS or len(s) < MIN_POOL_N:
            continue
        lo, hi = s.quantile([0.01, 0.99])
        if values[k] < lo or values[k] > hi:
            st.warning(f"{label_for(k)} = {values[k]:g} is in the extreme 1% of {position} data; "
                       "treat the estimate with caution.")

    # ---- composite + breakdown --------------------------------------------
    st.subheader(f"Composite Score: {composite:.0f} / 100 vs. {position} prospects (n={len(pool)})")
    st.caption("Athletic tests are percentiles (50 = average). "
               "Height/weight are 'size fit' scores (100 = typical build for the position).")

    breakdown = pd.DataFrame({
        "Metric": [label_for(k) for k in weights],
        "Type": ["Size fit" if k in SIZE_METRICS else "Percentile" for k in weights],
        "Prospect": [values[k] for k in weights],
        "Position median": [round(float(pool[k].median()), 2) for k in weights],
        "Score": [scores[k] for k in weights],
        "Weight": [f"{weights[k] / wsum * 100:.0f}%" for k in weights],
    })
    st.bar_chart(breakdown.set_index("Metric")["Score"])
    st.dataframe(breakdown, hide_index=True, use_container_width=True)

    if implausible:
        return

    # ---- draft estimate ----------------------------------------------------
    st.subheader("Estimated Draft Range")
    nbrs, reliable, match_keys = estimate_draft(pool, {k: values[k] for k in weights}, weights)

    has_undrafted = bool(pool["draft_round"].isna().any() or (pool["draft_round"] >= UNDRAFTED_ROUND).any())
    if not has_undrafted:
        st.caption("Note: this dataset only contains drafted players, so 'Undrafted' cannot be "
                   "predicted. The estimate is 'if drafted, where.'")

    if len(nbrs) == 0:
        st.warning("No historical players have enough overlapping data to compare against.")
    elif not reliable:
        st.warning(
            f"No reliable match: the closest {len(nbrs)} historical {position}s are still far from this "
            f"profile (median distance {nbrs['_dist'].median():.2f} std devs, limit {MAX_COMP_DISTANCE:.2f}). "
            "Showing the nearest players below, but no round estimate."
        )
    else:
        rounds = nbrs["draft_round"].fillna(UNDRAFTED_ROUND)
        rounds = rounds.where(rounds < UNDRAFTED_ROUND, UNDRAFTED_ROUND)
        lo, hi, med = rounds.quantile(0.25), rounds.quantile(0.75), rounds.median()
        p_top3 = (rounds <= 3).mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Typical Round", round_label(med))
        c2.metric("Likely Range (middle 50%)", f"{round_label(lo)} – {round_label(hi)}")
        c3.metric("Chance of Rounds 1-3", f"{p_top3:.0f}%")
        c4.metric("Match Quality", match_quality(nbrs["_dist"].median()),
                  help=f"Based on {len(nbrs)} nearest comps, median distance {nbrs['_dist'].median():.2f} std devs.")

        dist = rounds.value_counts(normalize=True).reindex(range(1, UNDRAFTED_ROUND + 1), fill_value=0) * 100
        dist.index = [f"Rd {i}" if i < UNDRAFTED_ROUND else "UDFA" for i in dist.index]
        st.caption("How the closest historical matches were actually drafted (% of comps):")
        st.bar_chart(dist.rename("% of comps"))

    # ---- comps table -------------------------------------------------------
    if len(nbrs) > 0:
        pro_cols = outcome_columns(pool)

        st.subheader("Closest Historical Comparisons")
        comps = nbrs.head(10).copy()
        comps["Similarity"] = (100 / (1 + comps["_dist"])).round(0)
        comps["Round"] = comps["draft_round"].apply(round_cell)

        # Show the 4 best-populated career columns next to each comp.
        show_pro = sorted(pro_cols, key=lambda c: -pool[c].notna().mean())[:4]

        id_cols = [c for c in ["player", "draft_year", "school"] if c in comps.columns]
        tail_cols = [c for c in ["draft_pick", "team"] if c in comps.columns]
        out = comps[id_cols + ["Round"] + tail_cols + ["Similarity"] + match_keys + show_pro].rename(columns={
            "player": "Player", "draft_year": "Year", "school": "School",
            "draft_pick": "Pick", "team": "Team",
            **{k: label_for(k) for k in match_keys},
            **{c: f"NFL: {pretty(c)}" for c in show_pro},
        })
        st.dataframe(out, hide_index=True, use_container_width=True)

        st.subheader("What Similar Players Became")
        if pro_cols:
            st.caption(
                f"NFL career totals for the {len(nbrs)} closest comps vs. the typical drafted {position}. "
                "These are outcomes of the comps, not inputs to the prospect's score."
            )
            st.dataframe(outcome_summary(nbrs, pool, pro_cols), hide_index=True, use_container_width=True)
        else:
            st.info("No NFL career columns found in master_players.csv. Re-run data_prep.py "
                    "(updated version) to carry them through from combined_review.csv.")

    with st.expander("Weights used for this evaluation"):
        st.write({label_for(k): f"{w / wsum * 100:.0f}%" for k, w in weights.items()})


if __name__ == "__main__":
    from streamlit.web import cli as stcli

    # Lets you launch this with `python DrafterApp.py` (or VS Code's Run button)
    # instead of typing `streamlit run` in a terminal.
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
