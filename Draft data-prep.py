"""
Builds master_players.csv from the pipeline's combined_review.csv.

As of this version, combined_review.csv already has ONE ROW PER PLAYER
with both combine measurables and their actual draft outcome joined --
no name-matching join needed here anymore. This script just cleans
column names/types and standardizes position labels.
"""
import re
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = SCRIPT_DIR / "combined_review.csv"
OUTPUT_CSV = SCRIPT_DIR / "master_players.csv"

# Maps raw position labels (from either source) -> a standard position group
# used for weighting and historical comparisons.
POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "FB",
    "WR": "WR", "CB/WR": "WR",
    "TE": "TE",
    "OT": "OT", "T": "OT",
    "OG": "OG", "G": "OG",
    "C": "C",
    "OL": "OL",
    "DE": "EDGE", "EDGE": "EDGE",
    "DT": "DT", "DL": "DT", "NT": "DT",
    "OLB": "LB", "ILB": "LB", "LB": "LB",
    "CB": "CB",
    "S": "S", "SAF": "S", "FS": "S", "SS": "S",
    "DB": "DB",
    "K": "K",
    "P": "P",
    "LS": "LS",
}


def parse_height(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"^(\d+)$", text)
    if m:
        return int(m.group(1))
    return None


def build():
    df = pd.read_csv(SOURCE_CSV)

    df["height_inches"] = df["ht"].apply(parse_height)
    df["weight_lbs"] = pd.to_numeric(df["wt"], errors="coerce")
    df["forty_yard"] = pd.to_numeric(df["fourty_yard"], errors="coerce")
    df["vertical"] = pd.to_numeric(df["vertical"], errors="coerce")
    df["bench_press"] = pd.to_numeric(df["bench"], errors="coerce")
    df["broad_jump"] = pd.to_numeric(df["broad_jump"], errors="coerce")
    df["three_cone"] = pd.to_numeric(df["three_cone"], errors="coerce")
    df["shuttle"] = pd.to_numeric(df["shuttle"], errors="coerce")
    df["draft_round"] = pd.to_numeric(df["rnd"], errors="coerce")
    df["draft_pick"] = pd.to_numeric(df["pick"], errors="coerce")

    # position group: prefer the draft-side label (usually cleaner/final),
    # fall back to the combine-side label.
    df["position_raw"] = df["pos_draft"].fillna(df["pos_combine"])
    df["position_group"] = df["position_raw"].map(POSITION_GROUP_MAP)
    df["position_group"] = df["position_group"].fillna(df["position_raw"])

    out_cols = [
        "player", "school", "position_raw", "position_group", "draft_year",
        "height_inches", "weight_lbs", "forty_yard", "vertical", "bench_press",
        "broad_jump", "three_cone", "shuttle", "draft_round", "draft_pick", "tm",
    ]
    out = df[out_cols].rename(columns={"tm": "team"})

    out = out[out["draft_round"].notna()].drop_duplicates(
        subset=["player", "draft_year", "position_group"]
    )

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Built {OUTPUT_CSV} with {len(out)} rows.")
    print(out["position_group"].value_counts())
    return out


if __name__ == "__main__":
    build()