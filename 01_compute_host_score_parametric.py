# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_CSV = HERE / "data" / "candidate_pool.csv"
DEFAULT_OUT_CSV = HERE / "outputs" / "candidates_with_host_score.csv"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def score_gap(x, gap_center, gap_width):
    if pd.isna(x):
        return 0.0
    return float(np.exp(-((x - gap_center) / gap_width) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--gap_center", type=float, default=3.8)
    parser.add_argument("--gap_width", type=float, default=1.2)
    parser.add_argument("--ehull_scale", type=float, default=0.02)
    parser.add_argument("--complexity_center", type=float, default=4.0)
    parser.add_argument("--complexity_scale", type=float, default=4.0)
    parser.add_argument("--re_like_floor", type=float, default=0.6)
    parser.add_argument("--w_gap", type=float, default=0.35)
    parser.add_argument("--w_stability", type=float, default=0.30)
    parser.add_argument("--w_complexity", type=float, default=0.15)
    parser.add_argument("--w_re", type=float, default=0.15)
    parser.add_argument("--w_site", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.candidate_csv).copy()
    if "formula" in df.columns and "host_formula" not in df.columns:
        df = df.rename(columns={"formula":"host_formula"})
    if "material_id" not in df.columns and "mp_id" in df.columns:
        df = df.rename(columns={"mp_id":"material_id"})

    re_cols = ["has_y", "has_gd", "has_la", "has_lu", "has_sc", "has_in"]
    for c in re_cols:
        if c not in df.columns:
            df[c] = False
    df["contains_re_like_cation"] = df[re_cols].any(axis=1)

    if "n_elements" not in df.columns:
        if "elements" in df.columns:
            df["n_elements"] = df["elements"].fillna("").apply(lambda s: len([x for x in str(s).split(",") if x.strip()]))
        else:
            df["n_elements"] = 0

    df["score_gap"] = df["band_gap"].apply(lambda x: score_gap(x, args.gap_center, args.gap_width))
    df["score_stability"] = np.exp(-df["energy_above_hull"].fillna(999.0) / args.ehull_scale)
    df["score_complexity"] = np.exp(-((df["n_elements"] - args.complexity_center) ** 2) / args.complexity_scale)
    df["score_re"] = df["contains_re_like_cation"].apply(lambda x: 1.0 if bool(x) else float(args.re_like_floor))
    df["score_site_flag"] = 1.0

    df["host_score"] = (
        args.w_gap * df["score_gap"] +
        args.w_stability * df["score_stability"] +
        args.w_complexity * df["score_complexity"] +
        args.w_re * df["score_re"] +
        args.w_site * df["score_site_flag"]
    )

    df = df.sort_values("host_score", ascending=False).reset_index(drop=True)
    ensure_dir(Path(args.out_csv).parent)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved to: {0}".format(args.out_csv))
    print(df[["host_formula", "host_score", "score_gap", "score_stability"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
