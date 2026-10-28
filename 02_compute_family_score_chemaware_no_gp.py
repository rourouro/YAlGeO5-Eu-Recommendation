# -*- coding: utf-8 -*-
"""
02_compute_family_score_chemaware_no_gp.py

Chemistry-aware anchor-family similarity for Eu3+ long-wavelength red phosphor host recommendation.

Compared with the original elemental Jaccard S_family, this version computes, for each
candidate-anchor pair:

    S_anchor = w_elem*S_elem + w_framework*S_framework + w_chem*S_chem

where
    S_elem      : weighted elemental-set similarity, with O down-weighted;
    S_framework : similarity of oxide framework-forming cations / cation groups;
    S_chem      : similarity of composition-derived cation chemical descriptors.

Then S_family is obtained by aggregating the top-k most similar anchors with anchor-level weights.

This script is intentionally rule-based and interpretable. It does NOT predict absolute PL intensity.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_CSV = HERE / "outputs" / "candidates_with_host_score.csv"
DEFAULT_ANCHOR_CSV = HERE / "data" / "anchors.csv"
DEFAULT_OUT_CSV = HERE / "outputs" / "candidates_with_family_score.csv"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_formula_column(df):
    if "host_formula" in df.columns:
        return "host_formula"
    if "formula" in df.columns:
        return "formula"
    raise KeyError("No host_formula or formula column found.")

DEFAULT_LEVEL_WEIGHT = {"A+": 1.05, "A": 1.00, "B": 0.85, "C": 0.70}

# Minimal periodic-property table for elements likely to appear in oxide hosts.
# Values: atomic_number Z, Pauling electronegativity chi, covalent radius pm r_cov,
# periodic table group, period.
ELEMENT_PROPS = {
    "H": (1, 2.20, 31, 1, 1), "Li": (3, 0.98, 128, 1, 2), "Be": (4, 1.57, 96, 2, 2),
    "B": (5, 2.04, 84, 13, 2), "C": (6, 2.55, 76, 14, 2), "N": (7, 3.04, 71, 15, 2),
    "O": (8, 3.44, 66, 16, 2), "F": (9, 3.98, 57, 17, 2),
    "Na": (11, 0.93, 166, 1, 3), "Mg": (12, 1.31, 141, 2, 3), "Al": (13, 1.61, 121, 13, 3),
    "Si": (14, 1.90, 111, 14, 3), "P": (15, 2.19, 107, 15, 3), "S": (16, 2.58, 105, 16, 3),
    "Cl": (17, 3.16, 102, 17, 3), "K": (19, 0.82, 203, 1, 4), "Ca": (20, 1.00, 176, 2, 4),
    "Sc": (21, 1.36, 170, 3, 4), "Ti": (22, 1.54, 160, 4, 4), "V": (23, 1.63, 153, 5, 4),
    "Cr": (24, 1.66, 139, 6, 4), "Mn": (25, 1.55, 139, 7, 4), "Fe": (26, 1.83, 132, 8, 4),
    "Co": (27, 1.88, 126, 9, 4), "Ni": (28, 1.91, 124, 10, 4), "Cu": (29, 1.90, 132, 11, 4),
    "Zn": (30, 1.65, 122, 12, 4), "Ga": (31, 1.81, 122, 13, 4), "Ge": (32, 2.01, 120, 14, 4),
    "As": (33, 2.18, 119, 15, 4), "Se": (34, 2.55, 120, 16, 4), "Br": (35, 2.96, 120, 17, 4),
    "Rb": (37, 0.82, 220, 1, 5), "Sr": (38, 0.95, 195, 2, 5), "Y": (39, 1.22, 190, 3, 5),
    "Zr": (40, 1.33, 175, 4, 5), "Nb": (41, 1.60, 164, 5, 5), "Mo": (42, 2.16, 154, 6, 5),
    "Ru": (44, 2.20, 146, 8, 5), "Rh": (45, 2.28, 142, 9, 5), "Pd": (46, 2.20, 139, 10, 5),
    "Ag": (47, 1.93, 145, 11, 5), "Cd": (48, 1.69, 144, 12, 5), "In": (49, 1.78, 142, 13, 5),
    "Sn": (50, 1.96, 139, 14, 5), "Sb": (51, 2.05, 139, 15, 5), "Te": (52, 2.10, 138, 16, 5),
    "I": (53, 2.66, 139, 17, 5), "Cs": (55, 0.79, 244, 1, 6), "Ba": (56, 0.89, 215, 2, 6),
    "La": (57, 1.10, 207, 3, 6), "Ce": (58, 1.12, 204, 3, 6), "Pr": (59, 1.13, 203, 3, 6),
    "Nd": (60, 1.14, 201, 3, 6), "Sm": (62, 1.17, 198, 3, 6), "Eu": (63, 1.20, 198, 3, 6),
    "Gd": (64, 1.20, 196, 3, 6), "Tb": (65, 1.20, 194, 3, 6), "Dy": (66, 1.22, 192, 3, 6),
    "Ho": (67, 1.23, 192, 3, 6), "Er": (68, 1.24, 189, 3, 6), "Tm": (69, 1.25, 190, 3, 6),
    "Yb": (70, 1.10, 187, 3, 6), "Lu": (71, 1.27, 187, 3, 6), "Hf": (72, 1.30, 175, 4, 6),
    "Ta": (73, 1.50, 170, 5, 6), "W": (74, 2.36, 162, 6, 6), "Re": (75, 1.90, 151, 7, 6),
    "Os": (76, 2.20, 144, 8, 6), "Ir": (77, 2.20, 141, 9, 6), "Pt": (78, 2.28, 136, 10, 6),
    "Au": (79, 2.54, 136, 11, 6), "Hg": (80, 2.00, 132, 12, 6), "Tl": (81, 1.62, 145, 13, 6),
    "Pb": (82, 2.33, 146, 14, 6), "Bi": (83, 2.02, 148, 15, 6),
}

# Elements that are more likely to act as Eu3+ substitution-site cations rather than framework cations.
# They are excluded from S_framework, because Eu3+ substitution feasibility belongs to S_host.
SITE_LIKE_CATIONS = {
    "Y", "Sc", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}

# Framework-forming cations and group labels used for group-aware framework similarity.
FRAMEWORK_GROUPS = {
    "B": "B",
    "Al": "AlGaIn", "Ga": "AlGaIn", "In": "AlGaIn",
    "Si": "SiGeSn", "Ge": "SiGeSn", "Sn": "SiGeSn",
    "P": "PNbTaV", "V": "PNbTaV", "Nb": "PNbTaV", "Ta": "PNbTaV",
    "Ti": "TiZrHf", "Zr": "TiZrHf", "Hf": "TiZrHf",
    "Mo": "MoW", "W": "MoW",
    "Sb": "SbBi", "Bi": "SbBi",
}


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse an inorganic formula, expanding parentheses.

    Ca3Y2(GeO4)3 is Ca3 Y2 Ge3 O12. A multiplier written after ')' is applied
    to every element inside the group. Formulas without parentheses are unchanged.
    """
    if not isinstance(formula, str):
        return {}
    s = formula.strip().replace("·", "").replace(" ", "")
    s = re.sub(r":.*$", "", s)

    def parse(i):
        comp = {}
        while i < len(s):
            ch = s[i]
            if ch == "(":
                inner, i = parse(i + 1)
                m = re.match(r"(\d*\.?\d*)", s[i:])
                num = float(m.group(1)) if m and m.group(1) else 1.0
                i += m.end() if m else 0
                for el, amt in inner.items():
                    comp[el] = comp.get(el, 0.0) + amt * num
            elif ch == ")":
                return comp, i + 1
            else:
                m = _FORMULA_TOKEN.match(s, i)
                if not m:
                    i += 1
                    continue
                el, num = m.group(1), m.group(2)
                comp[el] = comp.get(el, 0.0) + (float(num) if num else 1.0)
                i = m.end()
        return comp, i

    comp, _ = parse(0)
    return {el: amt for el, amt in comp.items() if amt > 0}


def parse_elements_cell(cell) -> Set[str]:
    if pd.isna(cell):
        return set()
    if isinstance(cell, str) and "," in cell:
        return set([x.strip() for x in cell.split(",") if x.strip()])
    return set(parse_formula(str(cell)).keys())


def weighted_jaccard(set_a: Set[str], set_b: Set[str], weights: Dict[str, float] = None) -> float:
    weights = weights or {}
    union = set_a | set_b
    if not union:
        return 0.0
    inter = set_a & set_b
    num = sum(weights.get(e, 1.0) for e in inter)
    den = sum(weights.get(e, 1.0) for e in union)
    return float(num / den) if den > 0 else 0.0


def get_framework_elements(comp: Dict[str, float]) -> Set[str]:
    elems = set(comp.keys())
    return set(e for e in elems if e in FRAMEWORK_GROUPS and e not in SITE_LIKE_CATIONS and e != "O")


def get_framework_groups(comp: Dict[str, float]) -> Set[str]:
    return set(FRAMEWORK_GROUPS[e] for e in get_framework_elements(comp) if e in FRAMEWORK_GROUPS)


def framework_similarity(comp_a: Dict[str, float], comp_b: Dict[str, float]) -> float:
    """Group-aware framework similarity.
    60% group-level similarity + 40% exact framework-element similarity.
    This gives partial credit to chemically similar pairs such as Al/Ga or Si/Ge.
    """
    fa = get_framework_elements(comp_a)
    fb = get_framework_elements(comp_b)
    ga = get_framework_groups(comp_a)
    gb = get_framework_groups(comp_b)
    elem_sim = weighted_jaccard(fa, fb)
    group_sim = weighted_jaccard(ga, gb)
    if not (fa or fb or ga or gb):
        return 0.0
    return 0.4 * elem_sim + 0.6 * group_sim


def cation_descriptor(comp: Dict[str, float]) -> np.ndarray:
    """Composition-derived cation chemical descriptor.
    Oxygen is excluded because all candidates are oxides and O weakens discrimination.

    This no-group/period variant keeps only three continuous elemental properties:
      atomic number, electronegativity, covalent radius.

    It removes group number and period number from S_chem to reduce overlap with
    S_framework, because S_framework already encodes periodic-table framework groups
    such as Al/Ga/In and Si/Ge/Sn.

    The final descriptor contains weighted means and weighted standard deviations of:
      atomic number, electronegativity, covalent radius.
    Therefore, descriptor dimension = 6.
    """
    items = [(e, amt) for e, amt in comp.items() if e != "O" and e in ELEMENT_PROPS and amt > 0]
    if not items:
        return np.zeros(6, dtype=float)
    elems = [e for e, _ in items]
    amounts = np.array([amt for _, amt in items], dtype=float)
    weights = amounts / amounts.sum()

    # ELEMENT_PROPS[e] = (Z, chi, r_cov, group, period)
    # Only keep the first three dimensions: Z, chi, r_cov.
    props = np.array([ELEMENT_PROPS[e][:3] for e in elems], dtype=float)  # n x 3

    mean = np.sum(props * weights[:, None], axis=0)
    var = np.sum(((props - mean) ** 2) * weights[:, None], axis=0)
    std = np.sqrt(np.maximum(var, 0.0))
    return np.concatenate([mean, std])


def zscore_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd, mu, sd


def chem_similarity(zvec_a: np.ndarray, zvec_b: np.ndarray) -> float:
    d2 = float(np.sum((zvec_a - zvec_b) ** 2))
    n = max(len(zvec_a), 1)
    return float(math.exp(-d2 / (2.0 * n)))


def compute_chemaware_family_scores(
    candidate_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    top_k: int = 3,
    level_weight: Dict[str, float] = None,
    only_anchor_levels: List[str] = None,
    elem_w: float = 1.0 / 3.0,
    framework_w: float = 1.0 / 3.0,
    chem_w: float = 1.0 / 3.0,
    oxygen_weight: float = 0.2,
    normalized_anchor_weight: bool = True,
) -> pd.DataFrame:
    level_weight = level_weight or DEFAULT_LEVEL_WEIGHT.copy()
    cdf = candidate_df.copy()
    adf = anchor_df.copy()
    formula_col = find_formula_column(cdf)
    anchor_formula_col = "host_formula" if "host_formula" in adf.columns else find_formula_column(adf)

    if only_anchor_levels:
        adf = adf[adf["anchor_level"].isin(only_anchor_levels)].copy()

    if "anchor_level" not in adf.columns:
        adf["anchor_level"] = "A"
    adf["_anchor_weight"] = adf["anchor_level"].map(level_weight).fillna(1.0).astype(float)

    # Composition and element sets.
    cdf["_comp"] = cdf[formula_col].apply(parse_formula)
    if "elements" in cdf.columns:
        cdf["_elem_set"] = cdf["elements"].apply(parse_elements_cell)
    else:
        cdf["_elem_set"] = cdf["_comp"].apply(lambda x: set(x.keys()))

    adf["_comp"] = adf[anchor_formula_col].apply(parse_formula)
    adf["_elem_set"] = adf["_comp"].apply(lambda x: set(x.keys()))

    # Chemical descriptors: fit z-score scaling on candidates + anchors together.
    # No-group/period variant: S_chem descriptor dimension = 6.
    c_desc = np.vstack([cation_descriptor(x) for x in cdf["_comp"]]) if len(cdf) else np.zeros((0, 6))
    a_desc = np.vstack([cation_descriptor(x) for x in adf["_comp"]]) if len(adf) else np.zeros((0, 6))
    all_desc = np.vstack([c_desc, a_desc])
    all_z, _, _ = zscore_matrix(all_desc)
    c_z = all_z[: len(cdf)]
    a_z = all_z[len(cdf):]

    elem_weights = {"O": float(oxygen_weight)}

    records = []
    anchor_rows = list(adf.reset_index(drop=True).iterrows())
    for ci, (_, row) in enumerate(cdf.iterrows()):
        comp_c = row["_comp"]
        elem_c = row["_elem_set"]
        z_c = c_z[ci]
        sims = []
        for ai, (_, arow) in enumerate(anchor_rows):
            comp_a = arow["_comp"]
            elem_a = arow["_elem_set"]
            z_a = a_z[ai]

            s_elem = weighted_jaccard(elem_c, elem_a, elem_weights)
            s_framework = framework_similarity(comp_c, comp_a)
            s_chem = chem_similarity(z_c, z_a)
            s_anchor = elem_w * s_elem + framework_w * s_framework + chem_w * s_chem
            aw = float(arow["_anchor_weight"])
            weighted_score = s_anchor * aw

            sims.append({
                "anchor_formula": arow[anchor_formula_col],
                "anchor_level": arow["anchor_level"],
                "anchor_weight": aw,
                "S_elem": float(s_elem),
                "S_framework": float(s_framework),
                "S_chem": float(s_chem),
                "S_anchor": float(s_anchor),
                "weighted_anchor_score": float(weighted_score),
            })

        sims_sorted = sorted(sims, key=lambda x: x["weighted_anchor_score"], reverse=True)
        top_sims = sims_sorted[:top_k]
        if top_sims:
            if normalized_anchor_weight:
                denom = sum(x["anchor_weight"] for x in top_sims)
                family_score = sum(x["anchor_weight"] * x["S_anchor"] for x in top_sims) / denom if denom > 0 else 0.0
            else:
                family_score = float(np.mean([x["weighted_anchor_score"] for x in top_sims]))
            avg_elem = float(np.mean([x["S_elem"] for x in top_sims]))
            avg_framework = float(np.mean([x["S_framework"] for x in top_sims]))
            avg_chem = float(np.mean([x["S_chem"] for x in top_sims]))
        else:
            family_score = avg_elem = avg_framework = avg_chem = 0.0

        record = row.drop(labels=["_comp", "_elem_set"]).to_dict()
        record["family_score"] = float(family_score)
        record["family_score_mode"] = "chemistry_aware_no_group_period"
        record["family_topk_detail"] = json.dumps(top_sims, ensure_ascii=False)
        record["topk_avg_S_elem"] = avg_elem
        record["topk_avg_S_framework"] = avg_framework
        record["topk_avg_S_chem"] = avg_chem
        if top_sims:
            best = top_sims[0]
            record["best_anchor_formula"] = best["anchor_formula"]
            record["best_anchor_level"] = best["anchor_level"]
            record["best_S_elem"] = best["S_elem"]
            record["best_S_framework"] = best["S_framework"]
            record["best_S_chem"] = best["S_chem"]
            record["best_S_anchor"] = best["S_anchor"]
            record["best_weighted_anchor_score"] = best["weighted_anchor_score"]
        else:
            record["best_anchor_formula"] = None
            record["best_anchor_level"] = None
            record["best_S_elem"] = 0.0
            record["best_S_framework"] = 0.0
            record["best_S_chem"] = 0.0
            record["best_S_anchor"] = 0.0
            record["best_weighted_anchor_score"] = 0.0
        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Compute chemistry-aware anchor-family score with group/period removed from S_chem.")
    parser.add_argument("--candidate_csv", type=str, default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--anchor_csv", type=str, default=str(DEFAULT_ANCHOR_CSV))
    parser.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--anchor_levels", type=str, default="", help='e.g. "A,B"; empty means all')
    parser.add_argument("--elem_w", type=float, default=1.0/3.0)
    parser.add_argument("--framework_w", type=float, default=1.0/3.0)
    parser.add_argument("--chem_w", type=float, default=1.0/3.0)
    parser.add_argument("--oxygen_weight", type=float, default=0.2)
    parser.add_argument("--old_weight_style", action="store_true", help="Use mean(S_anchor*anchor_weight), like old weighted_jaccard style.")
    args = parser.parse_args()

    wsum = args.elem_w + args.framework_w + args.chem_w
    if wsum <= 0:
        raise ValueError("elem_w + framework_w + chem_w must be > 0")
    # Normalize these three weights to avoid accidental scaling.
    elem_w = args.elem_w / wsum
    framework_w = args.framework_w / wsum
    chem_w = args.chem_w / wsum

    cdf = pd.read_csv(args.candidate_csv)
    adf = pd.read_csv(args.anchor_csv)
    only_levels = [x.strip() for x in args.anchor_levels.split(",") if x.strip()] or None

    out = compute_chemaware_family_scores(
        cdf, adf,
        top_k=args.top_k,
        only_anchor_levels=only_levels,
        elem_w=elem_w,
        framework_w=framework_w,
        chem_w=chem_w,
        oxygen_weight=args.oxygen_weight,
        normalized_anchor_weight=(not args.old_weight_style),
    )

    ensure_dir(Path(args.out_csv).parent)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved to: {0}".format(args.out_csv))
    print("[INFO] S_anchor weights: elem={:.3f}, framework={:.3f}, chem={:.3f}".format(elem_w, framework_w, chem_w))
    print("[INFO] S_chem descriptor: weighted mean/std of Z, electronegativity, covalent radius only; group and period removed.")
    formula_col = find_formula_column(out)
    cols = [formula_col, "family_score", "topk_avg_S_elem", "topk_avg_S_framework", "topk_avg_S_chem", "best_anchor_formula", "best_S_anchor"]
    print(out.sort_values("family_score", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
