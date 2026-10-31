# -*- coding: utf-8 -*-
"""
02_compute_family_score.py

Anchor-family score for ranking oxide hosts.

For each candidate-anchor pair:

    S_anchor = (S_elem + S_framework + S_chem) / 3

where
    S_elem      : weighted elemental-set similarity, with O down-weighted;
    S_framework : similarity of oxide framework-forming cations and cation groups;
    S_chem      : similarity of composition-derived cation descriptors.

S_family is the evidence-weighted mean of S_anchor over the top k anchors.
Candidates whose composition matches an anchor are removed before the result is written.

The score is a ranking priority. It does not predict absolute photoluminescence intensity.
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

# Element properties used by S_chem.
# Values: atomic number Z, Pauling electronegativity, covalent radius in pm.
ELEMENT_PROPS = {
    "H": (1, 2.20, 31), "Li": (3, 0.98, 128), "Be": (4, 1.57, 96),
    "B": (5, 2.04, 84), "C": (6, 2.55, 76), "N": (7, 3.04, 71),
    "O": (8, 3.44, 66), "F": (9, 3.98, 57),
    "Na": (11, 0.93, 166), "Mg": (12, 1.31, 141), "Al": (13, 1.61, 121),
    "Si": (14, 1.90, 111), "P": (15, 2.19, 107), "S": (16, 2.58, 105),
    "Cl": (17, 3.16, 102), "K": (19, 0.82, 203), "Ca": (20, 1.00, 176),
    "Sc": (21, 1.36, 170), "Ti": (22, 1.54, 160), "V": (23, 1.63, 153),
    "Cr": (24, 1.66, 139), "Mn": (25, 1.55, 139), "Fe": (26, 1.83, 132),
    "Co": (27, 1.88, 126), "Ni": (28, 1.91, 124), "Cu": (29, 1.90, 132),
    "Zn": (30, 1.65, 122), "Ga": (31, 1.81, 122), "Ge": (32, 2.01, 120),
    "As": (33, 2.18, 119), "Se": (34, 2.55, 120), "Br": (35, 2.96, 120),
    "Rb": (37, 0.82, 220), "Sr": (38, 0.95, 195), "Y": (39, 1.22, 190),
    "Zr": (40, 1.33, 175), "Nb": (41, 1.60, 164), "Mo": (42, 2.16, 154),
    "Ru": (44, 2.20, 146), "Rh": (45, 2.28, 142), "Pd": (46, 2.20, 139),
    "Ag": (47, 1.93, 145), "Cd": (48, 1.69, 144), "In": (49, 1.78, 142),
    "Sn": (50, 1.96, 139), "Sb": (51, 2.05, 139), "Te": (52, 2.10, 138),
    "I": (53, 2.66, 139), "Cs": (55, 0.79, 244), "Ba": (56, 0.89, 215),
    "La": (57, 1.10, 207), "Ce": (58, 1.12, 204), "Pr": (59, 1.13, 203),
    "Nd": (60, 1.14, 201), "Sm": (62, 1.17, 198), "Eu": (63, 1.20, 198),
    "Gd": (64, 1.20, 196), "Tb": (65, 1.20, 194), "Dy": (66, 1.22, 192),
    "Ho": (67, 1.23, 192), "Er": (68, 1.24, 189), "Tm": (69, 1.25, 190),
    "Yb": (70, 1.10, 187), "Lu": (71, 1.27, 187), "Hf": (72, 1.30, 175),
    "Ta": (73, 1.50, 170), "W": (74, 2.36, 162), "Re": (75, 1.90, 151),
    "Os": (76, 2.20, 144), "Ir": (77, 2.20, 141), "Pt": (78, 2.28, 136),
    "Au": (79, 2.54, 136), "Hg": (80, 2.00, 132), "Tl": (81, 1.62, 145),
    "Pb": (82, 2.33, 146), "Bi": (83, 2.02, 148),
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


def composition_key(formula: str):
    """Composition identity used to exclude reference anchors from ranking.

    Parentheses are expanded and stoichiometric amounts are normalized to
    atomic fractions, so Ca3Y2(GeO4)3, Ca3Y2Ge3O12, and Ca6Y4Ge6O24
    are recognized as the same chemical composition.
    """
    comp = parse_formula(formula)
    if not comp:
        return None
    total = sum(float(amt) for amt in comp.values())
    if total <= 0:
        return None
    return tuple(sorted((el, round(float(amt) / total, 8)) for el, amt in comp.items()))


def drop_anchor_hosts(candidates: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Remove candidates that are themselves anchors. Scoring is unchanged."""
    cand_col = find_formula_column(candidates)
    anch_col = "host_formula" if "host_formula" in anchors.columns else find_formula_column(anchors)
    anchor_keys = set()
    for formula in anchors[anch_col].dropna().astype(str):
        key = composition_key(formula)
        if key:
            anchor_keys.add(key)
    keep = pd.Series(
        [composition_key(formula) not in anchor_keys for formula in candidates[cand_col].astype(str)],
        index=candidates.index,
    )
    kept = candidates.loc[keep].reset_index(drop=True)
    n_removed = len(candidates) - len(kept)
    print("[INFO] removed {0} candidates whose composition matches an anchor; {1} retained for ranking.".format(n_removed, len(kept)))
    return kept


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
    """Stoichiometry-weighted cation descriptor used by S_chem.

    Oxygen is excluded. Each cation contributes its atomic number, Pauling
    electronegativity and covalent radius. The descriptor holds the weighted
    mean and weighted standard deviation of these three properties, so its
    length is 6.
    """
    items = [(e, amt) for e, amt in comp.items() if e != "O" and e in ELEMENT_PROPS and amt > 0]
    if not items:
        return np.zeros(6, dtype=float)
    elems = [e for e, _ in items]
    amounts = np.array([amt for _, amt in items], dtype=float)
    weights = amounts / amounts.sum()

    props = np.array([ELEMENT_PROPS[e] for e in elems], dtype=float)

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


def compute_family_scores(
    candidate_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    top_k: int = 3,
    level_weight: Dict[str, float] = None,
    only_anchor_levels: List[str] = None,
    elem_w: float = 1.0 / 3.0,
    framework_w: float = 1.0 / 3.0,
    chem_w: float = 1.0 / 3.0,
    oxygen_weight: float = 0.2,
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
            denom = sum(x["anchor_weight"] for x in top_sims)
            family_score = sum(x["anchor_weight"] * x["S_anchor"] for x in top_sims) / denom if denom > 0 else 0.0
            avg_elem = float(np.mean([x["S_elem"] for x in top_sims]))
            avg_framework = float(np.mean([x["S_framework"] for x in top_sims]))
            avg_chem = float(np.mean([x["S_chem"] for x in top_sims]))
        else:
            family_score = avg_elem = avg_framework = avg_chem = 0.0

        record = row.drop(labels=["_comp", "_elem_set"]).to_dict()
        record["family_score"] = float(family_score)
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
    parser = argparse.ArgumentParser(description="Compute the anchor-family score.")
    parser.add_argument("--candidate_csv", type=str, default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--anchor_csv", type=str, default=str(DEFAULT_ANCHOR_CSV))
    parser.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--anchor_levels", type=str, default="", help='e.g. "A,B"; empty means all')
    parser.add_argument("--elem_w", type=float, default=1.0/3.0)
    parser.add_argument("--framework_w", type=float, default=1.0/3.0)
    parser.add_argument("--chem_w", type=float, default=1.0/3.0)
    parser.add_argument("--oxygen_weight", type=float, default=0.2)
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

    out = compute_family_scores(
        cdf, adf,
        top_k=args.top_k,
        only_anchor_levels=only_levels,
        elem_w=elem_w,
        framework_w=framework_w,
        chem_w=chem_w,
        oxygen_weight=args.oxygen_weight,
    )

    out = drop_anchor_hosts(out, adf)

    ensure_dir(Path(args.out_csv).parent)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved to: {0}".format(args.out_csv))
    print("[INFO] S_anchor weights: elem={:.3f}, framework={:.3f}, chem={:.3f}".format(elem_w, framework_w, chem_w))
    print("[INFO] S_chem uses the weighted mean and standard deviation of atomic number, electronegativity and covalent radius.")
    formula_col = find_formula_column(out)
    cols = [formula_col, "family_score", "topk_avg_S_elem", "topk_avg_S_framework", "topk_avg_S_chem", "best_anchor_formula", "best_S_anchor"]
    print(out.sort_values("family_score", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
