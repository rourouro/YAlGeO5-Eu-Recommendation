# Host and family scores

Two scripts implement the scores used to rank oxide hosts.

- `01_compute_host_score_parametric.py` writes `host_score`.
- `02_compute_family_score_chemaware_no_gp.py` writes `family_score`.

Rank hosts with

```text
S_total = 0.4 * host_score + 0.6 * family_score
```

Run the host script first, then the family script. Default inputs and outputs are next to these scripts:

- `candidate_pool_core.csv`
- `data/anchors.csv`
- `outputs/candidates_with_host_score.csv`
- `outputs/candidates_with_family_score.csv`

Override them with `--candidate_csv`, `--anchor_csv`, and `--out_csv`.

The candidate table needs `host_formula` (or `formula`), `band_gap`, `energy_above_hull`, and either `n_elements` or `elements`. If present, `has_y`, `has_gd`, `has_la`, `has_lu`, `has_sc`, and `has_in` set the rare-earth-like term. The anchor table needs `host_formula` and `anchor_level` (`A+`, `A`, or `B`). Parentheses in formulas are expanded.

The anchor table is not included here.
