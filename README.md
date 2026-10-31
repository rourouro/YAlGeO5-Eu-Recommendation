# Host and family scores

Two scripts implement the scores used to rank oxide hosts.

- `01_compute_host_score.py` writes `host_score`.
- `02_compute_family_score.py` writes `family_score`.

Rank hosts with

```text
S_total = 0.4 * host_score + 0.6 * family_score
```

Before that sort, remove every candidate whose composition matches an anchor. Chemical compositions are compared after expanding parentheses and normalizing stoichiometric ratios. For example, `Ca3Y2(GeO4)3`, `Ca3Y2Ge3O12`, and `Ca6Y4Ge6O24` are equivalent for the exclusion rule. Script 02 applies this exclusion after scoring all input candidates and before saving `outputs/candidates_with_family_score.csv`. The candidate pool contains 1319 entries. The number retained for the final ranking depends on its overlap with the supplied reference anchors and is reported in the script log.

Run the host script first, then the family script. Default inputs and outputs are next to these scripts:

- `candidate_pool_core.csv`
- `data/anchors.csv`
- `outputs/candidates_with_host_score.csv`
- `outputs/candidates_with_family_score.csv`

Override them with `--candidate_csv`, `--anchor_csv`, and `--out_csv`.

The candidate table needs `host_formula` (or `formula`), `band_gap`, `energy_above_hull`, and either `n_elements` or `elements`. If present, `has_y`, `has_gd`, `has_la`, `has_lu`, `has_sc`, and `has_in` set the rare-earth-like term. The anchor table needs `host_formula` and `anchor_level` (`A+`, `A`, or `B`). Parentheses in formulas are expanded.

The anchor table is not included here because it contains unpublished experimental data. Accordingly, the full anchor-family scores and final ranking cannot be independently regenerated from this repository alone; the complete anchor dataset is available from the corresponding author upon reasonable request. The publicly included candidate data and host-scoring script can be used to reproduce `host_score`.

### Experimental selection and descriptor coverage

The original pool of 1,319 oxide candidates includes 12 compositions containing Ac, Pa, or Th. These materials were not selected for experimental validation because of radiological safety, experimental feasibility, and practical application considerations.

These 12 entries remain in the computational candidate pool. In `02_compute_family_score.py`, the elemental-property table (`ELEMENT_PROPS`) does not contain Ac, Pa, or Th; for these compositions, `S_chem` is therefore calculated using only cations with available properties, while `S_elem` retains the complete element sets.
