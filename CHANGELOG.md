# Changelog

## 2026-07-10 — v3 bug fixes + top-N retry restoration + SOAP integration

Three bug fixes to the substitution engine and one code restoration to
the benchmark script. Legacy benchmark CSVs are kept intact; new
benchmark runs produce canonical numbers matching the paper's
Table 2 and Table 3 valid-subset definitions.

### Fixed
- `csp_workflow_mp.substitution_engine.parse_formula` now delegates to
  `pymatgen.core.Composition`. The previous regex-only parser silently
  dropped parenthesised outer multipliers (e.g., `MgP2(H8O5)2` was
  parsed as `{Mg:1, P:2, H:8, O:5}` instead of `{Mg:1, P:2, H:16, O:10}`).
  Affected 23.3 % of Materials Project formulas.
- `csp_workflow_mp.substitution_engine._solve_and_return` fallback path
  now returns `success=False` when both `_solve_one_to_one` and
  `_solve_multi_element` fail. Previously it returned `success=True` with
  no assignment data; `apply_substitution` on that result returned the
  unchanged template as a silent "fake success" (62 K=1 targets in the
  earlier benchmark CSV, 18.9 % of the valid subset).
- `scripts/05_run_benchmark.py` target sampling changed from
  `rng.choice(len(df), size=n)` to `rng.choice(len(df), size=500)[:n]`
  so small-`n` runs use the same target order as the head of a
  500-sample run.

### Added
- `scripts/05_run_benchmark.py` restored the top-N template rank-order
  retry loop that had been removed in an earlier cleanup. Added
  `--n-retry` CLI flag (default 50, matching the paper's Methods §4.3
  policy) and `template_rank` column to `benchmark_raw.csv`.
- `scripts/05_run_benchmark.py` computes SOAP cosine similarity via
  `structural_eval_helpers.soap_cosine_similarity` and writes the
  `soap_cosine` column to `benchmark_raw.csv` alongside `sg_match`,
  `sm_match`, and `rmsd_angstrom`.
- Two regression tests in `tests/test_substitution_engine.py`:
  `test_parse_formula_handles_parentheses` and
  `test_no_fake_success_when_all_solvers_fail`.

### Changed
- `README.md` slimmed from 264 lines to 97 lines. The main-result
  headline was updated to the v3 canonical numbers (40.1 % → 63.0 %,
  a 22.9 pp gain). The partial-occupancy note was corrected from the
  previous inaccurate "dominant-species approximation" wording to
  "excluded from the relaxation step".
- `.gitignore` widened `results_windows_*/` to `results_*/`.

## 2026-07-06 — Initial paper-companion release

Companion code for Wu Y.-J. & Xu Y., *Space-Group-Guided Template
Retrieval for Composition-to-Structure Prediction* (manuscript in
preparation).
