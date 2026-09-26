# Fused cascade: cheap embedding classifier + frontier LLM on uncertain claims

Run everything from this directory (`experiments/fusion-cascade/`).

1. Frontier scores (Claude Sonnet 5 via `claude -p`, 40 claims/batch, criteria-only prompt):
   `ls batches/*.txt | xargs -P 6 -n 1 bash run_b.sh` — outputs `batches/*.out` are committed.
   `frontier_eval_set.json` = CLEF test (318, full) + ClaimBuster test (800, random seed 0);
   `frontier_calib_set.json` = the parse_ok calib rows, index-aligned with `batches/{clefcal,cbcal}_*.out`.
2. Final evaluation (5 seeds of 80% calib subsamples without replacement, paired bootstrap, McNemar):
   `uv run --isolated --no-project --with sentence-transformers --with scikit-learn --with scipy python final_eval.py`
   → `final_results.json`, `final_eval.log`.
3. Fairness checks: `single/` = 80 CLEF test claims scored one-by-one with the paper's few-shot prompt
   (AUC .991 vs batched .990 on the 69 parsed); `rerun/` = a second batched pass over CLEF test (score corr .973).
4. Figure: artifacts `nnppi-paper/figures/cascade_budget.py` reads a copy of `final_results.json`.

Note: an earlier run bootstrap-resampled calib WITH replacement; duplicates leaked across the stacker's
CV folds and deflated results. Fixed to subsampling without replacement (see comment in final_eval.py).
