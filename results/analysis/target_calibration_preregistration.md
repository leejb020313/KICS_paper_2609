# Call rate from a small labelled sample of the target data — pre-registered (2026-10-06, before running)

## Why
The learning sets (CLEF train, ClaimBuster 2012) do not reflect the test sets: Sonnet's accuracy is ~0.87 on them and
0.935 on the test, so any call rate chosen on them is too low (results/analysis/jev_replace_sonnet5_*.json,
natural_*.json). JEV-as-a-Judge (arXiv:2609.26550, p. 14) chooses its threshold "on a small set of labeled items from
the target workload" with a lower-confidence-bound rule. We do the same with stored scores only (no API call).

## Design (scripts/target_calibration.py), Sonnet 5 scores, stored JEV scores (jev-1.13.0)
For each dataset, pool the whole held-out set (CLEF 1,691; ClaimBuster 2,745). 100 random splits: 800 sentences are
the target calibration sample (labelled), the rest is the evaluation set.
On the calibration sample only: Sonnet threshold, JEV threshold (plain accuracy, the sample has the target ratio), and
the call rate. For a call rate b (grid 0, 5, 10, 20, ..., 100%), d_i = correct(cascade) - correct(Sonnet all-call)
per calibration sentence; the selected rate is the smallest b whose one-sided 95% lower bound
mean(d) - 1.645 sd(d)/sqrt(n) >= -m. Primary margin m = 1 pp; JEV-as-a-Judge's 2 pp reported as secondary.
Cascades: JEV -> Sonnet replacement (primary); SVM -> Sonnet replacement (SVM trained once on the learning set,
routing by |d|) as reference. Evaluation: accuracy on the evaluation split, McNemar vs Sonnet all-call per split.

## Decision rule (fixed now) — JEV -> Sonnet with a target-calibrated call rate is adopted only if, on both datasets
(primary margin): (1) significantly lower than Sonnet all-call in at most 5 of 100 splits, and (2) mean selected call
rate <= 50%. Otherwise NO-GO; reported as measured.
