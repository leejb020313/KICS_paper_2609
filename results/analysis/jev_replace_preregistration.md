# JEV -> Sonnet replacement with a learning-set call rate — pre-registered (2026-10-06, before running)

## Why
results/analysis/jev_prior_sonnet5_{clef,cb}.json: under the prior correction, JEV -> Sonnet replacement at 30-40%
calls averages 0.936 / 0.864-0.866 vs Sonnet all-call 0.935 / 0.862. Those rates were read off the test curve, and no
per-run test was made. Here the call rate is chosen on the learning set only.

## What is run (scripts/jev_replace.py), Sonnet 5 scores, stored JEV scores, 100 learning-set subsamples (80%)
Prior-weighted thresholds as in scripts/jev_prior.py (CLEF 24.1%, ClaimBuster 26.7%).
- Learning-set curve: for each call rate b, out-of-fold (5-fold) replacement predictions on the learning subsample
  (JEV and Sonnet thresholds fitted on the training folds), scored by prior-weighted accuracy.
- Call rate: the smallest b whose curve value is within 0.3 pp of the curve's maximum.
- Test: jevrep at the selected rate vs Sonnet all-call (prior threshold), McNemar per run; also vs the SVM -> Sonnet
  replacement at 50% (prior-corrected, as in scripts/prior_shift.py, recomputed here).

## Decision rule (fixed now)
GO (rewrite the paper around the prior-corrected JEV -> LLM cascade) only if, on both datasets:
1. jevrep at the selected rate is significantly lower than Sonnet all-call in at most 5 of 100 runs, and
2. the mean selected call rate is at most 50%.
Otherwise NO-GO; results are reported as measured.
