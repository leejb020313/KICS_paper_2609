# Option 1 — same operating-point rule for fused and replacement (pre-registered 2026-10-05, before running)

## Problem
Replacement uses an LLM threshold t chosen on the learning set to maximise accuracy; the fused cascade decides at a
fixed probability 0.5 of a class-balanced logistic regression. The two cascades therefore do not use the same rule,
and on ClaimBuster the fused cascade loses on positive-class F1 (the CheckThat! official measure).

## What is run (scripts/operating_point.py)
For each learning-set subsample (80%, seeds 0-99, as in final_eval.py), call rate ρ = 50% (and 30% as a secondary
point), routing unchanged (smallest |d| first):
- replacement: LLM threshold t chosen on the learning subsample;
- fused: logistic regression on (d, s) as before; decision threshold τ on its out-of-fold learning-set probabilities
  (5-fold, same folds that produce d);
- all-call LLM with the same rule, as reference.
Two objectives, applied identically to every method: (A) maximise learning-set accuracy, (B) maximise learning-set
positive-class F1. Threshold grid 0.02-0.98 step 0.01. Test metrics: accuracy and positive-class F1 (mean over runs,
runs where fused > replacement). LLM scores: Sonnet 5 Oct-04 clean (primary), Sonnet 5 Sept, Haiku 4.5.

## Decision rule
Option 1 "works" if, with the primary scores, under objective B the fused cascade's mean F1 ≥ replacement's on both
datasets and is higher in a majority of runs, while under objective A fused accuracy stays ≥ replacement's on both
datasets. Otherwise it is reported as not fixing the F1 gap. Results are reported whatever they are.
