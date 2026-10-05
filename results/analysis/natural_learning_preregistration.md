# Natural-ratio learning set instead of re-weighting — pre-registered (2026-10-06, before running)

## Why
The prior-shift analyses re-weighted the class-balanced learning set to the source corpus' positive rate. The user
asked for the direct version: build a learning set that has the natural ratio and refit everything without weights.
No new LLM/JEV scores are needed: from the balanced learning set, keep every negative and a random subset of positives
so that positives make up the source rate (CLEF 24.1%, ClaimBuster 26.7%); then take the usual 80% subsample.

## What is run (scripts/natural_learning.py), 100 runs, Sonnet 5 and Haiku 4.5, both datasets
Fitted on the natural-ratio subsample with NO class weights (SVM class_weight=None, logistic regression unweighted,
thresholds by plain accuracy):
- all-call LLM; SVM -> LLM replacement and fusion at every call rate; JEV alone; JEV -> LLM replacement and fusion
  (JEV rows: Sonnet only, as before); JEV replacement at a learning-set call rate (smallest b within 0.3 pp of the
  out-of-fold curve maximum).
McNemar per run.

## Decision rule (fixed now)
The re-weighting conclusions are CONFIRMED if, with Sonnet, on both datasets: (1) SVM-fused@50% is not higher than
all-call (mean) and (2) SVM-fused@50% is not higher than SVM-replacement@50% (mean). If either fails on a dataset, the
re-weighting result is NOT confirmed there and the paper discussion is reopened. JEV rows are reported as measured.
