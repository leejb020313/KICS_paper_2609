# Prior-shift correction — pre-registered (written 2026-10-05, before any corrected cascade was run)

## Why
`scripts/why_haiku.py` (results/analysis/why_haiku.json) showed that Sonnet 5 ranks sentences as well as or better than
Haiku 4.5 (test AUC CLEF 0.974 vs 0.975, ClaimBuster 0.925 vs 0.905), and that Haiku's higher all-call accuracy comes
from where 0.5 falls on each model's scale. The learning set is class-balanced (50% positive) while the source corpora
are not (CLEF 2024 train 24.1%, ClaimBuster 2012 26.7%; test 25.7%, 26.5%). Every threshold or intercept learned on the
learning set (all-call threshold, replacement threshold, fusion intercept) is therefore tuned for the wrong prior.
Re-weighting the learning set to the source corpus' positive rate (known without test labels) lifts Sonnet all-call to
0.934 / 0.863 — above the paper's fused cascade at 50% (0.924 / 0.853).

## What is run (scripts/prior_shift.py)
Same 100 learning-set subsamples, SVM, routing by |d| and call rates as the paper; only the class weights change from
"balanced" to the source prior pi (weight pi/0.5 for positives, (1-pi)/0.5 for negatives, on the balanced learning set):
- all-call: threshold chosen by prior-weighted learning-set accuracy
- replacement: same threshold for the called sentences
- fused: logistic regression on (d, s) fitted with prior weights
- variant "+svm": the SVM is also fitted with prior weights (changes d, routing and uncalled decisions)
For both LLMs (sonnet5, haiku45) and both datasets. Per run, McNemar (alpha 0.05) of fused@50% vs prior all-call.

## Decision rule (fixed now)
A. If prior-corrected fused@50% is significantly LOWER than prior-corrected all-call in more than half of the runs on
   either dataset (Sonnet), the paper's claim "half the calls without loss of accuracy" is false under a fair baseline
   and must be withdrawn or restated (e.g. as a cost-accuracy trade-off with the measured loss).
B. If it is never significantly lower in >= 95 of 100 runs on both datasets, the claim survives under the corrected
   baseline, and the paper must report the corrected numbers (the "+1.4~1.8%p over all-call" headline is replaced by the
   corrected comparison whatever its sign).
C. Fused vs replacement at every call rate is reported as measured under the correction.
Whatever the outcome, the uncorrected numbers are not reported as the main comparison any more, because the balanced
learning set handicaps every learned threshold.
