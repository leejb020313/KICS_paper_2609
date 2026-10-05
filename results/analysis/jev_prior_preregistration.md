# JEV as the first stage, under the prior-shift correction — pre-registered (2026-10-05, before running)

## Why
With the source-prior correction (results/analysis/prior_shift_preregistration.md) the embedding-SVM cascade fails:
Sonnet all-call 0.935 / 0.862 beats the fused cascade at 50%, and replacement beats fusion. CAUC's own gate agrees:
the complementarity rate of SVM vs LLM on the called half is about −5%p (results/analysis/cauc_cr.json). JEV is a much
stronger first stage than the SVM (Sept: 0.901 / 0.850 vs 0.859 / 0.790) from a different model family, i.e. closer to
the case in which CAUC reports gains from fusion. JEV scores for every learning and test sentence are already stored
(results/jev/nnppi/, jev-1.13.0), so no API call is made.

## What is run (scripts/jev_prior.py), Sonnet 5 scores (Oct-04 clean), 100 learning-set subsamples (80%)
All thresholds / intercepts fitted with source-prior class weights (CLEF 24.1%, ClaimBuster 26.7%):
- sonnet_all: Sonnet on every sentence, prior-weighted threshold
- jev_alone: JEV on every sentence, prior-weighted threshold
- jevrep_b: JEV -> Sonnet, the called sentences take the sonnet_all decision (replacement)
- jevfuse_b: JEV -> Sonnet, the called sentences take a prior-weighted logistic regression on (logit JEV, s)
- routing: call the fraction b of sentences with the smallest |logit JEV - logit t_JEV|; b in the paper's grid
- call rate: chosen on the learning set only, as the smallest b after which the out-of-fold prior-weighted
  learning-set accuracy of jevfuse rises by at most 0.3 pp (the main paper's rule); 30% and 50% also reported
- CR: CAUC Eq. 5 for JEV vs Sonnet (both Platt-calibrated with prior weights, out-of-fold) on the learning subsample,
  over the called fraction at the selected rate
- McNemar (alpha 0.05) per run.

## Decision rule (fixed now) — JEV becomes the main first stage only if ALL hold, on both datasets
1. CR > 0 on the called fraction in more than half of the runs;
2. jevfuse at the selected rate is significantly lower than sonnet_all in at most 5 of 100 runs;
3. jevfuse at the selected rate has a higher mean accuracy than jevrep at the same rate, is significantly lower in at
   most 5 runs, and is significantly higher in more than half of the runs on at least one dataset.
Otherwise JEV is not adopted, and the result is reported as measured.
