# CT22 (CheckThat! 2022 1A English tweets): analysis fixed before seeing any test result

Written 2026-09-30 while the Sonnet batches were still being scored; nothing on CT22 test had been computed.

Protocol: identical to CLEF 2024 / ClaimBuster in the paper (scripts/final_eval.py --full, 5 x 80% subsamples of
the learning set, McNemar on each run). Learning set = class-balanced train subsample (894), test = dev + dev_test +
test gold (918). Gemma 3 4B + NN-PPI is not run (no GPU); NN-PPI is compared through "all-call LLM + NN-PPI".
Operating point: call rate 50%, the paper's fixed operating point (not re-tuned on CT22). The learning-set curve is
reported alongside, to state whether the paper's selection rule would also have picked 50% on CT22.

The paper's claims "hold" on CT22 if all of these hold:
1. fused@50% is not significantly worse than the stronger all-call LLM setting on a majority (>= 3/5) of runs;
2. fused@50% is significantly better than all-call LLM + NN-PPI on >= 3/5 runs;
3. fused reaches the stronger all-call accuracy (mean) at a lower call rate than replacement;
4. fused mean accuracy >= replacement mean accuracy at every call rate > 0.
Also reported, not a pass/fail criterion: embedding SVM vs all-call LLM; LLM raw recall/precision.

If any of 1-4 fails, CT22 is not silently dropped: the result is shown to the user and either reported in the paper
or stated as a limitation.
