# Option 2 — neighbour-label rate as a third fusion input (pre-registered 2026-10-05, before running option 2)

## Idea
NN-PPI's signal is the labels of semantically similar learning-set sentences. Add it to the fusion: the logistic
regression takes (d, s, r) where r = positive-label rate of the k = 10 nearest learning-set sentences (cosine on the
same all-MiniLM-L6-v2 embeddings; leave-self-out on the learning set). Routing (smallest |d| first), call rate, data,
runs (100 × 80% subsamples) and the operating-point rules (0.5 / max-accuracy / max-F1 on the learning set) are the
same as option 1 (scripts/operating_point.py --knn).

## Decision rule (primary scores: Sonnet 5 Oct-04 clean)
Option 2 "works" if, at 50% calls, the 3-input fused cascade beats the 2-input fused cascade under the same rule
on BOTH datasets on accuracy AND positive-class F1 (mean, and higher in a majority of runs), and its gap to
replacement is larger than the 2-input gap. Sept Sonnet and Haiku are reported as robustness checks.
Results are reported whatever they are; if it does not work, the paper keeps the 2-input fusion.
