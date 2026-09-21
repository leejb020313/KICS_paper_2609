# Safety-Flag domain-transfer experiment

Applies the project's validated clip+trust-gate+local-ridge-regression correction
(`theta_gated_localreg`, ported from
`experiments/prompt-reorder-ablation/full_pipeline_compare.py`) to
[Safety-Flag](https://arxiv.org/abs/2609.19072) content-moderation items, and
compares it against the one baseline its authors tried (global temperature
scaling), on the same held-out test split for a fair comparison.

## Steps

1. `build_joined_data.py` -- Safety-Flag's public repo (`data/results/*.jsonl`)
   ships only item IDs + model outputs, no raw text (see their README). We
   verified the `id` field is the item's native row index in the source
   benchmark (checked exactly against `PKU-Alignment/BeaverTails` 30k_test row
   262: all 13 category fields match) and joined text back in from HuggingFace
   for BeaverTails and XSTest. Ethics commonsense does NOT line up this way
   (checked two official mirrors, sequence diverges at index 4) -- skipped.
2. `run_experiment.py` -- for 5 models x 2 datasets, 10 random 50/25/25
   pool/tune/test splits: raw ECE, our own temperature-scaling reproduction
   (fit on pool, matches the authors' own reported 2.8-6.0x ECE reduction when
   run through their code directly -- see below), and our gate+localreg
   correction (gate thresholds tuned on tune split only).
3. `significance_check.py` -- paired Wilcoxon signed-rank test across the 10
   seeds per condition.

## Gate (reproduced the authors' own claim first, no LLM re-inference needed)

Ran `reproduce.sh` from the Safety-Flag repo unmodified against their bundled
per-item outputs:

| model | ECE pre | ECE post (temp scaling) | reduction |
|---|---|---|---|
| Mistral-7B | 0.198 | 0.035 | 6.04x |
| Llama-3.1-8B | 0.368 | 0.118 | 3.19x |
| Qwen2.5-7B | 0.194 | 0.051 | 4.13x |
| Qwen2.5-32B | 0.125 | 0.038 | 3.52x |
| gemma-2-9b | 0.169 | 0.037 | 5.05x |
| OLMo-2-7B | 0.125 | 0.048 | 2.84x |

Matches the paper's claimed 2.8-6.0x reduction exactly -- gate passed. Note
this is pooled across all 7 benchmarks (n=1398/model); our own experiment
below is per-benchmark (n=200-300) to keep kNN neighbors within one semantic
domain, so its absolute ECE is not directly comparable to this table -- only
the temp-vs-ours comparison within each of our own splits is apples-to-apples.

## Result -- mixed, not a clean sweep

Our correction beats our own reproduction of temperature scaling (lower ECE)
in 7/10 (model, dataset) conditions; Wilcoxon signed-rank across 10 seeds:

- **5/10 significant wins for ours** (p<0.05): Llama-3.1-8B/xstest,
  OLMo-2-7B/xstest, Qwen2.5-7B/beavertails, gemma-2-9b/beavertails,
  gemma-2-9b/xstest.
- **1/10 significant loss** (p=0.002): Llama-3.1-8B/beavertails -- temp
  scaling wins by a wide margin here (ECE 0.132 vs our 0.243). Llama-3.1-8B is
  also the worst-calibrated model on this benchmark in the authors' own
  leaderboard (55% false-alarm rate, near-uniform over-flagging) -- plausible
  mechanistic read: when miscalibration is a uniform global bias, a single
  temperature parameter corrects it better than a local (per-item) method;
  local correction only has an edge when miscalibration is heterogeneous
  across items. Not verified further, flagged as a real limitation.
- **4/10 not significant**: Mistral-7B/beavertails, Mistral-7B/xstest,
  OLMo-2-7B/beavertails, Qwen2.5-7B/xstest.

**Framing if this goes in the paper:** "our already-validated correction
generalizes to a second, independent domain and beats the authors' own
baseline in most but not all conditions" -- not a clean-sweep claim like the
CI-coverage headline result. Report the loss condition and the non-significant
ones plainly; do not cherry-pick the 5 wins.
