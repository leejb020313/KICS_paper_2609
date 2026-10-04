# Clean re-scoring of Claude Sonnet 5 — pre-registration (2026-10-04, before any re-scored batch was read)

## Why
The September Sonnet 5 scores (paper's main results) came from plain `claude -p` run inside the repository, a mode
that loads user/project settings (CLAUDE.md, output style, hooks) into the context. Claude Haiku 4.5 was scored in
clean mode (`--restricted --strict-mcp-config`, empty temp dir). On the 920 test sentences that were scored both
ways, the plain-mode Sonnet scores flag 12% of sentences at 0.5 (labels: 25%), the clean-mode ones 23%
(acc@0.5 0.834 vs 0.860; Haiku clean 0.846). The "Haiku beats Sonnet" pattern and part of "the LLM's criterion
differs from the labels" are therefore confounded with the scoring environment.

## What is re-done
- Every Sonnet 5 prompt file (results/llm_scores/sonnet5/*.txt, 206 batches of 40: CLEF/ClaimBuster test + learning
  sets) is re-scored with `claude -p <prompt> --model claude-sonnet-5 --output-format json --restricted
  --strict-mcp-config` from an empty temp dir — the exact command used for Haiku 4.5 (scripts/run_haiku_batch.sh),
  Claude Code 2.1.285. The model id is checked in each reply (modelUsage).
- The old outputs move to results/llm_scores/sonnet5_cli_default/ (kept, not used). Prompts are unchanged.
- A batch whose reply cannot be parsed or misses ids is re-run (up to 3 times); remaining failures are reported.
- Everything downstream is re-run unchanged: 100 runs, same rules, margins fixed at 0.6pp (Sonnet) / 1.6pp (Haiku)
  as in results/analysis/runs100_preregistration.md, ρ = 50% fixed operating point.

## What will be reported (whatever the outcome)
1. Old vs new Sonnet: flag rate, recall/precision at 0.5, AUC, all-call accuracy (raw / tuned).
2. Fused@50% vs the stronger all-call setting: mean difference, runs significantly lower, runs meeting −0.6pp.
3. Fused vs replacement at every call rate; call rate at which each reaches the all-call mean.
4. Fused@50% vs all-call + NN-PPI.
5. Sonnet vs Haiku under identical scoring.
The paper text is rewritten to these numbers. If fused@50% is on average below the stronger all-call setting, the
abstract says so (accuracy cost of halving the calls), instead of "higher on average".

## Addendum (after scoring, before any evaluation)
- First pass hit the subscription session limit after 73 batches; the 133 limit replies were deleted and re-run after the reset.
- 15 batches came back with ids renumbered 0-39 (cb_0760 in the first pass, 14 in the second); all re-run (originals kept in results/llm_scores/checks/sonnet5_clean_renumbered/). After 3 re-runs one learning-set batch (clefcal_1240) was still renumbered; its keys were mapped by position to ids 1240-1279 (positional correlation with the old scores 0.914 vs 0.949 for a correctly numbered batch).
- Final: 206/206 batches, model claude-sonnet-5 on every call (scripts/check_clean_scores.py).
