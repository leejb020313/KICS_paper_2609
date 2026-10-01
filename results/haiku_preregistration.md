# Second LLM (Claude Haiku 4.5) — pre-registered criteria (written 2026-09-30, before any Haiku score was seen)

Same batches (results/frontier/batches/*.txt: NN-PPI criteria prompt, 40 statements per call), same learning sets,
5 runs, McNemar per run, same learning-set rate rule as the paper. Only the LLM changes (sonnet -> haiku).
Scored in clean CLI mode (--restricted --strict-mcp-config, neutral cwd), model id logged per call.

Criteria at the paper's call rate (50%) and at the rate chosen by the learning-set rule:
1. fused is NOT significantly lower than the better Haiku all-call setting (raw / tuned) in >= 3/5 runs.
2. fused is significantly higher than Haiku all-call + NN-PPI in >= 3/5 runs.
3. fused mean accuracy >= replacement mean accuracy (significance reported as k/5).

Whatever the outcome, it is reported; a failed criterion is disclosed, not dropped.
