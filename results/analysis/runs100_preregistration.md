# 100-run re-evaluation — pre-registration (written 2026-10-04, before any 100-run result was computed)

Trigger: advisor comment on ver.2 draft, 3.2 "5회는 너무 적은 듯… 좀더 해 볼 수 없나?" (author replied "100회로 해보겠습니다").

## What changes
- Runs: 100 learning-set subsamples (80% without replacement, `np.random.default_rng(seed)`, seeds 0–99).
  Seeds 0–4 are the 5 runs already reported, so the 100 runs extend them; nothing else changes
  (same stored LLM scores, same test sets, same model/threshold/k/ρ rules, 2,000 bootstrap resamples per run).
- No new LLM calls. Only the learning-set subsample varies across runs; the test set is fixed, so the runs are
  correlated and 100 runs give a more stable mean/spread, not 100 independent replications. The paper says so.

## Reporting rules (fixed now)
1. Table 1 and Fig. 2: mean (Fig. 2 band: std) over the 100 runs.
2. Significance (McNemar, α=0.05, per run): reported as "100회 중 k회". Table-1 dagger = significantly lower than the
   proposed row in **more than half (≥51) of the 100 runs** (was ≥3 of 5).
3. Non-inferiority margins are **fixed at the values printed in the 5-run version**: δ = 0.6%p (Claude Sonnet 5),
   δ = 1.6%p (Claude Haiku 4.5). Primary statistic: number of runs (of 100) whose 95% bootstrap CI lower bound of
   (cascade@50% − stronger all-call setting) is ≥ −δ. The margin is no longer re-derived from the data
   (the 5-run version used the observed worst bound rounded up; with 100 runs that rule would just chase the minimum).
4. Wording decision for the core claim ("호출을 절반으로 줄이면서도 … 같은 수준의 정확도를 유지"):
   - keep it if fused@50% meets δ=0.6%p in ≥95 of 100 runs on **both** datasets;
   - otherwise report the counts only and weaken the abstract/conclusion to the mean result
     (e.g. "평균 정확도는 같거나 높았다"), stating the count of runs that fail.
5. Replacement@50% is reported with the same statistic and the same δ (count of runs meeting δ).
6. "Mean reach" call rates (first ρ whose 100-run mean accuracy ≥ stronger all-call mean) are recomputed and
   printed as they come out.
7. Whatever the outcome, the numbers are reported as computed; nothing is dropped because it got worse.
