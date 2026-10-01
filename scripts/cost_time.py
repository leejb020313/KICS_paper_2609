"""LLM cost (USD, list price) and processing time of the cascade vs calling the LLM on every sentence.

The September scoring runs kept only the text reply, so latency and token usage are re-measured on a sample of
23 test batches (every 5th test batch, 40 sentences each), re-sent to claude-sonnet-5 in clean CLI mode
(scripts/run_sonnet_costcheck.sh -> results/llm_scores/checks/sonnet5_cost/). The CLI adds its own system prompt to
every request; its size is measured with a trivial prompt (sonnet5_cost/overhead_*.json) and subtracted, so
the prompt tokens below are those of our prompt alone (NN-PPI criteria + 40 statements), as a direct API call would
send them. Only single-turn calls are used (in a few calls the CLI took a second turn, which inflates latency).
Cost = input tokens x $2/M + output tokens x $10/M (Claude Sonnet 5 list price, no caching or batch discount).
Time = sequential API time of the LLM batches + embedding and SVM on CPU for every sentence (the cascade only).

    uv run --locked python scripts/cost_time.py      -> results/paper/cost_time.json
"""
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import PAPER, SCORES, embed, load_frontier  # noqa: E402

DIR = os.path.join(SCORES, "checks", "sonnet5_cost")
PRICE_IN, PRICE_OUT = 2 / 1e6, 10 / 1e6  # USD per token, Claude Sonnet 5
BATCH = 40
N = 1000  # report per 1,000 sentences
RATES = (0.3, 0.4, 0.5, 1.0)
TRIVIAL_PROMPT_TOKENS = 8  # "Reply with the single word OK."


def total_input(d):
    it = d["usage"]["iterations"][0]
    return it["input_tokens"] + it["cache_creation_input_tokens"] + it["cache_read_input_tokens"]


def main():
    overhead = [total_input(json.load(open(f, encoding="utf-8"))) - TRIVIAL_PROMPT_TOKENS
                for f in sorted(glob.glob(os.path.join(DIR, "overhead_*.json")))]
    assert len(set(overhead)) == 1, overhead
    calls = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(DIR, "*.json")))
             if not os.path.basename(f).startswith("overhead")]
    assert all(list(d["modelUsage"]) == ["claude-sonnet-5"] for d in calls)
    single = [d for d in calls if d.get("num_turns") == 1]
    tok_in = np.mean([total_input(d) - overhead[0] for d in single])
    tok_out = np.mean([d["usage"]["output_tokens"] for d in single])
    lat = np.mean([d["duration_api_ms"] / 1000 for d in single])
    usd_batch = tok_in * PRICE_IN + tok_out * PRICE_OUT

    # embedding + SVM on CPU, as in scripts/sanity_checks.py [3]
    test, calib, items = load_frontier("cb")
    svm = SVC(class_weight="balanced", random_state=0).fit(calib.X, calib.y)
    texts = [r["text"] for r in items]
    embed(texts[:8])  # warm-up
    t0 = time.perf_counter()
    svm.decision_function(embed(texts))
    svm_s = (time.perf_counter() - t0) / len(texts)

    per = {}
    for rho in RATES:
        batches = rho * N / BATCH
        cascade = rho < 1
        per[f"{rho}"] = dict(usd=batches * usd_batch, seconds=batches * lat + (N * svm_s if cascade else 0.0))
    per["svm_only"] = dict(usd=0.0, seconds=N * svm_s)
    out = dict(n_calls=len(calls), n_single_turn=len(single), cli_overhead_tokens=overhead[0],
               prompt_tokens_per_batch=float(tok_in), output_tokens_per_batch=float(tok_out),
               latency_s_per_batch=float(lat), usd_per_batch=float(usd_batch), svm_ms_per_sentence=1000 * svm_s,
               per_1000_sentences=per)
    with open(os.path.join(PAPER, "cost_time.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_1000_sentences"}, indent=1))
    print("per 1,000 sentences      USD      seconds")
    for k, v in per.items():
        print(f"  {k:>10s}  {v['usd']:8.4f}  {v['seconds']:8.1f}")


if __name__ == "__main__":
    main()
