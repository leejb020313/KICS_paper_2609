"""Compare usage of the same test batches in the Oct-01 cost check (checks/sonnet5_cost) and the Oct-04 clean
re-score (sonnet5/): input tokens (incl. the CLI's own system prompt), output and thinking tokens, latency.

    uv run --isolated --no-project --with numpy python scripts/cost_overhead_check.py
"""
import glob
import json
import os

import numpy as np

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "llm_scores")


def tin(d):
    it = d["usage"]["iterations"][0]
    return it["input_tokens"] + it["cache_creation_input_tokens"] + it["cache_read_input_tokens"]


rows = []
for f in sorted(glob.glob(os.path.join(S, "checks", "sonnet5_cost", "*.json"))):
    b = os.path.basename(f)
    if b.startswith("overhead"):
        continue
    a, n = json.load(open(f, encoding="utf-8")), json.load(open(os.path.join(S, "sonnet5", b), encoding="utf-8"))
    rows.append((tin(n) - tin(a), a["usage"]["output_tokens"], n["usage"]["output_tokens"],
                 a["usage"]["output_tokens_details"].get("thinking_tokens", 0), n["usage"]["output_tokens_details"].get("thinking_tokens", 0),
                 a["duration_api_ms"] / 1000, n["duration_api_ms"] / 1000, a.get("num_turns"), n.get("num_turns")))
r = np.array(rows, dtype=float)
print("input-token difference new-old: min %d max %d" % (r[:, 0].min(), r[:, 0].max()))
print("output tokens old %.0f new %.0f | thinking old %.0f new %.0f | latency old %.1fs new %.1fs | turns old %s new %s"
      % (r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), r[:, 4].mean(), r[:, 5].mean(), r[:, 6].mean(),
         sorted(set(r[:, 7])), sorted(set(r[:, 8]))))
