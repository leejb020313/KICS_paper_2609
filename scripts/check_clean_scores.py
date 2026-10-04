"""Validate the clean Sonnet 5 re-scoring: every batch parsed, same sentence ids as the earlier outputs, scores in
[0, 1], model id claude-sonnet-5. Prints the CLI cost and mean thinking tokens.

    uv run --isolated --no-project python scripts/check_clean_scores.py
"""
import glob
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "llm_scores")


def parse(path):
    return json.loads(re.search(r"\{.*\}", open(path, encoding="utf-8", errors="ignore").read(), re.S).group(0))


bad, models, cost, think = [], {}, 0.0, []
for t in sorted(glob.glob(os.path.join(ROOT, "sonnet5", "*.txt"))):
    n, b = t[:-4], os.path.basename(t[:-4])
    old = parse(os.path.join(ROOT, "sonnet5_cli_default", b + ".out"))
    try:
        d = json.load(open(n + ".json", encoding="utf-8"))
        cost += d.get("total_cost_usd", 0)
        think.append(d["usage"]["output_tokens_details"].get("thinking_tokens", 0))
        for k in d.get("modelUsage") or {}:
            models[k] = models.get(k, 0) + 1
        new = parse(n + ".out")
    except Exception as e:  # noqa: BLE001
        bad.append((b, repr(e)[:60]))
        continue
    if set(new) != set(old):
        bad.append((b, f"ids {len(new)} vs {len(old)}"))
    elif not all(0 <= float(v) <= 1 for v in new.values()):
        bad.append((b, "score out of range"))
print("models", models, f"CLI cost ${cost:.2f}", "mean thinking tokens", round(sum(think) / max(1, len(think))), "batches", len(think))
print("issues", bad if bad else "none")
