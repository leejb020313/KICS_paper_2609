"""Input-token accounting of the clean Sonnet run (diagnostic for cost_time.py).

    uv run --isolated --no-project python scripts/usage_dump.py
"""
import glob
import json
import os

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "llm_scores", "sonnet5")
rows = []
for f in sorted(glob.glob(os.path.join(S, "*.json"))):
    b = os.path.basename(f)
    if not b.startswith(("clef_", "clefdev_", "clefoff_", "cb_", "cbrest_")):
        continue
    d = json.load(open(f, encoding="utf-8"))
    u, it = d["usage"], d["usage"]["iterations"]
    rows.append((b, len(it), it[0]["input_tokens"], it[0]["cache_creation_input_tokens"], it[0]["cache_read_input_tokens"],
                 u["input_tokens"], u["cache_creation_input_tokens"], u["cache_read_input_tokens"], d.get("total_cost_usd")))
tot = sorted(r[2] + r[3] + r[4] for r in rows)
print("iterations[0] total input: min", tot[0], "median", tot[len(tot) // 2], "max", tot[-1])
for r in rows[:4] + rows[-4:]:
    print(r)
print("n iterations:", sorted({r[1] for r in rows}))
