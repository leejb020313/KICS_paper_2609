"""Run this FIRST. Verifies llama-server actually returns top_logprobs before
you spend hours on a full scoring pass.

Scores 10 CLEF claims in all three modes and prints the raw token distributions
so you can see with your own eyes whether the channel is de-quantized.

    python smoke_test.py
    python smoke_test.py --base-url http://127.0.0.1:8080/v1/chat/completions
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_with_logprobs import build_prompt, call_llm, SCORERS, token_distribution, DEFAULT_BASE_URL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--dataset", default="../../data/processed/clef_test.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset).head(args.n)

    try:
        probe = call_llm(args.base_url, "Answer with one word: Yes or No. Is 2+2=4?\nAnswer:",
                         args.top_logprobs, 4, 0.0)
    except Exception as e:
        print(f"FAIL: cannot reach the server at {args.base_url}\n  {e}")
        return
    dist = token_distribution(probe, 0)
    if not dist:
        print("FAIL: server responded but returned NO logprobs.")
        print("      Your llama-server build/flags do not expose top_logprobs on")
        print("      /v1/chat/completions. Try a newer llama.cpp build, or switch to")
        print("      the native /completion endpoint with n_probs.")
        print(f"      raw choice keys: {list(probe.keys())}")
        return
    print(f"OK: logprobs are live. Probe returned {len(dist)} alternatives for the first token.")
    print(f"    top 5: {[(t, round(p, 4)) for t, p in dist[:5]]}\n")

    for mode in ["yesno", "digit", "verbal"]:
        print("=" * 74)
        print(f"MODE: {mode}")
        print("=" * 74)
        for _, row in df.iterrows():
            choice = call_llm(args.base_url, build_prompt(mode, row["text"]),
                              args.top_logprobs, 4, 0.0)
            score, detail = SCORERS[mode](choice)
            sampled = (choice.get("message", {}) or {}).get("content", "")
            text = row["text"][:64] + ("..." if len(row["text"]) > 64 else "")
            score_str = f"{score:.4f}" if score is not None else "  FAIL"
            print(f"  gold={row['label']}  score={score_str}  sampled={sampled!r}")
            print(f"    {text}")
            if detail:
                print(f"    {json.dumps(detail, ensure_ascii=False)}")
        print()

    print("What to look for:")
    print("  - scores should be CONTINUOUS (e.g. 0.6173, not 0.60/0.70)")
    print("  - mass_on_poles / mass_on_digits should be close to 1.0; if it is tiny,")
    print("    the model is answering in some other format and the prompt needs a tweak")
    print("  - if every score is identical, something is wrong -- stop and report back")


if __name__ == "__main__":
    main()
