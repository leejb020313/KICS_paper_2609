"""Score claims with Gemma 3 4B through a local llama.cpp server (the NN-PPI small-LM setting).

Writes one JSON line per claim and resumes from an existing output file, so it can be
interrupted. Sampling follows NN-PPI Appendix C (T=1.0, top-k=64, top-p=0.95).

    llama-server -m google_gemma-3-4b-it-Q4_K_M.gguf --port 8080 -c 8192 -ngl 99
    python scripts/score_gemma.py --dataset data/processed/clef_test.csv --out results/gemma/clef_test_scores.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.prompt import build_prompt  # noqa: E402


def call_llm(base_url, prompt, max_tokens, temperature, top_k, top_p):
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "temperature": temperature, "top_k": top_k, "top_p": top_p}
    r = requests.post(base_url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_response(raw):
    """Parse the last JSON object that contains a confidence_score."""
    matches = list(re.finditer(r"\{[^{}]*\"confidence_score\"[^{}]*\}", raw, re.DOTALL))
    if matches:
        try:
            obj = json.loads(matches[-1].group(0))
            return {"confidence_score": float(obj.get("confidence_score")), "justification": obj.get("justification"),
                    "parse_ok": True}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"confidence_score": None, "justification": None, "parse_ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with columns Sentence_id, text, label")
    ap.add_argument("--out", required=True, help="output JSONL (resumable)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["Sentence_id"] for line in f if line.strip()}
    todo = df[~df["Sentence_id"].isin(done)]
    print(f"{len(todo)} of {len(df)} claims to score ({len(done)} already done).")

    with open(out, "a", encoding="utf-8") as f:
        for n, row in enumerate(todo.itertuples(), 1):
            result = {"confidence_score": None, "justification": None, "parse_ok": False, "raw_response": None}
            for attempt in range(args.retries):
                try:
                    raw = call_llm(args.base_url, build_prompt(row.text), args.max_tokens, args.temperature,
                                   args.top_k, args.top_p)
                    result = {**parse_response(raw), "raw_response": raw}
                    break
                except Exception as e:
                    print(f"  [{row.Sentence_id}] attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
            record = {"Sentence_id": int(row.Sentence_id), "text": row.text, "label": int(row.label), **result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {len(done) + n}/{len(df)}")


if __name__ == "__main__":
    main()
