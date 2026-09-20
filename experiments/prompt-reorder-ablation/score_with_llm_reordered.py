"""Run on the machine with the local llama.cpp server (same setup as
../../RUNBOOK_local_scoring.md), using the REORDERED prompt (prompt_reordered.py)
instead of the original ../../src/nnppi/prompt.py.

Identical to ../../src/nnppi/score_with_llm.py except:
  - imports build_prompt from prompt_reordered.py (justification-first schema)
  - parse_response looks for "justification" key first, "confidence_score" second
    (regex is key-order-agnostic, so this is really just a naming note)

Usage:
    python score_with_llm_reordered.py --dataset ../../data/processed/clef_test.csv --out results/clef_test_scores_reordered.jsonl
    python score_with_llm_reordered.py --dataset ../../data/processed/clef_calib.csv --out results/clef_calib_scores_reordered.jsonl
    (repeat for claimbuster_test.csv, claimbuster_calib.csv if you have time/budget)

Prereqs: same llama-server already running for the main repro:
    llama-server -m <your Gemma 3 4B gguf> --port 8080 -c 8192 -ngl 99
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from prompt_reordered import build_prompt

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"


def call_llm(base_url: str, prompt: str, max_tokens: int, enable_thinking: bool, temperature: float,
             top_k: int = None, top_p: float = None) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    r = requests.post(base_url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_response(raw: str) -> dict:
    """Strip <think>...</think> if present, then parse the trailing JSON object.
    Key order in the regex doesn't matter -- json.loads reads whichever key is present."""
    text = raw
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    matches = list(re.finditer(r"\{[^{}]*\"confidence_score\"[^{}]*\}", text, re.DOTALL))
    if not matches:
        return {"confidence_score": None, "justification": None, "parse_ok": False}
    try:
        obj = json.loads(matches[-1].group(0))
        score = float(obj.get("confidence_score"))
        return {"confidence_score": score, "justification": obj.get("justification"), "parse_ok": True}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"confidence_score": None, "justification": None, "parse_ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with columns: Sentence_id, text, label")
    ap.add_argument("--out", required=True, help="Output JSONL path (resumable)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--enable-thinking", dest="enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["Sentence_id"])
        print(f"Resuming: {len(done_ids)} already scored.")

    remaining = df[~df["Sentence_id"].isin(done_ids)]
    print(f"{len(remaining)} claims to score out of {len(df)}.")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in remaining.iterrows():
            prompt = build_prompt(row["text"])
            result = None
            for attempt in range(args.retries):
                try:
                    raw = call_llm(args.base_url, prompt, args.max_tokens, args.enable_thinking, args.temperature,
                                    top_k=args.top_k, top_p=args.top_p)
                    result = parse_response(raw)
                    result["raw_response"] = raw
                    break
                except Exception as e:
                    print(f"  [{row['Sentence_id']}] attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            if result is None:
                result = {"confidence_score": None, "justification": None, "parse_ok": False, "raw_response": None}

            record = {
                "Sentence_id": row["Sentence_id"],
                "text": row["text"],
                "label": int(row["label"]),
                **result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            done = len(done_ids) + (i - remaining.index[0] + 1)
            if done % 25 == 0:
                print(f"  {done}/{len(df)} done")

    print("Done.")


if __name__ == "__main__":
    main()
