"""Join Safety-Flag's per-item model outputs (item-id + flag_prob + gt only,
no raw text redistributed -- see their README) with the actual item text from
the public source benchmarks, by matching on the item's native index in the
source dataset.

Verified match (not assumed): beavertails id=262 in Safety-Flag's jsonl has
the exact same 13-field category dict as PKU-Alignment/BeaverTails 30k_test
row 262. xstest id=N (0-indexed) matches Paul/XSTest row N exactly (checked
against 'type'/'focus' metadata for ids 0-2: kill/terminate/terminate).

Ethics commonsense does NOT line up this way (checked both official mirrors,
sequence diverges at index 4) -- skipped for this experiment, not worth the
time to reverse-engineer their subsample order.
"""
import json
import os

from datasets import load_dataset

SF_DATA = r"C:\Users\leejb\AppData\Local\Temp\safety-flag-benchmark\data\results"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["Llama-3.1-8B", "Mistral-7B", "Qwen2.5-7B", "OLMo-2-7B", "gemma-2-9b"]


def load_solo_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_beavertails():
    print("Loading PKU-Alignment/BeaverTails 30k_test ...")
    src = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    for model in MODELS:
        path = os.path.join(SF_DATA, f"{model}__beavertails.jsonl")
        if not os.path.exists(path):
            continue
        rows = load_solo_jsonl(path)
        out = []
        for r in rows:
            idx = int(r["id"])
            src_row = src[idx]
            out.append({
                "id": r["id"],
                "text": src_row["prompt"],
                "gt": 1 if r["gt"] == "(A)" else 0,
                "flag_prob": r["solo"]["flag_prob"],
            })
        out_path = os.path.join(OUT_DIR, f"{model}__beavertails.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"  {model}: {len(out)} items -> {out_path}")


def build_xstest():
    print("Loading Paul/XSTest ...")
    src = load_dataset("Paul/XSTest")["train"]
    for model in MODELS:
        path = os.path.join(SF_DATA, f"{model}__xstest.jsonl")
        if not os.path.exists(path):
            continue
        rows = load_solo_jsonl(path)
        out = []
        for r in rows:
            idx = int(r["id"])
            src_row = src[idx]
            out.append({
                "id": r["id"],
                "text": src_row["prompt"],
                "gt": 1 if r["gt"] == "(A)" else 0,
                "flag_prob": r["solo"]["flag_prob"],
            })
        out_path = os.path.join(OUT_DIR, f"{model}__xstest.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"  {model}: {len(out)} items -> {out_path}")


if __name__ == "__main__":
    build_beavertails()
    build_xstest()
