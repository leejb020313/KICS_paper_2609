"""Recover parse_ok=False rows from raw_response without re-scoring.

The observed failure mode is the model emitting unescaped double quotes
inside the "justification" string, which breaks strict json.loads. We only
need confidence_score for calibration, so extract it directly with a regex
instead of requiring the whole object to be valid JSON.

Usage:
    python src/nnppi/reparse_failures.py results/clef_calib_scores.jsonl
Overwrites the file in place with recovered rows (adds "recovered": true).
"""
import json
import re
import sys
from pathlib import Path


def recover_confidence_score(raw_response: str):
    if not raw_response:
        return None
    text = raw_response
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r'"confidence_score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if not m:
        return None
    try:
        score = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 <= score <= 1.0):
        return None
    return score


def main():
    path = Path(sys.argv[1])
    rows = []
    n_recovered = 0
    n_still_failed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("parse_ok"):
                score = recover_confidence_score(rec.get("raw_response") or "")
                if score is not None:
                    rec["confidence_score"] = score
                    rec["parse_ok"] = True
                    rec["recovered"] = True
                    n_recovered += 1
                else:
                    n_still_failed += 1
            rows.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{path}: recovered {n_recovered}, still unrecoverable {n_still_failed}, total {len(rows)}")


if __name__ == "__main__":
    main()
