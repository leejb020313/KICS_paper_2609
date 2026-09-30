"""Score statements with TypeSafe Jev (a decision-only "System One" model) through one Noul question each.

Jev returns P(yes) directly instead of generating text. Two question wordings:
  short   one plain check-worthiness question
  nnppi   the NN-PPI check-worthiness criteria (the same criteria the Sonnet prompt uses), passed as
          structured instructions
The wording is chosen on calibration data only (scripts/jev_pilot.py).

Ids are positions in data/frontier/{eval,calib,full}_set.json. Output: results/jev/<variant>/<set key>.jsonl,
one {"id", "noul", "input_tokens", "model"} line per statement; reruns skip ids already scored.
The API key is read from the TYPESAFE_API_KEY environment variable and never written anywhere.

    python scripts/score_jev.py --variant nnppi --sets clefcal,cbcal [--ids-file pilot_ids.json] [--workers 8]
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATA, RESULTS, read_json  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
CRITERIA = [
    "High-stakes, society-level, quantitative or study-based claims are very highly check-worthy.",
    "Broad policy mechanism or sector-wide quantitative claims are highly check-worthy.",
    "Mid-tier, localized or mixed claims with numbers/opinions are medium check-worthy.",
    "Isolated incidents, hearsay, or loosely phrased generalizations are low check-worthy.",
    "Personal stories, greetings, meta, nostalgia, logistics are very low check-worthy.",
    "Statements containing factual claims that are not check-worthy by above check-worthiness criteria or non "
    "claim statements (e.g. opinions, speculations, feelings, rhetorical statements, campaign slogans, "
    "predictions) are not check-worthy.",
]  # verbatim from cwcascade/prompt.py (NN-PPI), bullets split into a list
QUESTIONS = {
    "short": {"type": "noul",
              "instructions": "Does this statement contain a factual claim that is worth fact-checking?"},
    "nnppi": {"type": "noul",
              "instructions": {"check_worthiness_criteria": CRITERIA,
                               "question": "Acting as an impartial fact-checker, is this statement a "
                                           "check-worthy claim according to `check_worthiness_criteria`?"},
              "criteria": {"true": "A check-worthy factual claim",
                           "false": "Not check-worthy, or not a factual claim"}},
}


def ask(text, variant, key, retries=5):
    body = json.dumps({"state": text, "model": MODEL, "questions": {"cw": QUESTIONS[variant]}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            return dict(noul=float(out["answers"]["cw"]["noul"]), input_tokens=out["usage"]["input_tokens"],
                        model=out["model"])
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt if not (isinstance(e, urllib.error.HTTPError) and e.code == 429) else 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(QUESTIONS), required=True)
    ap.add_argument("--sets", required=True, help="comma-separated set keys, e.g. clefcal,cbcal,clef,cb")
    ap.add_argument("--ids-file", help="JSON {set key: [ids]} restricting which ids to score")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]

    sets = {k: v for fn in ("eval_set.json", "calib_set.json", "full_set.json")
            for k, v in read_json(os.path.join(DATA, "frontier", fn)).items()}
    only = read_json(args.ids_file) if args.ids_file else None
    out_dir = os.path.join(RESULTS, "jev", args.variant)
    os.makedirs(out_dir, exist_ok=True)
    for name in args.sets.split(","):
        path = os.path.join(out_dir, f"{name}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done = {json.loads(line)["id"] for line in f if line.strip()}
        ids = [i for i in (only[name] if only else range(len(sets[name]))) if i not in done]
        print(f"{name}: {len(ids)} to score ({len(done)} done)", flush=True)
        tokens, t0 = 0, time.time()
        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(ask, sets[name][i]["text"], args.variant, key): i for i in ids}
            for n, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                tokens += res["input_tokens"]
                f.write(json.dumps({"id": futs[fut], **res}) + "\n")
                f.flush()
                if n % 500 == 0:
                    print(f"  {n}/{len(ids)}", flush=True)
        print(f"  done in {time.time() - t0:.0f}s, {tokens:,} input tokens", flush=True)


if __name__ == "__main__":
    main()
