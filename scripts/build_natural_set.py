"""Natural-ratio learning sets, to replace the class-balanced ones (data/frontier/natural_set.json).

  clefnat  simple random sample of 2,405 sentences (the size of the old learning set) from CLEF 2024 train
           (22,501 sentences, 24.1% check-worthy), seed 42
  cbnat    every 2012-debate sentence of ClaimBuster (the pool the old balanced set was drawn from), de-duplicated
No sentence of either test set is included (CLEF test = dev + dev-test + official test; ClaimBuster test = 2016).

    python scripts/build_natural_set.py <raw dir with clef2024/ and ClaimBuster_Datasets.zip>
"""
import io
import json
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
raw = sys.argv[1]

train = pd.read_csv(os.path.join(raw, "clef2024", "train.tsv"), sep="\t")
train["label"] = (train["class_label"] == "Yes").astype(int)
clef = train.sample(n=2405, random_state=42)

z = zipfile.ZipFile(os.path.join(raw, "ClaimBuster_Datasets.zip"))
full = pd.concat([pd.read_csv(io.BytesIO(z.read(f"ClaimBuster_Datasets/datasets/{n}.csv"))) for n in ("groundtruth", "crowdsourced")],
                 ignore_index=True)
full["year"] = full["File_id"].astype(str).str.slice(0, 4)
full["label"] = (full["Verdict"] == 1).astype(int)
cb = full[full["year"] == "2012"].drop_duplicates("Sentence_id").sample(frac=1, random_state=42)

out = {k: [{"Sentence_id": str(r.Sentence_id), "text": r.Text, "label": int(r.label)} for r in df.itertuples()]
       for k, df in (("clefnat", clef), ("cbnat", cb))}
# guard: no overlap with any test sentence
tests = json.load(open(os.path.join(ROOT, "data", "frontier", "eval_set.json"), encoding="utf-8"))
tests.update(json.load(open(os.path.join(ROOT, "data", "frontier", "full_set.json"), encoding="utf-8")))
TEST_PARTS = {"clefnat": ("clef", "clefdev", "clefoff"), "cbnat": ("cb", "cbrest")}
for k, parts in TEST_PARTS.items():
    ids = {r["Sentence_id"] for p in parts for r in tests[p]}
    texts = {r["text"].strip().lower() for p in parts for r in tests[p]}
    n_id = sum(r["Sentence_id"] in ids for r in out[k])
    n_tx = sum(r["text"].strip().lower() in texts for r in out[k])
    print(f"{k}: id overlap with its own test set {n_id}, identical-text overlap {n_tx}")
    assert n_id == 0, f"{k}: learning sentence id found in its test set"
    out[k] = [r for r in out[k] if r["text"].strip().lower() not in texts]  # drop exact-duplicate texts (e.g. "Thank you.")
with open(os.path.join(ROOT, "data", "frontier", "natural_set.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
for k, v in out.items():
    print(k, len(v), "positive rate", round(sum(r["label"] for r in v) / len(v), 4))
