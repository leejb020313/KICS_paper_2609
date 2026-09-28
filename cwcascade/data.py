"""Loading of the evaluation data, the LLM scores and the sentence embeddings.

Every path is resolved from the repository root, so scripts can be run from anywhere.
"""
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# dataset key -> (frontier calib key, Gemma calib scores, Gemma test scores)
DATASETS = {
    "clef": ("clefcal", "clef_calib_scores", "clef_test_scores"),
    "cb": ("cbcal", "claimbuster_calib_scores", "claimbuster_test_scores"),
}
EMBEDDER = "all-MiniLM-L6-v2"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_frontier_scores(folder="batches"):
    """Parse the frontier LLM outputs in results/frontier/<folder>/*.out.

    Each file holds one JSON object {id: score}; the file-name prefix (clef, cb,
    clefcal, cbcal) says which set the ids index into.
    """
    scores = {}
    for path in glob.glob(os.path.join(RESULTS, "frontier", folder, "*.out")):
        key = os.path.basename(path).split("_")[0]
        with open(path, encoding="utf-8", errors="ignore") as f:
            obj = json.loads(re.search(r"\{.*\}", f.read(), re.S).group(0))
        scores.setdefault(key, {}).update({int(i): float(s) for i, s in obj.items()})
    return scores


_model = None


def embed(texts):
    """L2-normalised all-MiniLM-L6-v2 embeddings (the embedder NN-PPI uses)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDER)
    return np.asarray(_model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


@dataclass
class Split:
    X: np.ndarray       # embeddings
    y: np.ndarray       # labels (1 = check-worthy)
    s: np.ndarray       # LLM score in [0, 1]


# full held-out test sets: the paper's test set followed by the parts in data/frontier/full_set.json
FULL_PARTS = {"clef": ("clefdev", "clefoff"), "cb": ("cbrest",)}


def load_frontier(name, full=False):
    """Test and calibration splits with Claude Sonnet 5 scores for dataset `name`.

    With full=True the test split is the whole held-out set (FULL_PARTS appended); each test
    item then carries a "part" key naming the set it came from.
    """
    calib_key = DATASETS[name][0]
    test_items = [dict(r, part=name) for r in read_json(os.path.join(DATA, "frontier", "eval_set.json"))[name]]
    calib_items = read_json(os.path.join(DATA, "frontier", "calib_set.json"))[calib_key]
    fs = read_frontier_scores()
    test_s = [fs[name][i] for i in range(len(test_items))]
    if full:
        extra = read_json(os.path.join(DATA, "frontier", "full_set.json"))
        for part in FULL_PARTS[name]:
            test_items += [dict(r, part=part) for r in extra[part]]
            test_s += [fs[part][i] for i in range(len(extra[part]))]
    have = sorted(fs[calib_key])
    calib_items = [calib_items[i] for i in have]
    test = Split(embed([r["text"] for r in test_items]), np.array([r["label"] for r in test_items]),
                 np.array(test_s))
    calib = Split(embed([r["text"] for r in calib_items]), np.array([r["label"] for r in calib_items]),
                  np.array([fs[calib_key][i] for i in have]))
    return test, calib, test_items


def load_gemma(name, test_items):
    """Gemma 3 4B scores: the full parsed calibration set, and the same test items as the frontier set.

    Test sentences whose response could not be parsed get score 0. Returns test scores None
    when some test sentence has not been scored by Gemma at all.
    """
    _, calib_file, test_file = DATASETS[name]
    calib_rows = [r for r in read_jsonl(os.path.join(RESULTS, "gemma", calib_file + ".jsonl")) if r.get("parse_ok")]
    by_id = {}
    for f in (test_file, test_file + "_full"):  # *_full: Gemma scores of the extra full-set items
        if os.path.exists(path := os.path.join(RESULTS, "gemma", f + ".jsonl")):
            by_id.update({str(r["Sentence_id"]): r for r in read_jsonl(path)})
    if any(r["Sentence_id"] not in by_id for r in test_items):
        return None, None
    test_s = np.array([by_id[r["Sentence_id"]]["confidence_score"] if by_id[r["Sentence_id"]].get("parse_ok") else 0.0
                       for r in test_items], float)
    calib = Split(embed([r["text"] for r in calib_rows]), np.array([r["label"] for r in calib_rows]),
                  np.array([r["confidence_score"] for r in calib_rows], float))
    return test_s, calib


def best_threshold(s, y):
    """Score threshold with the highest accuracy on (s, y), searched on a 0.01 grid."""
    ts = np.linspace(0.02, 0.98, 97)
    return ts[np.argmax([np.mean((s >= t) == y) for t in ts])]
