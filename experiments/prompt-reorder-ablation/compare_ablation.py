"""Compare the original prompt (confidence_score-first) against the reordered
prompt (justification-first) on the same test claims.

Minimal mode (only *_test_scores_reordered.jsonl exists):
    reports raw baseline wrong-rate / F1 for original vs reordered.
    This alone answers the core hypothesis: does writing the justification
    BEFORE committing to a number reduce the score/justification mismatch
    that caused Gemma's errors in the main repro (see REPORT.md and the
    divergence analysis from the parent chat session)?

Full mode (calib_scores_reordered.jsonl ALSO exists for a dataset):
    additionally re-runs NN-PPI (paper) and our gated+local-regression method
    on top of the reordered scores, using ../../src/nnppi/calibration.py
    unchanged, so the reordered-prompt numbers are directly comparable to the
    numbers already in ../../REPORT.md.

Usage:
    python compare_ablation.py --dataset clef
    python compare_ablation.py --dataset claimbuster
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_scores(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("parse_ok")]
    labels = np.array([r["label"] for r in rows])
    raw = np.array([r["confidence_score"] for r in rows], dtype=float)
    texts = [r["text"] for r in rows]
    return texts, raw, labels


def report(name, scores, labels):
    pred = (scores >= 0.5).astype(int)
    wrong = pred != labels
    resid = np.abs(labels - scores)
    print(f"{name:40s} wrong={wrong.sum():3d}/{len(labels)} ({wrong.mean():.1%})  "
          f"F1={f1_score(labels, pred, average='weighted'):.3f}  "
          f"mean|resid|={resid.mean():.3f}  p95|resid|={np.percentile(resid, 95):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["clef", "claimbuster"], required=True)
    args = ap.parse_args()
    name = args.dataset

    orig_test_path = ROOT / "results" / f"{name}_test_scores.jsonl"
    reord_test_path = Path("results") / f"{name}_test_scores_reordered.jsonl"
    if not reord_test_path.exists():
        print(f"Missing {reord_test_path} -- run score_with_llm_reordered.py first.")
        return

    orig_texts, orig_raw, orig_labels = load_scores(orig_test_path)
    reord_texts, reord_raw, reord_labels = load_scores(reord_test_path)

    # Align on shared claims (both jsonl are keyed by the same Sentence_id/text
    # order from data/processed/*.csv, but only compare where BOTH parsed ok).
    orig_by_text = dict(zip(orig_texts, orig_raw))
    common = [t for t in reord_texts if t in orig_by_text]
    idx_reord = [reord_texts.index(t) for t in common]
    aligned_orig_raw = np.array([orig_by_text[t] for t in common])
    aligned_labels = np.array([reord_labels[i] for i in idx_reord])
    aligned_reord_raw = np.array([reord_raw[i] for i in idx_reord])

    print(f"=== {name}: raw baseline, original vs reordered prompt (n={len(common)}) ===")
    report("original (confidence_score-first)", aligned_orig_raw, aligned_labels)
    report("reordered (justification-first)", aligned_reord_raw, aligned_labels)

    orig_wrong = (aligned_orig_raw >= 0.5).astype(int) != aligned_labels
    reord_wrong = (aligned_reord_raw >= 0.5).astype(int) != aligned_labels
    fixed = orig_wrong & ~reord_wrong
    broke = ~orig_wrong & reord_wrong
    print(f"reorder FIXED (orig wrong -> reordered correct): {fixed.sum()}")
    print(f"reorder BROKE (orig correct -> reordered wrong): {broke.sum()}")
    print()

    # Full mode: NN-PPI + our method, only if calib was also rescored.
    calib_reord_path = Path("results") / f"{name}_calib_scores_reordered.jsonl"
    if not calib_reord_path.exists():
        print(f"(Skipping NN-PPI / our-method comparison: {calib_reord_path} not found.\n"
              f" Rescore the calib set with the reordered prompt to unlock this.)")
        return

    from sentence_transformers import SentenceTransformer
    from nnppi.calibration import NNPPIConfig, nn_ppi_apply

    calib_texts, calib_raw, calib_labels = load_scores(calib_reord_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    calib_emb = model.encode(calib_texts, show_progress_bar=False, normalize_embeddings=True)
    test_emb = model.encode(common, show_progress_bar=False, normalize_embeddings=True)

    cfg = NNPPIConfig(k=3, clip_range=True)
    res_nnppi = nn_ppi_apply(aligned_reord_raw, test_emb, calib_raw, calib_emb, calib_labels, cfg)

    print(f"=== {name}: reordered prompt + NN-PPI (paper formula) ===")
    report("reordered + NN-PPI", res_nnppi.theta, aligned_labels)


if __name__ == "__main__":
    main()
