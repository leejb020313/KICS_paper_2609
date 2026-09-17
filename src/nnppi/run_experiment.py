"""Driver: once results/*_scores.jsonl exist, run every condition and print a
summary table. Nothing here needs GPU -- embeddings + calibration are CPU-only.

Usage:
    python src/nnppi/run_experiment.py --dataset clef
    python src/nnppi/run_experiment.py --dataset claimbuster
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from nnppi.calibration import (
    NNPPIConfig, nn_ppi_apply, conformal_scale_fit,
    temperature_scale_fit, temperature_scale_apply,
    platt_scale_fit, platt_scale_apply,
    isotonic_fit, isotonic_apply,
)
from nnppi.evaluate import report_f1, scores_to_preds, mcnemar_test, ci_coverage, out_of_range_rate

ROOT = Path(__file__).parent.parent.parent


def load_scores(path: Path):
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


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["clef", "claimbuster"], required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    results_dir = ROOT / args.results_dir
    calib_texts, calib_raw, calib_labels = load_scores(results_dir / f"{args.dataset}_calib_scores.jsonl")
    test_texts, test_raw, test_labels = load_scores(results_dir / f"{args.dataset}_test_scores.jsonl")
    print(f"{args.dataset}: calib n={len(calib_labels)} (parse_ok), test n={len(test_labels)} (parse_ok)")

    calib_emb = embed(calib_texts)
    test_emb = embed(test_texts)

    rows = []

    # Baseline: raw uncalibrated
    m = report_f1(test_labels, test_raw)
    rows.append({"condition": "raw_uncalibrated", "k": "-", **m})

    # Standard calibration baselines
    T = temperature_scale_fit(calib_raw, calib_labels)
    ts_scores = temperature_scale_apply(test_raw, T)
    rows.append({"condition": "temperature_scaling", "k": "-", **report_f1(test_labels, ts_scores)})

    platt = platt_scale_fit(calib_raw, calib_labels)
    platt_scores = platt_scale_apply(test_raw, platt)
    rows.append({"condition": "platt_scaling", "k": "-", **report_f1(test_labels, platt_scores)})

    iso = isotonic_fit(calib_raw, calib_labels)
    iso_scores = isotonic_apply(test_raw, iso)
    rows.append({"condition": "isotonic_regression", "k": "-", **report_f1(test_labels, iso_scores)})

    # NN-PPI variants x k sweep
    configs = {
        "nnppi_baseline": NNPPIConfig(),
        "nnppi_+simweight": NNPPIConfig(similarity_weighted=True),
        "nnppi_+clip": NNPPIConfig(clip_range=True),
        "nnppi_+weightedvar": NNPPIConfig(weighted_variance=True),
        "nnppi_full(A+B+C)": NNPPIConfig(similarity_weighted=True, clip_range=True, weighted_variance=True),
    }
    baseline_preds = None
    for name, base_cfg in configs.items():
        for k in (3, 5, 10):
            cfg = NNPPIConfig(k=k, similarity_weighted=base_cfg.similarity_weighted,
                               clip_range=base_cfg.clip_range, weighted_variance=base_cfg.weighted_variance)
            res = nn_ppi_apply(test_raw, test_emb, calib_raw, calib_emb, calib_labels, cfg)
            m = report_f1(test_labels, res.theta)
            row = {
                "condition": name, "k": k, **m,
                "ci_coverage": round(ci_coverage(test_labels, res.theta, res.ci_half_width), 3),
                "mean_ci_width": round(float(np.mean(res.ci_half_width) * 2), 3),
                "out_of_range_rate": round(res.out_of_range_rate, 3),
            }
            rows.append(row)
            if name == "nnppi_baseline" and k == 5:
                baseline_preds = scores_to_preds(res.theta)
            if name == "nnppi_full(A+B+C)" and k == 5 and baseline_preds is not None:
                full_preds = scores_to_preds(res.theta)
                mc = mcnemar_test(test_labels, baseline_preds, full_preds)
                print(f"McNemar baseline vs full (k=5): {mc}")

    # Split-conformal calibration of the CI width, fit ONLY on a held-out
    # slice of the calibration set (never the test set) -- see if a properly
    # calibrated multiplier closes more of the coverage gap than z=1.96.
    rng = np.random.default_rng(0)
    n_calib = len(calib_labels)
    perm = rng.permutation(n_calib)
    n_holdout = max(1, int(0.2 * n_calib))
    holdout_idx, neighbor_idx = perm[:n_holdout], perm[n_holdout:]
    full_cfg_k5 = NNPPIConfig(k=5, similarity_weighted=True, clip_range=True, weighted_variance=True)
    q = conformal_scale_fit(
        calib_raw[holdout_idx], calib_emb[holdout_idx], calib_labels[holdout_idx],
        calib_raw[neighbor_idx], calib_emb[neighbor_idx], calib_labels[neighbor_idx],
        full_cfg_k5, target_coverage=0.95,
    )
    res = nn_ppi_apply(test_raw, test_emb, calib_raw, calib_emb, calib_labels, full_cfg_k5)
    conformal_half_width = q * res.sqrt_var
    rows.append({
        "condition": "nnppi_full+conformal", "k": 5, **report_f1(test_labels, res.theta),
        "ci_coverage": round(ci_coverage(test_labels, res.theta, conformal_half_width), 3),
        "mean_ci_width": round(float(np.mean(conformal_half_width) * 2), 3),
        "out_of_range_rate": round(res.out_of_range_rate, 3),
    })
    print(f"conformal scale q (target 95%, fit on calib holdout only) = {q:.3f}  (parametric z=1.96)")

    print(f"\n=== {args.dataset} summary ===")
    header = ["condition", "k", "weighted_f1", "class0_f1", "class1_f1",
              "ci_coverage", "mean_ci_width", "out_of_range_rate"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(round(r.get(c, ""), 3) if isinstance(r.get(c), float) else r.get(c, "")) for c in header))


if __name__ == "__main__":
    main()
