"""Apply the project's validated clip+trust-gate+local-ridge-regression
correction (theta_gated_localreg, ported verbatim from
experiments/prompt-reorder-ablation/full_pipeline_compare.py) to Safety-Flag
content-moderation items, and compare its ECE against:
  (a) raw flag_prob (no correction)
  (b) our own split-fit temperature scaling, reproducing the ONE baseline
      the Safety-Flag authors themselves tried (their reported ECE 2.8-6.0x
      reduction, verified separately against their own reproduce.sh)

All three are evaluated on the SAME held-out test split for a fair
apples-to-apples comparison (the authors' number is pooled across all 7
benchmarks; ours is per-benchmark, so it is not directly the same number --
this script produces the correct like-for-like baseline instead of reusing
theirs).
"""
import glob
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from nnppi.calibration import temperature_scale_fit, temperature_scale_apply  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    return np.asarray(EMB_MODEL.encode(list(texts), show_progress_bar=False, normalize_embeddings=True))


def theta_gated_localreg(raw_scores, emb, calib_raw, calib_emb, calib_labels,
                          k_reg=15, purity_thresh=0.7, sim_thresh=0.6, ridge=2.0):
    k_reg = min(k_reg, len(calib_raw) - 1)
    sims = emb @ calib_emb.T
    idx = np.argpartition(-sims, kth=k_reg - 1, axis=1)[:, :k_reg]
    theta = np.empty(len(raw_scores))
    for i in range(len(raw_scores)):
        j = idx[i]
        x = calib_raw[j]
        y = calib_labels[j].astype(float)
        nb_sims = sims[i, j]
        purity = max(y.mean(), 1 - y.mean())
        max_sim = nb_sims.max()
        p_trust = np.clip((purity - 0.5) / max(purity_thresh - 0.5, 1e-6), 0, 1)
        s_trust = np.clip((max_sim - 0.1) / max(sim_thresh - 0.1, 1e-6), 0, 1)
        lam = p_trust * s_trust
        if x.std() < 1e-6:
            a, b = float(np.mean(y - x)), 1.0
        else:
            X = np.vstack([x, np.ones_like(x)]).T
            XtX = X.T @ X + ridge * np.eye(2)
            coef = np.linalg.solve(XtX, X.T @ y)
            b, a = coef[0], coef[1]
        a_shrunk = lam * a
        b_shrunk = 1.0 + lam * (b - 1.0)
        theta[i] = np.clip(a_shrunk + b_shrunk * raw_scores[i], 0, 1)
    return theta


def ece_equalwidth(conf, correct, n_bins=10):
    """Same definition as the Safety-Flag authors' own rigor_analysis.py."""
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_one(path, model_name, dataset_name, n_seeds=10):
    rows = load(path)
    texts = [r["text"] for r in rows]
    gt = np.array([r["gt"] for r in rows])
    raw = np.array([r["flag_prob"] for r in rows])
    emb = embed(texts)

    n = len(rows)
    results = {"raw": [], "temp": [], "ours": []}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_pool = int(0.5 * n)
        n_tune = int(0.25 * n)
        pool_idx = perm[:n_pool]
        tune_idx = perm[n_pool:n_pool + n_tune]
        test_idx = perm[n_pool + n_tune:]

        raw_pool, raw_tune, raw_test = raw[pool_idx], raw[tune_idx], raw[test_idx]
        y_pool, y_tune, y_test = gt[pool_idx], gt[tune_idx], gt[test_idx]
        emb_pool, emb_tune, emb_test = emb[pool_idx], emb[tune_idx], emb[test_idx]

        # raw (as-is flag_prob used directly as confidence)
        conf_raw = raw_test
        correct_raw = (conf_raw >= 0.5).astype(int) == y_test
        results["raw"].append(ece_equalwidth(conf_raw, correct_raw))

        # temperature scaling, fit on pool, evaluated on test -- our own
        # reproduction of the ONE baseline the Safety-Flag authors tried
        T = temperature_scale_fit(raw_pool, y_pool)
        conf_temp = temperature_scale_apply(raw_test, T)
        correct_temp = (conf_temp >= 0.5).astype(int) == y_test
        results["temp"].append(ece_equalwidth(conf_temp, correct_temp))

        # ours: gate thresholds tuned on tune split (never test), pool = neighbor set
        best = None
        for pt in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for st in [0.2, 0.4, 0.6, 0.99]:
                th = theta_gated_localreg(raw_tune, emb_tune, raw_pool, emb_pool, y_pool,
                                           purity_thresh=pt, sim_thresh=st)
                wr = np.mean((th >= 0.5).astype(int) != y_tune)
                if best is None or wr < best[0]:
                    best = (wr, pt, st)
        _, best_pt, best_st = best
        conf_ours = theta_gated_localreg(raw_test, emb_test, raw_pool, emb_pool, y_pool,
                                          purity_thresh=best_pt, sim_thresh=best_st)
        correct_ours = (conf_ours >= 0.5).astype(int) == y_test
        results["ours"].append(ece_equalwidth(conf_ours, correct_ours))

    def summarize(key):
        arr = np.array(results[key])
        return arr.mean(), arr.std(ddof=1)

    raw_m, raw_s = summarize("raw")
    temp_m, temp_s = summarize("temp")
    ours_m, ours_s = summarize("ours")
    print(f"{model_name:15s} {dataset_name:12s} n={n:4d}  "
          f"ECE raw={raw_m:.3f}+-{raw_s:.3f}  "
          f"temp={temp_m:.3f}+-{temp_s:.3f} ({raw_m/max(temp_m,1e-9):.2f}x)  "
          f"ours={ours_m:.3f}+-{ours_s:.3f} ({raw_m/max(ours_m,1e-9):.2f}x)")
    return dict(model=model_name, dataset=dataset_name, n=n,
                ece_raw=raw_m, ece_raw_sd=raw_s,
                ece_temp=temp_m, ece_temp_sd=temp_s,
                ece_ours=ours_m, ece_ours_sd=ours_s,
                ece_raw_seeds=results["raw"], ece_temp_seeds=results["temp"],
                ece_ours_seeds=results["ours"])


if __name__ == "__main__":
    all_results = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl"))):
        fname = os.path.basename(path).replace(".jsonl", "")
        model_name, dataset_name = fname.split("__")
        all_results.append(run_one(path, model_name, dataset_name))

    out_path = os.path.join(os.path.dirname(__file__), "results", "ece_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")
