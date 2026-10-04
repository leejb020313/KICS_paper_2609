"""Weighted F1 (the measure NN-PPI reports) of each method, 100-run means, current result file.

    uv run --isolated --no-project python scripts/wf1_view.py
"""
import json

R = json.load(open("results/paper/final_results_full.json", encoding="utf-8"))
for d in ("clef", "cb"):
    M = R[d]["metrics"]
    print(d, " ".join(f"{k}={M[k]['wf1'][0]:.3f}" for k in
                      ("svm", "gemma_nnppi_sel", "sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5", "fuse_0.5", "fuse_1.0")))
