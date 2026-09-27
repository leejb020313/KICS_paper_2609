"""Accuracy vs. fraction of claims sent to the frontier LLM (fused vs. replacement cascade).

Reads results/final_results.json (written by final_eval.py) and writes figures/cascade_budget.{pdf,svg,png}.
"""
import json
import os

import numpy as np
from figstyle import COLUMN, PALETTE, figure_grid, panel_labels, save, use_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "final_results.json"), encoding="utf-8"))
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
NAMES = {"clef": "CLEF 2024", "cb": "ClaimBuster"}


def series(m, prefix):
    mu = np.array([m[f"{prefix}_{b}"]["acc"][0] for b in BUDGETS])
    sd = np.array([m[f"{prefix}_{b}"]["acc"][1] for b in BUDGETS])
    return mu, sd


def main():
    use_style()
    fig, axes = figure_grid(2, 1, width=COLUMN, ratio=1.02, sharex=True)
    axes = list(np.ravel(axes))
    x = 100 * np.array(BUDGETS)
    for ax, key in zip(axes, ["clef", "cb"]):
        m = R[key]["metrics"]
        for prefix, color, marker, label in [("fuse", "blue", "o", "Fused cascade (ours)"),
                                              ("replace", "orange", "s", "Replacement cascade")]:
            mu, sd = series(m, prefix)
            ax.plot(x, mu, marker=marker, markersize=3, color=PALETTE[color], linewidth=1.1, label=label, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=PALETTE[color], alpha=0.2, linewidth=0)
        son, son_sd = m["sonnet_thr"]["acc"]
        ax.axhline(son, color=PALETTE["black"], linestyle="--", linewidth=0.9, label="Frontier LLM, all claims")
        ax.axhspan(son - son_sd, son + son_sd, color=PALETTE["black"], alpha=0.08, linewidth=0)
        gem = m["gemma_nnppi_sel"]["acc"][0]  # k chosen on calib holdout, not test
        ax.axhline(gem, color=PALETTE["green"], linestyle=":", linewidth=1.0, label="Gemma 3 4B + NN-PPI")
        ax.text(0.98, {"clef": 0.22, "cb": 0.25}[key], f"{NAMES[key]} (n={R[key]['n_test']})", transform=ax.transAxes, ha="right", va="center")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(-2, 102)
        ax.grid(True, axis="y")
    axes[1].set_xlabel("Claims sent to frontier LLM (%)")
    panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.20, hspace=0.14)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.01))
    save(fig, os.path.join(ROOT, "figures", "cascade_budget"), formats=("pdf", "png"))


if __name__ == "__main__":
    main()
