"""그림 1: 프런티어 LLM 호출률에 따른 테스트 정확도 (결합형 vs 교체형 캐스케이드).

Reads results/final_results.json (written by final_eval.py) and writes figures/cascade_budget.{pdf,png}
at the printed width of one KICS column (8.2 cm), so it is inserted without rescaling.

With --full, reads results/final_results_full.json (final_eval.py --full) and writes
figures/cascade_budget_full.{pdf,png}: the same curves on the full held-out sets, with both
full-call baselines (calib-tuned threshold, and + NN-PPI) as reference lines instead of Gemma.
"""
import json
import os
import sys

import matplotlib as mpl
import numpy as np
from figstyle import PALETTE, figure_grid, panel_labels, save, use_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = "--full" in sys.argv
R = json.load(open(os.path.join(ROOT, "results", "final_results_full.json" if FULL else "final_results.json"),
                   encoding="utf-8"))
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
NAMES = {"clef": f"CLEF 2024 (n={R['clef']['n_test']})", "cb": f"ClaimBuster (n={R['cb']['n_test']})"}
if FULL:  # four-digit n does not fit the column; the caption states the test sizes
    NAMES = {"clef": "CLEF 2024", "cb": "ClaimBuster"}
WIDTH_IN = 8.2 / 2.54  # one column of the KICS two-column layout


def series(m, prefix):
    mu = np.array([m[f"{prefix}_{b}"]["acc"][0] for b in BUDGETS])
    sd = np.array([m[f"{prefix}_{b}"]["acc"][1] for b in BUDGETS])
    return mu, sd


# figure text: English in the full-set figure (matches the English method diagram), Korean in the original one
T = dict(fuse="Fused cascade (ours)", replace="Replacement cascade", thr="All-call LLM (stronger setting)",
         nnppi="All-call LLM + NN-PPI", x="LLM call rate (%)", y="Accuracy", font="Arial") if FULL else     dict(fuse="결합형 캐스케이드 (제안)", replace="교체형 캐스케이드", thr="전량 호출 (임계값 조정)",
         x="LLM 호출률 (%)", y="정확도", font="Malgun Gothic")


def main():
    use_style()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [T["font"]],
                         "axes.unicode_minus": False, "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})
    fig, axes = figure_grid(1, 2, width=WIDTH_IN, ratio=0.58)
    x = 100 * np.array(BUDGETS)
    for ax, key in zip(axes, ["clef", "cb"]):
        m = R[key]["metrics"]
        for prefix, color, marker, label in [("fuse", "blue", "o", T["fuse"]),
                                              ("replace", "orange", "s", T["replace"])]:
            mu, sd = series(m, prefix)
            ax.fill_between(x, mu - sd, mu + sd, color=PALETTE[color], alpha=0.15, linewidth=0)
            ax.plot(x, mu, marker=marker, markersize=2.6, color=PALETTE[color], linewidth=1.2, label=label, zorder=3)
        # the stronger of the two all-call settings on this dataset (tuned threshold on CLEF, raw 0.5 cut-off on
        # ClaimBuster), i.e. the baseline the text claims parity with; the weaker one would flatter the cascade
        best = max(("sonnet_raw", "sonnet_thr"), key=lambda n: m[n]["acc"][0]) if FULL else "sonnet_thr"
        ax.axhline(m[best]["acc"][0], color=PALETTE["black"], linestyle="--", linewidth=1.0, label=T["thr"])
        if FULL:
            ax.axhline(m["sonnet_nnppi"]["acc"][0], color=PALETTE["green"], linestyle="-.", linewidth=1.1,
                       label=T["nnppi"])
        else:
            ax.axhline(m["gemma_nnppi_sel"]["acc"][0], color=PALETTE["green"], linestyle=":", linewidth=1.2,
                       label="Gemma 3 4B + NN-PPI")
        ax.axvline(50, color=PALETTE["black"], linewidth=0.6, alpha=0.25, zorder=0)
        # the operating point chosen on the calibration set
        ax.plot([50], [m["fuse_0.5"]["acc"][0]], marker="o", markersize=6.5, markerfacecolor="none",
                markeredgecolor=PALETTE["blue"], markeredgewidth=1.1, zorder=4)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(T["x"])
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel(T["y"])
    if FULL:
        axes[0].set_ylim(0.85, 0.93)
        axes[0].set_yticks([0.86, 0.88, 0.90, 0.92])
        axes[1].set_ylim(0.78, 0.86)
        axes[1].set_yticks([0.78, 0.80, 0.82, 0.84, 0.86])
    else:
        axes[0].set_ylim(0.825, 0.91)
        axes[0].set_yticks([0.84, 0.86, 0.88, 0.90])
        axes[1].set_ylim(0.79, 0.865)
        axes[1].set_yticks([0.80, 0.82, 0.84, 0.86])
    panel_labels(axes, labels=[f"(a) {NAMES['clef']}", f"(b) {NAMES['cb']}"], weight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.155, right=0.955, top=0.90, bottom=0.40, wspace=0.40)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    save(fig, os.path.join(ROOT, "figures", "cascade_budget_full" if FULL else "cascade_budget"),
         formats=("pdf", "png"))


if __name__ == "__main__":
    main()
