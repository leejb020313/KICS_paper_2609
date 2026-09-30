"""Figure 2 of the JEV paper: test accuracy against LLM call rate, JEV -> LLM (fused / replaced) and SVM -> LLM (fused).

Reads results/jev_strict_full.json (scripts/jev_strict.py) and writes figures/jev_budget_full.{pdf,png} at the printed
width of one KICS column (8.2 cm), same style as make_figure.py. Circles mark the call rate each fused cascade picks
on the learning set; the dashed line is the stronger all-call LLM setting of each dataset.

    cd scripts && uv run --locked --with matplotlib python make_jev_figure.py
"""
import json
import os

import matplotlib as mpl
import numpy as np
from figstyle import PALETTE, figure_grid, panel_labels, save, use_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "jev_strict_full.json"), encoding="utf-8"))
BUDGETS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
WIDTH_IN = 8.2 / 2.54
SERIES = [("jevfuse", "blue", "o", "JEV → LLM fused (ours)"), ("jevrep", "orange", "s", "JEV → LLM replacement"),
          ("fuse", "black", "^", "SVM → LLM fused")]


def main():
    use_style()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "axes.unicode_minus": False,
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, axes = figure_grid(1, 2, width=WIDTH_IN, ratio=0.58)
    x = 100 * np.array(BUDGETS)
    for ax, key in zip(axes, ["clef", "cb"]):
        r = R[key]
        m = r["metrics"]
        for v, color, marker, label in SERIES:
            mu = np.array([m[f"{v}_{b}"]["acc"][0] for b in BUDGETS])
            sd = np.array([m[f"{v}_{b}"]["acc"][1] for b in BUDGETS])
            ax.fill_between(x, mu - sd, mu + sd, color=PALETTE[color], alpha=0.13, linewidth=0)
            ax.plot(x, mu, marker=marker, markersize=2.4, color=PALETTE[color], linewidth=1.1, label=label, zorder=3)
        ax.axhline(r["mean_acc"][r["best_full"]], color=PALETTE["black"], linestyle="--", linewidth=0.9,
                   label="All-call LLM (stronger setting)")
        for v in ("jevfuse", "fuse"):  # call rate chosen on the learning set
            b = r["rate"][v]
            ax.plot([100 * b], [m[f"{v}_{b}"]["acc"][0]], marker="o", markersize=6.5, markerfacecolor="none",
                    markeredgecolor=PALETTE["blue" if v == "jevfuse" else "black"], markeredgewidth=1.0, zorder=4)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("LLM call rate (%)")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.85, 0.93)
    axes[0].set_yticks([0.86, 0.88, 0.90, 0.92])
    axes[1].set_ylim(0.78, 0.87)
    axes[1].set_yticks([0.78, 0.80, 0.82, 0.84, 0.86])
    panel_labels(axes, labels=["(a) CLEF 2024", "(b) ClaimBuster"], weight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.155, right=0.955, top=0.90, bottom=0.40, wspace=0.40)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.8,
               columnspacing=0.8, handlelength=1.8, handletextpad=0.4)
    save(fig, os.path.join(ROOT, "figures", "jev_budget_full"), formats=("pdf", "png"))


if __name__ == "__main__":
    main()
