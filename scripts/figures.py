"""Schedule disagreement by model, dataset and policy (with the irrelevant-peer
control), with question-level bootstrap 95% CIs read from results/summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("llama3-8b", "Llama-3-8B"), ("qwen2.5-7b", "Qwen2.5-7B")]
DATASETS = [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue"), ("2wiki", "2Wiki")]
SERIES = [("full", "Full sharing", "#2a78d6", ""),
          ("evidence_only", "Evidence-only", "#eb6834", "///"),
          ("irrelevant", "Irrelevant peer (control)", "#1baf7a", "...")]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def value(R, key):
    return R["irrelevant"]["disagreement"] if key == "irrelevant" else R["policies"][key]["disagreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="paper/fig_disagreement.pdf")
    args = ap.parse_args()
    S = json.loads((ROOT / args.summary).read_text())
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "hatch.color": "white", "hatch.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
    w = 0.26
    for ax, (mk, mn) in zip(axes, MODELS):
        for si, (key, label, color, hatch) in enumerate(SERIES):
            xs, ys, lo, hi = [], [], [], []
            for di, (dk, _) in enumerate(DATASETS):
                R = S.get(mk, {}).get(dk)
                if not R or (key == "irrelevant" and "irrelevant" not in R):
                    continue
                v = value(R, key)
                xs.append(di + (si - 1) * w)
                ys.append(v[0] * 100)
                lo.append((v[0] - v[1]) * 100)
                hi.append((v[2] - v[0]) * 100)
            ax.bar(xs, ys, w * 0.92, color=color, hatch=hatch, label=label,
                   edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6,
                        capsize=1.5, zorder=3)
        ax.set_xticks(range(len(DATASETS)), ["Hotpot", "MuSiQue", "2Wiki"], fontsize=6.5)
        ax.set_title(mn, fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Schedule disagreement (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
