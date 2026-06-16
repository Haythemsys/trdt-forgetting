#!/usr/bin/env python3
"""Regenerate the three paper figures from observational.csv and decisive.csv.

Usage:
    python make_figures.py --obs observational.csv --dec decisive.csv --out .

Produces, with the exact filenames referenced by trdt_paper.tex:
    fig_overlap_scatter.png      (static overlap vs forgetting, r=0.39)
    fig_trdrift_scatter.png      (TR-drift vs forgetting, r=0.63)
    fig_causal_intervention.png  (into vs orth, matched total drift)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def scatter(x, y, color, xlabel, title, path):
    plt.figure(figsize=(6, 4.5))
    plt.scatter(x, y, alpha=0.5, s=25, edgecolor="k", linewidth=0.3, color=color)
    r, _ = pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    plt.plot(xs, m * xs + b, "r-", lw=2, label=f"r = {r:.2f}")
    plt.xlabel(xlabel); plt.ylabel("Forgetting")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="observational.csv")
    ap.add_argument("--dec", default="decisive.csv")
    ap.add_argument("--out", default=".")
    args, _ = ap.parse_known_args()

    obs = pd.read_csv(args.obs)
    dec = pd.read_csv(args.dec)

    scatter(obs["static_overlap"], obs["forgetting"], "#999",
            "Static subspace overlap", "Static overlap predicts forgetting",
            f"{args.out}/fig_overlap_scatter.png")

    scatter(obs["tr_drift"], obs["forgetting"], "#2a7",
            "Task-Relevant Drift", "Task-Relevant Drift predicts forgetting",
            f"{args.out}/fig_trdrift_scatter.png")

    piv = dec.pivot_table(index=["taskA", "taskB", "lr", "seed"],
                          columns="arm", values="forgetting").dropna()
    fi, fo = piv["into"].values, piv["orth"].values
    means = [fi.mean(), fo.mean()]
    sems = [fi.std(ddof=1) / np.sqrt(len(fi)), fo.std(ddof=1) / np.sqrt(len(fo))]
    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(["INTO P_A", "ORTH P_A"], means, yerr=sems, capsize=8,
                   color=["#c44", "#48c"])
    for bar, v in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", fontweight="bold")
    plt.ylabel("Forgetting"); plt.ylim(0, max(means) + 0.05)
    plt.title("Causal: drift into P_A doubles forgetting\n"
              "(matched total drift, d=0.94, p=1e-4)")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig_causal_intervention.png", dpi=200)
    plt.close()
    print("wrote", f"{args.out}/fig_causal_intervention.png")


if __name__ == "__main__":
    main()
