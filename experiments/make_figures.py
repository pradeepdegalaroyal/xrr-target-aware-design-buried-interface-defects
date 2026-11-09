"""Figures for the study (earth-tone palette, 200 dpi)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"uniform": "#8c5a2b", "random": "#c9a227", "uncertainty": "#3b7f7a", "all_parameter": "#6b4f3a", "target_aware": "#b5532a"}
LAB = {"uniform": "uniform scan", "random": "random setting", "uncertainty": "predictive-variance sampling",
       "all_parameter": "all-parameter design", "target_aware": "target-aware design"}
MARK = {"uniform": "o", "random": "s", "uncertainty": "^", "all_parameter": "D", "target_aware": "v"}
LS = {"uniform": "-", "random": "--", "uncertainty": "-.", "all_parameter": ":", "target_aware": "-"}


def _style(ax):
    ax.grid(alpha=0.25, color="#8c7b6b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _per_seed_rmse(df, keys):
    return df.assign(se=df.err**2).groupby(keys + ["seed"]).se.mean().pipe(np.sqrt).reset_index()


def make_all(res: Path, fig: Path) -> None:
    fig.mkdir(exist_ok=True)
    main = pd.read_csv(res / "design_main.csv")
    rob = pd.read_csv(res / "design_robust.csv")
    amo = pd.read_csv(res / "amortised.csv")
    steps = int(main.step.max())
    dose = float(json.loads((res / "summary.json").read_text())["dose_photons_per_step"]) if (res / "summary.json").exists() else 3e4

    # 1: line + band, RMSE vs photons
    ps = _per_seed_rmse(main, ["method", "photons"])
    fig1, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, g in ps.groupby("method"):
        a = g.groupby("photons").se.agg(["mean", "std"]).reset_index()
        ax.plot(a.photons, a["mean"], LS[m], marker=MARK[m], ms=4, color=COL[m], label=LAB[m])
        ax.fill_between(a.photons, a["mean"] - a["std"], a["mean"] + a["std"], color=COL[m], alpha=0.12)
    ax.set_xlabel("cumulative incident photons")
    ax.set_ylabel("RMSE of defect thickness (nm)")
    ax.set_title("Posterior-mean error versus photon budget (mean ± std over seeds)")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig1.tight_layout()
    fig1.savefig(fig / "rmse_vs_photons.png", dpi=200)
    plt.close(fig1)

    # 2: violin + seeds
    fin = main[main.step == steps].copy()
    fin["abs_err"] = fin.err.abs()
    order = list(COL)
    fig2, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [fin[fin.method == m].abs_err.values for m in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, m in zip(parts["bodies"], order):
        pc.set_facecolor(COL[m])
        pc.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        g = fin[fin.method == m]
        sm = g.groupby("seed").abs_err.mean()
        ax.scatter(i + 1 + rng.normal(0, 0.05, len(g)), g.abs_err, s=8, color=COL[m], alpha=0.6)
        ax.scatter(np.full(len(sm), i + 1), sm.values, s=40, marker="D", facecolor="white", edgecolor="#2b2118", zorder=3, label="seed mean" if i == 0 else None)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([LAB[m].replace(" ", "\n", 1) for m in order], fontsize=8)
    ax.set_ylabel("absolute error of defect thickness (nm)")
    ax.set_title(f"Final error after {steps} measurements (dots: runs, diamonds: seed means)")
    _style(ax)
    fig2.tight_layout()
    fig2.savefig(fig / "final_error_violin.png", dpi=200)
    plt.close(fig2)

    # 3: ECDF
    fig3, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in order:
        v = np.sort(fin[fin.method == m].abs_err.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COL[m], ls=LS[m], lw=1.8, label=LAB[m])
    ax.set_xlabel("absolute error of defect thickness (nm)")
    ax.set_ylabel("empirical cumulative probability")
    ax.set_title("Empirical CDF of final absolute error")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    fig3.tight_layout()
    fig3.savefig(fig / "final_error_ecdf.png", dpi=200)
    plt.close(fig3)

    # 4: robustness line + band
    rs = _per_seed_rmse(rob, ["method", "bg_true"])
    fig4, ax = plt.subplots(figsize=(6.0, 4.0))
    for m, g in rs.groupby("method"):
        a = g.groupby("bg_true").se.agg(["mean", "std"]).reset_index()
        ax.plot(a.bg_true, a["mean"], LS[m], marker=MARK[m], color=COL[m], label=LAB[m])
        ax.fill_between(a.bg_true, a["mean"] - a["std"], a["mean"] + a["std"], color=COL[m], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("true background (counts per step; analyst assumes 1)")
    ax.set_ylabel("RMSE of defect thickness (nm)")
    ax.set_title("Robustness to unmodelled detector background")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig4.tight_layout()
    fig4.savefig(fig / "background_robustness.png", dpi=200)
    plt.close(fig4)

    # 5: amortised learning curves
    mm = amo[amo.variant == "main"]
    fig5, ax = plt.subplots(figsize=(6.0, 4.0))
    cols = {"ridge_full": "#8c5a2b", "mlp_full": "#c9a227", "mlp_small": "#6b4f3a", "mlp_large": "#7a8f5a", "vqc": "#3b7f7a"}
    labs = {"ridge_full": "ridge, 40 features", "mlp_full": "MLP, 40 features", "mlp_small": "MLP 8-4, 4 components", "mlp_large": "MLP 32-32, 4 components", "vqc": "4-qubit re-uploading circuit"}
    for m, g in mm.groupby("model"):
        a = g.groupby("n_train").rmse.agg(["mean", "std"]).reset_index()
        ax.plot(a.n_train, a["mean"], marker="o", ms=4, color=cols[m], ls="-" if m == "vqc" else "--", label=labs[m])
        ax.fill_between(a.n_train, a["mean"] - a["std"], a["mean"] + a["std"], color=cols[m], alpha=0.12)
    ax.axhline(mm.prior_rmse.mean(), color="#2b2118", lw=1, ls=":", label="prior-mean predictor")
    ax.set_xscale("log")
    ax.set_xlabel("labelled training scans")
    ax.set_ylabel("test RMSE of defect thickness (nm)")
    ax.set_title("Amortised regressors, one uniform scan as input")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig5.tight_layout()
    fig5.savefig(fig / "amortised_learning_curves.png", dpi=200)
    plt.close(fig5)
