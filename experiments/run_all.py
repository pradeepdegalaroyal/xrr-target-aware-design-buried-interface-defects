"""Reproduce every result and figure: ``python experiments/run_all.py`` (``--quick`` for a smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from xrdesign.amortised import fit_predict, simulate_scan
from xrdesign.design import METHODS, T_CRIT_NM, Library, run_experiment
from xrdesign.physics import Grid, sample_prior

DOSE = 3.0e4  # incident photons per measurement step
BG_NOMINAL = 1.0  # background counts per step assumed by the analyst
ROBUST_METHODS = ("uniform", "target_aware")
BG_LEVELS = (1.0, 10.0, 50.0, 200.0)


def design_main(cfg: dict, libs: list[Library]) -> pd.DataFrame:
    rows = []
    for seed, lib in enumerate(libs):
        rng = np.random.default_rng(1000 + seed)
        truths = sample_prior(rng, cfg["truths"])
        for k, th in enumerate(truths):
            for m in METHODS:
                for r in run_experiment(m, lib, th, cfg["steps"], DOSE, BG_NOMINAL, BG_NOMINAL, np.random.default_rng(10 * seed + k)):
                    rows.append(dict(seed=seed, truth=k, **r))
        print(f"  design seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def design_robust(cfg: dict, libs: list[Library]) -> pd.DataFrame:
    rows = []
    for seed, lib in enumerate(libs[: cfg["robust_seeds"]]):
        rng = np.random.default_rng(2000 + seed)
        truths = sample_prior(rng, cfg["robust_truths"])
        for bg in BG_LEVELS:
            for k, th in enumerate(truths):
                for m in ROBUST_METHODS:
                    r = run_experiment(m, lib, th, cfg["steps"], DOSE, bg, BG_NOMINAL, np.random.default_rng(10 * seed + k), record_every=cfg["steps"])
                    rows.append(dict(seed=seed, truth=k, bg_true=bg, **r[-1]))
        print(f"  robustness seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def amortised(cfg: dict) -> pd.DataFrame:
    rows = []
    dose = cfg["amortised_dose"]
    for seed in range(cfg["amortised_seeds"]):
        rng = np.random.default_rng(3000 + seed)
        x_all, y_all = simulate_scan(rng, max(cfg["train_sizes"]) + cfg["n_test"], dose, BG_NOMINAL)
        x_te, y_te = x_all[-cfg["n_test"] :], y_all[-cfg["n_test"] :]
        prior_rmse = float(np.sqrt(np.mean((y_te - y_all[: -cfg["n_test"]].mean()) ** 2)))
        for n in cfg["train_sizes"]:
            x_tr, y_tr = x_all[:n], y_all[:n]
            for model in ("ridge_full", "mlp_full", "mlp_small", "mlp_large", "vqc"):
                p = fit_predict(model, x_tr, y_tr, x_te, seed, n_layers=2)
                rows.append(dict(seed=seed, n_train=n, model=model, variant="main", shots=0, rmse=float(np.sqrt(np.mean((p - y_te) ** 2))), prior_rmse=prior_rmse))
            if n == cfg["train_sizes"][-1]:
                for shots in cfg["shots"]:
                    p = fit_predict("vqc", x_tr, y_tr, x_te, seed, shots=shots, n_layers=2)
                    rows.append(dict(seed=seed, n_train=n, model="vqc", variant="shots", shots=shots, rmse=float(np.sqrt(np.mean((p - y_te) ** 2))), prior_rmse=prior_rmse))
                p = fit_predict("vqc", x_tr, y_tr, x_te, seed, n_layers=1)
                rows.append(dict(seed=seed, n_train=n, model="vqc", variant="one_layer", shots=0, rmse=float(np.sqrt(np.mean((p - y_te) ** 2))), prior_rmse=prior_rmse))
        print(f"  amortised seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def summarise(main: pd.DataFrame, robust: pd.DataFrame, amo: pd.DataFrame, steps: int) -> dict:
    out: dict = {"threshold_nm": T_CRIT_NM, "dose_photons_per_step": DOSE, "steps": steps}
    fin = main[main.step == steps]
    per_seed = fin.assign(se=fin.err**2).groupby(["method", "seed"]).se.mean().pipe(np.sqrt).reset_index()
    out["final_rmse_nm"] = {
        m: {"mean": float(g.se.mean()), "std": float(g.se.std(ddof=1))} for m, g in per_seed.groupby("method")
    }
    out["final_width90_nm"] = {m: float(g.width90.mean()) for m, g in fin.groupby("method")}
    out["final_coverage90"] = {m: float(g.covered.mean()) for m, g in fin.groupby("method")}
    out["final_decision_accuracy"] = {m: float(g.decision_correct.mean()) for m, g in fin.groupby("method")}
    out["final_median_ess"] = {m: float(g.ess.median()) for m, g in fin.groupby("method")}
    u = per_seed[per_seed.method == "uniform"].set_index("seed").se
    out["paired_rmse_ratio_vs_uniform"] = {
        m: {"mean": float((g.set_index("seed").se / u).mean()), "std": float((g.set_index("seed").se / u).std(ddof=1)),
            "seeds_better": int(((g.set_index("seed").se / u) < 1).sum()), "n_seeds": int(len(u))}
        for m, g in per_seed.groupby("method")
    }
    if len(robust):
        rs = robust.assign(se=robust.err**2).groupby(["method", "bg_true", "seed"]).se.mean().pipe(np.sqrt).reset_index()
        out["robust_rmse_nm"] = {
            f"{m}|{bg}": {"mean": float(g.se.mean()), "std": float(g.se.std(ddof=1))} for (m, bg), g in rs.groupby(["method", "bg_true"])
        }
        out["robust_coverage90"] = {f"{m}|{bg}": float(g.covered.mean()) for (m, bg), g in robust.groupby(["method", "bg_true"])}
    if len(amo):
        mm = amo[amo.variant == "main"]
        out["amortised_rmse_nm"] = {
            f"{m}|{n}": {"mean": float(g.rmse.mean()), "std": float(g.rmse.std(ddof=1))} for (m, n), g in mm.groupby(["model", "n_train"])
        }
        out["amortised_prior_rmse_nm"] = float(amo.prior_rmse.mean())
        for v in ("shots", "one_layer"):
            g = amo[amo.variant == v]
            if len(g):
                out[f"amortised_vqc_{v}"] = {str(s): {"mean": float(h.rmse.mean()), "std": float(h.rmse.std(ddof=1))} for s, h in g.groupby("shots")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute)")
    args = ap.parse_args()
    if args.quick:
        cfg = dict(seeds=1, truths=2, steps=8, lib=4000, robust_seeds=1, robust_truths=1, amortised_seeds=1,
                   train_sizes=[32, 64], n_test=100, shots=[100], amortised_dose=1e6)
        out = ROOT / "results_quick"
    else:
        cfg = dict(seeds=5, truths=6, steps=48, lib=50000, robust_seeds=3, robust_truths=4, amortised_seeds=3,
                   train_sizes=[64, 256, 1024], n_test=400, shots=[1000, 100], amortised_dose=1e6)
        out = ROOT / "results"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    print("building particle libraries", flush=True)
    libs = [Library.build(np.random.default_rng(seed), cfg["lib"], Grid(n=80 if not args.quick else 40)) for seed in range(cfg["seeds"])]
    print("experiment 1: sequential design", flush=True)
    main_df = design_main(cfg, libs)
    print("experiment 2: unmodelled background sweep", flush=True)
    rob_df = design_robust(cfg, libs)
    print("experiment 3: amortised regressors", flush=True)
    amo_df = amortised(cfg)
    main_df.to_csv(out / "design_main.csv", index=False)
    rob_df.to_csv(out / "design_robust.csv", index=False)
    amo_df.to_csv(out / "amortised.csv", index=False)
    summ = summarise(main_df, rob_df, amo_df, cfg["steps"])
    summ["config"] = cfg
    summ["runtime_s"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    from make_figures import make_all  # noqa: E402

    make_all(out, ROOT / ("figures" if not args.quick else "figures_quick"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
