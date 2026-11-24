"""SQ1, evidence horizon: how much more REX evidence would move learning from the fleet
to the component?

SQ1 finds that the fleet bounds the population-level bias `c` (90% interval within a
factor of 2.4) while individual component posteriors remain mostly prior (median evidence
weight w = 0.098). This script asks how much additional in-orbit exposure would be needed
before component-level posteriors became data-driven, and whether more exposure buys that
or only sharpens `c`.

METHOD — forward simulation from the fitted model, refitting the SAME model.

  1. Fit the baseline on the real fleet (unchanged calibration of SQ1).
  2. Per replicate, draw ONE grid node (alpha_0*, c*) from the fitted posterior.
     That draw is the replicate's GROUND TRUTH, which makes coverage testable.
  3. Draw each component's rate from its posterior CONDITIONAL on that truth,
     lambda_i ~ Gamma(alpha_0* + k_i, beta_i* + T_i).
  4. Add exposure. Generate the extra failures it would produce,
     k_extra_i ~ Poisson(lambda_i * dT_i), and append them to the REAL counts.
  5. Refit the unchanged model on (k + k_extra, T_new) and record the metrics.

At scale = 1 there is no extra exposure, so every replicate reproduces the real
fit exactly and coverage must come out at the nominal 90%. That is the built-in
calibration check.

TWO EXPOSURE PATTERNS

  spread        every component's exposure scaled by s -- "fly the whole fleet
                s times longer"
  concentrated  the same total added exposure, but delivered to only a fraction
                of component types (those with the highest expected failure
                count, i.e. deliberately targeted) -- "instrument a subset
                heavily". Total fleet exposure matches the spread case exactly,
                so the two differ only in how evidence is distributed.

    python sq1_evidence_horizon.py               # the thesis setting: 100 reps, scales 1 2 5 10 20
    python sq1_evidence_horizon.py --reps 200 --scales 1 2 5 10 20 50

writes results/sq1_horizon.json and the figure sq1_horizon.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline
from calibration import (CalibrationData, evidence_weights, fit_hyperparameters,
                         load_calibration_data, pool)

SCALES = [1, 2, 5, 10, 20]
REPS = 100
SEED = 11
NGRID = 90
CONC_FRACTION = 0.20          # share of component types receiving the added exposure


# ─────────────────────────────────────────────────────────────────────────────
def draw_truth(fit, rng) -> tuple:
    """One (alpha_0, c) from the fitted grid posterior: the replicate's truth."""
    w = fit.W.ravel()
    g = rng.choice(w.size, p=w)
    return float(fit.A0.ravel()[g]), float(fit.C.ravel()[g])


def scaled_exposure(T: np.ndarray, mu: np.ndarray, s: float,
                    mode: str, frac: float) -> np.ndarray:
    """Exposure after adding (s-1) x total, either spread or concentrated."""
    if s == 1.0 or mode == "spread":
        return T * s
    n_target = max(1, int(round(frac * T.size)))
    idx = np.argsort(-mu)[:n_target]                 # target the informative types
    extra_total = (s - 1.0) * T.sum()
    T_new = T.copy()
    T_new[idx] += extra_total * (T[idx] / T[idx].sum())
    return T_new


def one_replicate(data: CalibrationData, fit, s: float, mode: str,
                  frac: float, seed: int, ngrid: int) -> dict:
    rng = np.random.default_rng(seed)
    a0_t, c_t = draw_truth(fit, rng)

    beta = a0_t / (c_t * data.lam_prior)
    lam = rng.gamma(a0_t + data.k, 1.0 / (beta + data.T))     # conditional posterior

    T_new = scaled_exposure(data.T, data.mu, s, mode, frac)
    dT = np.maximum(T_new - data.T, 0.0)
    k_new = data.k + rng.poisson(lam * dT)

    d2 = CalibrationData(label=data.label, lam_prior=data.lam_prior,
                         T=T_new, k=k_new, mission=data.mission)
    f2 = fit_hyperparameters(d2, ngrid=ngrid)
    w = evidence_weights(d2, f2)

    return dict(
        c_med=f2.c[0], c_lo=f2.c[1], c_hi=f2.c[2],
        ci_width=f2.c[2] - f2.c[1], ci_ratio=f2.c[2] / f2.c[1],
        a0_med=f2.a0[0],
        w_med=float(np.median(w)),
        f_w50=float((w > 0.5).mean()), f_w90=float((w > 0.9).mean()),
        K=int(k_new.sum()),
        c_true=c_t, a0_true=a0_t,
        covered=bool(f2.c[1] <= c_t <= f2.c[2]),
    )


def summarise(reps: List[dict]) -> dict:
    def q(key, p=50):
        return float(np.percentile([r[key] for r in reps], p))
    return dict(
        n_reps=len(reps),
        ci_width=q("ci_width"), ci_width_lo=q("ci_width", 5), ci_width_hi=q("ci_width", 95),
        ci_ratio=q("ci_ratio"),
        w_med=q("w_med"), w_med_lo=q("w_med", 5), w_med_hi=q("w_med", 95),
        f_w50=q("f_w50"), f_w90=q("f_w90"),
        K=q("K"), K_lo=q("K", 5), K_hi=q("K", 95),
        c_med=q("c_med"),
        coverage=float(np.mean([r["covered"] for r in reps])),
    )


def run(csvs, scales, reps, seed, ngrid, frac) -> dict:
    data = pool([load_calibration_data(c) for c in csvs])
    fit = fit_hyperparameters(data, ngrid=ngrid)
    base_w = evidence_weights(data, fit)
    print(f"[horizon] baseline n={data.n} K={int(data.k.sum())} "
          f"c={fit.c[0]:.3f} [{fit.c[1]:.3f}, {fit.c[2]:.3f}]  "
          f"median w={np.median(base_w):.4f}")

    out: Dict[str, Dict[str, dict]] = {"spread": {}, "concentrated": {}}
    raw: Dict[str, Dict[str, list]] = {"spread": {}, "concentrated": {}}
    for mode in ("spread", "concentrated"):
        for si, s in enumerate(scales):
            if mode == "concentrated" and s == 1:
                continue
            rs = [one_replicate(data, fit, float(s), mode, frac,
                                seed + 10_000 * si + r, ngrid)
                  for r in range(reps)]
            out[mode][str(s)] = summarise(rs)
            raw[mode][str(s)] = rs
            g = out[mode][str(s)]
            print(f"[horizon] {mode:<12} s={s:<3} K={g['K']:6.0f}  "
                  f"CI width={g['ci_width']:.3f}  w_med={g['w_med']:.4f}  "
                  f"f(w>0.5)={g['f_w50']:.3f}  cov={g['coverage']:.2f}")

    return dict(
        settings=dict(scales=scales, reps=reps, seed=seed, ngrid=ngrid,
                      conc_fraction=frac),
        baseline=dict(n=int(data.n), K=int(data.k.sum()),
                      c=[float(x) for x in fit.c], a0=[float(x) for x in fit.a0],
                      w_med=float(np.median(base_w)),
                      f_w50=float((base_w > 0.5).mean()),
                      f_w90=float((base_w > 0.9).mean()),
                      total_Mh=float(data.T.sum() / 1e6)),
        summary=out, raw=raw)


# ─────────────────────────────────────────────────────────────────────────────
def figure(p: dict, out_png: Path) -> None:
    scales = p["settings"]["scales"]
    S = p["summary"]
    cs = [s for s in scales]

    def get(mode, key, xs):
        return [S[mode][str(s)][key] for s in xs]

    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.0))
    A = ax.ravel()
    # Spread pattern only: the concentrated arm is still computed and stored in the
    # JSON, but the report does not show it.
    kw_s = dict(marker="o", ms=5, lw=1.8, color="#3D9BE9")

    A[0].plot(cs, get("spread", "ci_ratio", cs), **kw_s)
    A[0].set_ylabel(r"$c$ credible-interval ratio  $q_{95}/q_{05}$")

    A[1].plot(cs, get("spread", "w_med", cs), **kw_s)
    A[1].axhline(0.5, color="black", ls=":", lw=1)
    A[1].set_ylabel(r"median evidence weight  $w_i$")
    A[1].set_yscale("log")

    A[2].plot(cs, get("spread", "f_w50", cs), **kw_s)
    A[2].set_ylabel(r"fraction with $w_i > 0.5$")

    A[3].plot(cs, get("spread", "f_w90", cs), **kw_s)
    A[3].set_ylabel(r"fraction with $w_i > 0.9$")

    A[4].plot(cs, get("spread", "K", cs), **kw_s)
    A[4].set_ylabel("observed failures $K$")

    A[5].plot(cs, get("spread", "coverage", cs), **kw_s)
    A[5].axhline(0.90, color="black", ls=":", lw=1)
    A[5].set_ylim(0.7, 1.0)
    A[5].set_ylabel("coverage of the true $c$")

    for a in A:
        a.set_xlabel(r"exposure multiple $s$")
        a.set_xscale("log")
        a.set_xticks(scales)
        a.set_xticklabels([str(s) + r"$\times$" for s in scales])
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def table(p: dict) -> None:
    b = p["baseline"]
    print("\n" + "=" * 96)
    print("EVIDENCE HORIZON — median over replicates")
    print("=" * 96)
    print(f"baseline: n={b['n']} components, K={b['K']} failures, "
          f"{b['total_Mh']:.1f} Mh, c CI [{b['c'][1]:.3f}, {b['c'][2]:.3f}], "
          f"median w={b['w_med']:.4f}, f(w>0.5)={b['f_w50']:.3f}")
    for mode in ("spread", "concentrated"):
        print(f"\n-- {mode} --")
        print(f"{'s':>4}{'K':>8}{'c CI width':>12}{'CI ratio':>10}"
              f"{'median w':>11}{'f(w>0.5)':>10}{'f(w>0.9)':>10}{'coverage':>10}")
        for s in p["settings"]["scales"]:
            g = p["summary"][mode].get(str(s))
            if not g:
                continue
            print(f"{s:>4}{g['K']:>8.0f}{g['ci_width']:>12.3f}{g['ci_ratio']:>10.2f}"
                  f"{g['w_med']:>11.4f}{g['f_w50']:>10.3f}{g['f_w90']:>10.3f}"
                  f"{g['coverage']:>10.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--missions", nargs="*", default=None)
    ap.add_argument("--scales", nargs="*", type=float, default=SCALES)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ngrid", type=int, default=NGRID)
    ap.add_argument("--conc-fraction", type=float, default=CONC_FRACTION)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    wb = pipeline.find_workbook(args.dir) if not args.workbook else Path(args.workbook)
    missions = args.missions or pipeline.list_missions(wb)
    csvs = [pipeline.run(m, workbook=wb) for m in missions]   # rebuilt from the workbook

    scales = [int(s) if float(s).is_integer() else s for s in args.scales]
    out_dir = Path(args.out_dir) if args.out_dir else pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    p = run(csvs, scales, args.reps, args.seed, args.ngrid, args.conc_fraction)
    table(p)

    slim = {k: v for k, v in p.items() if k != "raw"}
    (out_dir / "sq1_horizon.json").write_text(json.dumps(slim, indent=1))
    figure(p, out_dir / "sq1_horizon.png")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
