"""
sensitivity.py — SQ1 sensitivity analysis of the conjugate calibration.

calibration.py answers the FIRST half of SQ1 (optimise the prior-strength
parameters); this script answers the SECOND half: how robust are the resulting
posterior reliability estimates to every choice the method makes? It re-runs
fit_hyperparameters() under systematic perturbations of

    A. the hyper-priors on (log alpha_0, log c)   — sd sweep, centre shifts
    B. the data                                    — re-included failures, leave-
                                                     one-out, input scalings
    C. hypothetical evidence                       — synthetic failure injection
    D. the model structure                         — no-bias, fixed-strength,
                                                     point plug-in vs mixture
    E. the numerics                                — grid resolution/range, MC seed

and measures, for every scenario, the movement of: the (alpha_0, c) posterior,
the per-component posterior means (vs the baseline, matched by (mission, subsystem,
component); median and maximum % change across components), and the typical survivor
(median over zero-failure components of posterior mean / handbook).

Like calibration.py this runs on one mission or on several pooled together: give it
a list of components CSVs and the whole scenario suite is re-run on the pooled fleet,
so the reported sensitivity is that of the single shared (alpha_0, c). Components are
matched to the baseline per mission, so same-named components from different missions
are never confused with each other.

Usage:
    python sensitivity.py [components_csv ...] [--name NAME] [--out-dir DIR]
writes four figures and <NAME>_sensitivity.json next to the CSV (single mission) or
into results/ (pooled; sq1_update.py drives this run).
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import replace as dc_replace
from typing import List, Optional

from scipy.stats import gamma as gamma_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibration
from calibration import (CalibrationData, load_pooled, as_csv_list, resolve_output,
                         fit_hyperparameters, component_posteriors, row_key,
                         FIT, A0_RANGE, C_RANGE, NGRID, PRIOR_SD, SEED)

PRIOR_SD_SWEEP = [0.5, 0.75, 1.0, 1.5, 2.0]   # joint sweep of both hyper-prior sds


# ─────────────────────────────────────────────────────────────────────────────
# Data perturbations (all return a NEW CalibrationData; the input is never touched)
# ─────────────────────────────────────────────────────────────────────────────
def scale_priors(d: CalibrationData, f: float) -> CalibrationData:
    return dc_replace(d, lam_prior=d.lam_prior * f)


def scale_exposure(d: CalibrationData, f: float) -> CalibrationData:
    return dc_replace(d, T=d.T * f)


def inject_failure(d: CalibrationData, i: int) -> CalibrationData:
    k = d.k.copy(); k[i] += 1
    return dc_replace(d, k=k)


def drop_component(d: CalibrationData, i: int) -> CalibrationData:
    m = np.ones(d.n, bool); m[i] = False
    return d.subset(m)


def add_component(d: CalibrationData, mission: str, label: str, lam: float, T: float,
                  k: int) -> CalibrationData:
    return d.with_extra(mission, label, lam, T, k)


def excluded_failures(components_csvs) -> List[dict]:
    """Failed components that load_calibration_data drops for lack of a usable
    prior (placeholder FIT of 0 or blank) but that DO have exposure — candidates
    for re-inclusion with an imputed prior. Scanned across every pooled mission."""
    out = []
    for csv in as_csv_list(components_csvs):
        df = pd.read_csv(csv)
        fit = pd.to_numeric(df['fit'], errors='coerce')
        hours = pd.to_numeric(df['operational_hours'], errors='coerce')
        fails = pd.to_numeric(df['n_failures'], errors='coerce')
        for i, r in df.iterrows():
            if (fails[i] or 0) > 0 and (pd.isna(fit[i]) or fit[i] <= 0):
                if pd.notna(hours[i]) and hours[i] > 0:
                    out.append(dict(mission=str(r['mission']).strip(),
                                    label=f"{r['subsystem_name']}/{r['component_name']}",
                                    T=float(hours[i]), k=int(fails[i])))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scenario evaluation
# ─────────────────────────────────────────────────────────────────────────────
def shrink_of(rows: List[dict]) -> float:
    """Typical survivor: the median, over zero-failure components, of each component's
    posterior MEAN divided by its handbook rate (each component is summarised by its
    mean, the median is taken across components)."""
    surv = [r for r in rows if r["k"] == 0 and r["hb"] > 0]
    return float(np.median([r["mean"] / r["hb"] for r in surv]))


def compare_components(rows: List[dict], base_rows: List[dict]) -> dict:
    """How far the components move against the baseline. Each component is summarised
    by its posterior MEAN; the absolute % change is taken per component, and the median
    and the maximum are taken across the components present in both. Matching is by
    (mission, subsystem, component), so a component name that occurs in more than one
    mission is compared against its own mission's baseline entry."""
    base = {row_key(r): r["mean"] for r in base_rows}
    dv = [abs(r["mean"] / base[row_key(r)] - 1.0) * 100
          for r in rows if row_key(r) in base]
    return dict(move_med=round(float(np.median(dv)), 1),
                move_max=round(float(np.max(dv)), 1))


def run_scenario(name: str, group: str, note: str, data: CalibrationData,
                 base_rows: Optional[List[dict]], seed: int = SEED,
                 **fit_kw) -> dict:
    fit = fit_hyperparameters(data, **fit_kw)
    rows = component_posteriors(data, fit, seed=seed)
    out = dict(name=name, group=group, note=note,
               a0=[round(v, 3) for v in fit.a0], c=[round(v, 3) for v in fit.c],
               shrink=round(shrink_of(rows), 3))
    if base_rows is not None:
        out.update(compare_components(rows, base_rows))
    return out, fit, rows


def point_plugin_rows(data: CalibrationData, a0: float, c: float) -> List[dict]:
    """Analytic per-component posteriors with (a0, c) plugged in as a point
    (no grid mixture) — what ignoring hyperparameter uncertainty would give."""
    rows = []
    for i in range(data.n):
        beta = a0 / (c * data.lam_prior[i])
        g = gamma_dist(data.k[i] + a0, scale=1.0 / (beta + data.T[i]))
        lo, med, hi = (g.ppf(q) * FIT for q in (.05, .5, .95))
        sub, comp = (data.label[i].split("/", 1) + [""])[:2]
        rows.append(dict(mission=data.mission[i], sub=sub.strip(), name=comp.strip(),
                         k=int(data.k[i]), hb=float(data.lam_prior[i] * FIT),
                         med=float(med), lo=float(lo), hi=float(hi), mean=float(g.mean() * FIT)))
    return rows


def ci_width_ratio(rows_a: List[dict], rows_b: List[dict]) -> float:
    """Median ratio of 90% CI widths (log-rate) A/B, matched by
    (mission, subsystem, component)."""
    b = {row_key(r): r for r in rows_b}
    ratios = [(np.log(r["hi"]) - np.log(r["lo"])) /
              (np.log(b[row_key(r)]["hi"]) - np.log(b[row_key(r)]["lo"]))
              for r in rows_a if row_key(r) in b]
    return float(np.median(ratios))


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_priorsd(sweep: List[dict], mission: str, out_png: Path) -> None:
    """Hyper-prior sd sweep: how (alpha_0, c, shrink) respond to the only truly
    arbitrary setting of the method."""
    sd = [s["sd"] for s in sweep]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
#    fig.suptitle(f"{mission} — sensitivity to the hyper-prior sd  "
#                 r"(log $\alpha_0$, log c ~ N(0, sd$^2$), swept jointly)",
#                 fontsize=13, fontweight="bold")
    for a, key, col, lab in ((ax[0], "c", "C2", "bias c"),
                             (ax[1], "a0", "C0", r"strength $\alpha_0$")):
        med = [s[key][0] for s in sweep]
        lo = [s[key][1] for s in sweep]; hi = [s[key][2] for s in sweep]
        a.fill_between(sd, lo, hi, color=col, alpha=.15, label="90% CI")
        a.plot(sd, med, "o-", color=col, lw=2, label="posterior median")
        a.axvline(1.0, color="C3", ls="--", lw=1, label="baseline sd = 1")
        if key == "c":
            a.axhline(1, color="k", ls=":", lw=1)
        a.set_yscale("log")
        a.set_xlabel("hyper-prior sd"); a.set_ylabel(lab)
        a.legend(fontsize=8); a.grid(alpha=.3, which="both")
    a = ax[2]
    a.plot(sd, [s["shrink"] for s in sweep], "o-", color="C3", lw=2, label="typical survivor")
    a.axvline(1.0, color="C3", ls="--", lw=1, label="baseline sd = 1")
    a.legend(fontsize=8, loc="lower right")
    a.set_xlabel("hyper-prior sd"); a.set_ylabel("typical survivor  posterior mean / handbook")
    a.set_ylim(0, 1)    # full ratio range, so the size of the movement is read correctly
    a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_tornado(scen: List[dict], base: dict, mission: str, out_png: Path) -> None:
    """Every scenario's displacement of the bias c and of the typical survivor."""
    rows = [s for s in scen if s.get("tornado", True)]
    rows = sorted(rows, key=lambda s: abs(np.log(s["c"][0] / base["c"][0])))
    y = np.arange(len(rows))
    gcol = {"hyper-prior": "C0", "data": "C2", "injection": "C3",
            "structure": "C4", "numerics": "C7"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.34 * len(rows) + 2.4), sharey=True)
#    fig.suptitle(f"{mission} — scenario tornado: movement of bias c and typical "
#                 f"survivor vs baseline", fontsize=13, fontweight="bold")
    for a, key, xlab in ((ax[0], "c", f"bias c (baseline {base['c'][0]})"),
                         (ax[1], "shrink", f"typical survivor mean/handbook (baseline {base['shrink']})")):
        for i, s in enumerate(rows):
            v = s[key][0] if key == "c" else s[key]
            b = base[key][0] if key == "c" else base[key]
            a.barh(i, v - b, left=b, color=gcol[s["group"]], alpha=.85, height=.62)
        a.axvline(b, color="k", lw=1.2)
        a.set_xlabel(xlab); a.grid(alpha=.3, axis="x")
    ax[0].set_yticks(y); ax[0].set_yticklabels([s["name"] for s in rows], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=.85) for c in gcol.values()]
    ax[1].legend(handles, gcol.keys(), fontsize=8, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130); plt.close(fig)


def fig_loo(loo: List[dict], c_base: float, mission: str, out_png: Path,
            multi: bool = False) -> None:
    """Leave-one-out influence on the bias c.

    One bar per component is unreadable at fleet scale, so the left panel names
    only the components that actually move the estimate and the right panel shows
    that the remainder do not."""
    dc = np.array([l["c"] - c_base for l in loo])
    n_top = int(min(18, len(loo)))
    order = np.argsort(-np.abs(dc))[:n_top][::-1]       # least influential at the bottom

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 5.0),
                           gridspec_kw=dict(width_ratios=[1.55, 1]))

    a = ax[0]
    y = np.arange(n_top)
    a.barh(y, dc[order], height=0.72,
           color=["C3" if v < 0 else "C0" for v in dc[order]])
    labels = []
    for j in order:
        lbl = loo[j]["name"].split("/", 1)[-1]
        if multi and loo[j].get("mission"):
            lbl = f"{lbl} [{loo[j]['mission']}]"
        labels.append(lbl if len(lbl) <= 44 else lbl[:42] + "...")
    a.set_yticks(y)
    a.set_yticklabels(labels, fontsize=8)
    a.axvline(0, color="k", lw=1)
    a.set_ylim(-0.7, n_top - 0.3)
    a.set_xlabel(r"$\Delta$ bias c median (leave-one-out $-$ baseline)")
    a.grid(alpha=.3, axis="x")

    b = ax[1]
    b.hist(dc, bins=44, color="C0")
    b.set_yscale("log")
    b.axvline(0, color="k", lw=1)
    b.set_xlabel(r"$\Delta$ bias c median")
    b.set_ylabel("components (log scale)")
    b.grid(alpha=.3, axis="y")
    quiet = int((np.abs(dc) < 0.0005).sum())
    b.annotate(f"{quiet} of {len(loo)} leave the" + chr(10) + "estimate unmoved",
               xy=(0.03, 0.94), xycoords="axes fraction", ha="left", va="top",
               fontsize=8.5, color="#444444")

    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def fig_inject(curves: List[dict], mission: str, out_png: Path) -> None:
    """What one hypothetical failure would do to the (alpha_0, c) marginals."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
#    fig.suptitle(f"{mission} — synthetic failure injection: posterior of "
#                 r"($\alpha_0$, c) if ONE more failure were observed",
#                 fontsize=13, fontweight="bold")
    for a, key, xlab in ((ax[0], "c_marg", "bias c"),
                         (ax[1], "a_marg", r"strength $\alpha_0$")):
        for cu in curves:
            x, yv = cu[key]
            a.plot(x, yv / np.max(yv), lw=2.2 if cu["style"] == "base" else 1.6,
                   ls="-" if cu["style"] == "base" else "--",
                   color=cu["color"], label=cu["label"])
        if key == "c_marg":
            a.axvline(1, color="k", ls=":", lw=1)
        a.set_xscale("log"); a.set_xlabel(xlab); a.set_ylabel("rel. posterior density")
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run(components_csv, save: bool = True, name: Optional[str] = None,
        out_dir=None) -> dict:
    """Run the full scenario suite on one mission or on several pooled together.

    `components_csv` is a single components CSV or a list of them; with a list the
    perturbations act on the pooled fleet, so what is being stress-tested is the one
    shared (alpha_0, c).
    """
    csvs = as_csv_list(components_csv)
    data = load_pooled(csvs)
    if data.n == 0:
        raise ValueError(f"Nothing to analyse in {', '.join(str(c) for c in csvs)}.")
    mission, out_dir = resolve_output(csvs, data, name, out_dir)
    missions = data.missions
    multi = len(missions) > 1

    def tag(i: int) -> str:
        """Component label for scenario names, mission-qualified when pooling."""
        short = data.label[i].split("/", 1)[-1]
        return f"{short} [{data.mission[i]}]" if multi else short

    # ── baseline ─────────────────────────────────────────────────────────────
    base, base_fit, base_rows = run_scenario(
        "baseline", "baseline", "settings exactly as calibration.py", data, None)
    base["tornado"] = False
    scen = [base]

    def add(*args, **kw):
        s, fit, rows = run_scenario(*args, base_rows=base_rows, **kw)
        scen.append(s)
        return s, fit, rows

    # boundary-mass check on the baseline grid (edge rows/columns of W)
    W = base_fit.W
    boundary = dict(a0_lo=float(W[0, :].sum()), a0_hi=float(W[-1, :].sum()),
                    c_lo=float(W[:, 0].sum()), c_hi=float(W[:, -1].sum()))

    # ── A. hyper-prior sensitivity ───────────────────────────────────────────
    sweep = []
    for sd in PRIOR_SD_SWEEP:
        if sd == 1.0:
            s = dict(base, name=f"prior sd = {sd}")
        else:
            s, _, _ = add(f"prior sd = {sd}", "hyper-prior",
                          f"both hyper-prior sds set to {sd} (baseline 1.0)",
                          data, prior_sd_a0=sd, prior_sd_c=sd)
        sweep.append(dict(s, sd=sd))
    add("sd(log a0) = 0.5 only", "hyper-prior", "strength prior tighter", data, prior_sd_a0=0.5)
    add("sd(log a0) = 2.0 only", "hyper-prior", "strength prior looser", data, prior_sd_a0=2.0)
    add("sd(log c) = 0.5 only", "hyper-prior", "bias prior tighter", data, prior_sd_c=0.5)
    add("sd(log c) = 2.0 only", "hyper-prior", "bias prior looser", data, prior_sd_c=2.0)
    add("centre c at 0.5", "hyper-prior", "prior expects a 2x-pessimistic handbook",
        data, prior_mean_lc=np.log(0.5))
    add("centre c at 2.0", "hyper-prior", "prior expects a 2x-optimistic handbook",
        data, prior_mean_lc=np.log(2.0))
    add("centre a0 at 0.2", "hyper-prior", "prior expects a weak handbook prior",
        data, prior_mean_la0=np.log(0.2))
    add("centre a0 at 5.0", "hyper-prior", "prior expects a strong handbook prior",
        data, prior_mean_la0=np.log(5.0))

    # ── B. data sensitivity ──────────────────────────────────────────────────
    lam_med = float(np.median(data.lam_prior))
    lam_q90 = float(np.quantile(data.lam_prior, .90))
    readd = excluded_failures(csvs)
    readded = []
    for r in readd:
        who = r["label"].split("/", 1)[-1] + (f" [{r['mission']}]" if multi else "")
        for how, lam in (("median", lam_med), ("q90", lam_q90)):
            d2 = add_component(data, r["mission"], r["label"], lam, r["T"], r["k"])
            s, _, _ = add(f"re-include {who} ({how} prior)",
                          "data",
                          f"failed component re-included with imputed handbook prior = "
                          f"fleet {how} ({lam*FIT:.1f} FIT), T={r['T']:.0f} h, k={r['k']}",
                          d2)
            readded.append(dict(s, mission=r["mission"], target=r["label"],
                                imputed_FIT=round(lam * FIT, 1)))
    add("handbook rates x2", "data", "every handbook FIT doubled",
        scale_priors(data, 2.0))
    add("handbook rates /2", "data", "every handbook FIT halved", scale_priors(data, 0.5))
    add("exposure +20%", "data", "all operational hours +20%", scale_exposure(data, 1.2))
    add("exposure -20%", "data", "all operational hours -20%", scale_exposure(data, 0.8))

    # leave-one-out jackknife (fit only; per-component posteriors not needed).
    # A finer grid (180) so the small median displacements are resolved rather
    # than quantised to the 90-point grid step; the baseline is refit on the
    # same grid so the differences are apples-to-apples.
    base180 = fit_hyperparameters(data, ngrid=180)
    loo = []
    for i in range(data.n):
        f = fit_hyperparameters(drop_component(data, i), ngrid=180)
        loo.append(dict(mission=data.mission[i], name=data.label[i],
                        mu=float(data.mu[i]),
                        c=round(f.c_med, 4), a0=round(f.a0_med, 4)))
    loo_base = dict(c=[round(v, 4) for v in base180.c],
                    a0=[round(v, 4) for v in base180.a0])

    # ── C. synthetic failure injection ───────────────────────────────────────
    mu = data.mu
    targets = [("highest-mu", int(np.argmax(mu))),
               ("median-mu", int(np.argsort(mu)[data.n // 2])),
               ("lowest-mu", int(np.argmin(mu)))]
    inj_curves = [dict(label=f"baseline (K={int(data.k.sum())}), c med {base['c'][0]}",
                       style="base", color="k",
                       c_marg=(base_fit.C[0, :], base_fit.W.sum(axis=0)),
                       a_marg=(base_fit.A0[:, 0], base_fit.W.sum(axis=1)))]
    inj = []
    for j, (kind, i) in enumerate(targets):
        who = tag(i)
        s, f, _ = add(f"inject 1 failure: {who}", "injection",
                      f"{kind} component (mu={mu[i]:.3f} expected failures, "
                      f"T={data.T[i]:.0f} h, handbook {data.lam_prior[i]*FIT:.1f} FIT)",
                      inject_failure(data, i))
        inj.append(dict(s, mission=data.mission[i], target=data.label[i],
                        mu=round(float(mu[i]), 4)))
        inj_curves.append(dict(label=f"{who} ({kind}), c med {s['c'][0]}",
                               style="inj", color=f"C{j}",
                               c_marg=(f.C[0, :], f.W.sum(axis=0)),
                               a_marg=(f.A0[:, 0], f.W.sum(axis=1))))

    # ── D. structural sensitivity ────────────────────────────────────────────
    add("no bias (c = 1 fixed)", "structure", "strength-only model", data, fix_c=1.0)
    for a0v in (0.1, 1.0, 10.0):
        add(f"a0 fixed = {a0v}", "structure", "strength pinned, bias free", data, fix_a0=a0v)
    pt_rows = point_plugin_rows(data, base_fit.a0_med, base_fit.c_med)
    point = dict(name="point plug-in (a0,c) = posterior median", group="structure",
                 note="no grid mixture: hyperparameter uncertainty ignored",
                 a0=base["a0"], c=base["c"], shrink=round(shrink_of(pt_rows), 3),
                 ci_ratio=round(ci_width_ratio(pt_rows, base_rows), 3),
                 **compare_components(pt_rows, base_rows), tornado=False)
    scen.append(point)

    # ── E. numerics ──────────────────────────────────────────────────────────
    add("grid 45 x 45", "numerics", "half the grid resolution", data, ngrid=45)
    add("grid 180 x 180", "numerics", "double the grid resolution", data, ngrid=180)
    add("wider grid ranges", "numerics", "a0 in [0.01,100], c in [0.01,20]",
        data, a0_range=(0.01, 100.0), c_range=(0.01, 20.0))
    add("MC seed 12", "numerics", "same fit, different posterior draws (pure MC error)",
        data, seed=12)

    # ── figures + JSON ───────────────────────────────────────────────────────
    results = dict(mission=mission, missions=missions,
                   by_mission=calibration.mission_breakdown(data),
                   baseline=base, scenarios=scen[1:],
                   sweep=[dict(sd=s["sd"], c=s["c"], a0=s["a0"], shrink=s["shrink"])
                          for s in sweep],
                   loo=loo, loo_base=loo_base, boundary_mass=boundary,
                   readded=readded, injections=inj,
                   imputed=dict(median_FIT=round(lam_med * FIT, 1),
                                q90_FIT=round(lam_q90 * FIT, 1)),
                   settings=dict(ngrid=NGRID, prior_sd=PRIOR_SD,
                                 a0_range=list(A0_RANGE), c_range=list(C_RANGE)))

    print("=" * 74)
    print(f"SENSITIVITY ANALYSIS — {mission}  (baseline c = {base['c'][0]}, "
          f"a0 = {base['a0'][0]}, shrink = {base['shrink']})")
    if multi:
        print(f"  pooled over {len(missions)} missions: {', '.join(missions)}  "
              f"(N = {data.n} components, never merged across missions)")
    print("=" * 74)
    for s in scen[1:]:
        extra = f"  component means moved med {s.get('move_med','-')}% / max {s.get('move_max','-')}%"
        print(f"  [{s['group']:<11}] {s['name']:<38} c={s['c'][0]:<7} "
              f"shrink={s['shrink']:<7}{extra}")
    print(f"  boundary mass: {', '.join(f'{k}={v:.2e}' for k, v in boundary.items())}")
    print("=" * 74)

    if save:
        fig_priorsd(sweep, mission, out_dir / f"{mission}_sens_priorsd.png")
        fig_tornado(scen[1:], base, mission, out_dir / f"{mission}_sens_tornado.png")
        fig_loo(loo, loo_base["c"][0], mission, out_dir / f"{mission}_sens_loo.png",
                multi=multi)
        fig_inject(inj_curves, mission, out_dir / f"{mission}_sens_inject.png")
        with open(out_dir / f"{mission}_sensitivity.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"figures + {mission}_sensitivity.json written to {out_dir}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="SQ1 sensitivity analysis of the "
                                             "conjugate calibration. Give several "
                                             "components CSVs to stress-test the "
                                             "pooled multi-mission fit.")
    ap.add_argument("components_csv", nargs="*",
                    help="pipeline components CSV(s) (default: every mission in "
                         "the workbook)")
    ap.add_argument("--name", default=None,
                    help="output name for a pooled run (default: FLEET)")
    ap.add_argument("--out-dir", default=None, help="where to write figures + JSON")
    ap.add_argument("--no-show", action="store_true", help="(figures are always saved)")
    args = ap.parse_args(argv)
    run(args.components_csv or calibration.default_components_csvs(),
        name=args.name, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
