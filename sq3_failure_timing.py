"""SQ3, evidence: are in-orbit failures spread over a mission's life at the constant rate
the handbook assumes, and what bias do the handbook rates carry in routine operations?

No new model is introduced. The failure record is examined against the constant rate
that both the handbook and the calibration of SQ1 assume, and the SQ1 calibration is
then re-used UNCHANGED on the part of the record the handbook describes.

PART 1 - is the timing of the failures consistent with a constant rate?

  failure age   time from launch to each recorded failure. A failure dated before
                launch (a ground failure) is placed at launch, as pipeline.py does.

  trend test    under a constant rate, with the operating units running from launch,
                a failure is equally likely anywhere in its mission's observed life,
                so its relative position u = t / T_m is uniform on [0, 1] whatever the
                rate. The standardized mean  z = (mean(u) - 1/2) * sqrt(12 K)  is then
                approximately standard normal (a Laplace-type trend test); z < 0 means
                failures come early. Repeated without the records dated on or before
                launch day, without each mission in turn, and on the failures after
                the commissioning boundary only.

  early counts  failures in the first 30 / 90 / 365 days against the number a constant
                rate would place there, given each mission's own failure count. Exact
                Poisson-binomial tail.

  intensity     observed failures per age band divided by the handbook-expected failures
                accrued in that band (exact windowed exposure from pipeline.py), with a
                90% interval from a Jeffreys Gamma(K + 1/2, E) posterior.

  model check   posterior predictive check with the SQ1 calibration: replicate the fleet
                from the fitted model and compare the total failure count and the count
                in the first 90 days with what was observed.

PART 2 - what does this imply for how the data should update the handbook?

  refit         the SQ1 calibration, unchanged, on the exposure and failures after a
                commissioning boundary tau (30 / 90 / 180 / 365 days; 90 days is the
                reference), and leave-one-mission-out at the reference boundary, both
                on the full record and after the boundary.

  early phase   failures before tau against the handbook-expected failures before tau.

    python sq3_failure_timing.py

writes results/sq3_timing.json and four figures (sq3_*.png).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.stats import gamma as gamma_dist, norm

import pipeline
from calibration import (CalibrationData, FIT, fit_hyperparameters,
                         load_calibration_data, pool)

SEED = 11
HOUR_PER_DAY = 24.0
HOUR_PER_YEAR = 8766.0
REFERENCE_BOUNDARY_D = 90                   # end of launch and commissioning, days
BOUNDARIES_D = [30, 90, 180, 365]           # sensitivity of Part 2 to that boundary
EARLY_WINDOWS_D = [30, 90, 365]
BAND_EDGES_D = [0, 30, 90, 365, 3 * 365.25, 7 * 365.25, None]   # None = reference date
N_BAND_REPS = 20000                          # constant-rate envelope of the cumulative curve
N_PPC = 20000                                # posterior predictive replicates

# figure tokens (light surface): one accent for what was observed or what the
# handbook-scoped evidence gives, a neutral for the constant-rate reference
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_failures(wb: Path, missions: List[str]) -> List[dict]:
    """One row per failure record with its age and its mission's observed window."""
    f = pipeline.read_sheet(pipeline.FAILURES_SHEET, wb)
    rows = []
    for _, r in f.iterrows():
        m = str(r["mission"]).strip()
        if m not in missions:
            continue
        ctx_launch = pipeline.get_launch_date(m, wb)
        T = pipeline.get_survived_hours(m, wb)
        t_raw = (pipeline._parse_date(r["failure_date"]) - ctx_launch).total_seconds() / 3600.0
        t = max(0.0, min(t_raw, T))
        comp = pipeline.canonical_name(str(r["component"]).strip())
        rows.append(dict(mission=m, subsystem=str(r["subsystem"]).strip(), component=comp,
                         label=f"{str(r['subsystem']).strip()}/{comp}",
                         t_raw_h=t_raw, t_h=t, age_d=t / HOUR_PER_DAY, T_h=T, u=t / T,
                         pre_launch=bool(t_raw < 0), launch_day=bool(t_raw < HOUR_PER_DAY)))
    return rows


class Fleet:
    """The workbook's missions, with component tables buildable for any exposure window."""

    def __init__(self, wb: Path, missions: List[str], scratch: Path):
        self.wb, self.missions, self.scratch = wb, missions, scratch
        self.rbd = {m: pipeline.build_rbd(m, wb) for m in missions}
        self.ctx = {m: pipeline.MissionContext.load(m, wb) for m in missions}

    def data(self, window: Optional[Tuple[float, Optional[float]]] = None,
             exclude: Tuple[str, ...] = ()) -> CalibrationData:
        """Pooled calibration data for an exposure window (hours since launch), using
        exactly the SQ1 loader, so exclusions (no FIT, no exposure) are identical."""
        tag = "full" if window is None else f"w{window[0]:.0f}_{window[1]}"
        parts = []
        for m in self.missions:
            if m in exclude:
                continue
            ctx = self.ctx[m] if window is None else replace(self.ctx[m], window=window)
            path = self.scratch / f"{m}_{tag}.csv"
            if not path.exists():
                pipeline.build_clusters(self.rbd[m], ctx).to_csv(path, index=False)
            d = load_calibration_data(path, mission=m)
            if d.n:
                parts.append(d)
        return pool(parts)


def keyed(data: CalibrationData) -> Dict[Tuple[str, str], int]:
    return {(m, l): i for i, (m, l) in enumerate(zip(data.mission, data.label))}


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 - timing
# ─────────────────────────────────────────────────────────────────────────────
def trend_test(u: np.ndarray) -> dict:
    """Standardized mean of relative failure positions; N(0, 1) under a constant rate."""
    u = np.asarray(u, dtype=float)
    K = len(u)
    z = (u.mean() - 0.5) * np.sqrt(12.0 * K)
    return dict(K=int(K), mean_u=float(u.mean()), median_u=float(np.median(u)),
                z=float(z), p_two_sided=float(2 * norm.sf(abs(z))),
                p_early=float(norm.cdf(z)))


def poisson_binomial_sf(p: np.ndarray, k: int) -> float:
    """P(X >= k) for a sum of independent Bernoulli(p_j)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pj in p:
        dist[1:] = dist[1:] * (1 - pj) + dist[:-1] * pj
        dist[0] *= (1 - pj)
    return float(dist[k:].sum())


def jeffreys(K: int, E: float) -> Tuple[float, float]:
    lo, hi = gamma_dist.ppf([0.05, 0.95], K + 0.5, scale=1.0 / E)
    return float(lo), float(hi)


def timing_tests(fails: List[dict], boundary_h: float, missions: List[str]) -> dict:
    u = np.array([f["u"] for f in fails])
    out = dict(all=trend_test(u))
    out["without_pre_launch"] = trend_test([f["u"] for f in fails if not f["pre_launch"]])
    out["without_launch_day"] = trend_test([f["u"] for f in fails if not f["launch_day"]])
    out["leave_one_mission_out"] = {
        m: trend_test([f["u"] for f in fails if f["mission"] != m]) for m in missions}
    for label, tau in (("after_boundary", boundary_h), ("after_one_year", 365 * HOUR_PER_DAY)):
        late = [(f["t_h"] - tau) / (f["T_h"] - tau) for f in fails if f["t_h"] > tau]
        out[label] = trend_test(late)
    return out


def early_counts(fails: List[dict]) -> List[dict]:
    rows = []
    for d in EARLY_WINDOWS_D:
        W = d * HOUR_PER_DAY
        p = np.array([min(W, f["T_h"]) / f["T_h"] for f in fails])
        obs = int(sum(f["t_h"] <= W for f in fails))
        missions = sorted({f["mission"] for f in fails if f["t_h"] <= W})
        rows.append(dict(days=d, observed=obs, expected=float(p.sum()),
                         p_at_least=poisson_binomial_sf(p, obs),
                         missions_with_failure=missions))
    return rows


def intensity_by_age(fleet: Fleet, fails: List[dict], full: CalibrationData) -> List[dict]:
    key_full = keyed(full)
    rows = []
    edges = [e * HOUR_PER_DAY if e is not None else None for e in BAND_EDGES_D]
    for lo_h, hi_h in zip(edges[:-1], edges[1:]):
        d = fleet.data(window=(lo_h, hi_h))
        idx = [key_full[k] for k in zip(d.mission, d.label)]
        assert len(idx) == d.n, "a windowed component is missing from the full record"
        E = float((d.lam_prior * d.T).sum())
        hi_cmp = np.inf if hi_h is None else hi_h
        K = int(sum((f["t_h"] <= hi_cmp) if lo_h == 0 else (lo_h < f["t_h"] <= hi_cmp)
                    for f in fails))
        lo_r, hi_r = jeffreys(K, E)
        miss = np.array(d.mission)
        E_by_mission = {m: float((d.lam_prior * d.T)[miss == m].sum()) for m in fleet.missions}
        rows.append(dict(start_d=lo_h / HOUR_PER_DAY,
                         end_d=None if hi_h is None else hi_h / HOUR_PER_DAY,
                         K=K, E=E, ratio=K / E, lo=lo_r, hi=hi_r,
                         exposure_Mh=float(d.T.sum() / 1e6), E_by_mission=E_by_mission))
    return rows


def predictive_check(full: CalibrationData, early: CalibrationData) -> dict:
    """Replicate the fleet from the SQ1 posterior; compare total and first-window counts."""
    fit = fit_hyperparameters(full)
    T_early = np.zeros(full.n)
    kf = keyed(full)
    for j, k in enumerate(zip(early.mission, early.label)):
        T_early[kf[k]] = early.T[j]
    rng = np.random.default_rng(SEED)
    W = fit.W.ravel()
    g = rng.choice(W.size, size=N_PPC, p=W)
    a0s, cs = fit.A0.ravel()[g], fit.C.ravel()[g]
    tot, ear = np.empty(N_PPC, dtype=int), np.empty(N_PPC, dtype=int)
    for s in range(0, N_PPC, 2000):
        e = min(s + 2000, N_PPC)
        beta = a0s[s:e, None] / (cs[s:e, None] * full.lam_prior[None, :])
        lam = rng.gamma(a0s[s:e, None] + full.k[None, :], 1.0 / (beta + full.T[None, :]))
        tot[s:e] = rng.poisson(lam * full.T[None, :]).sum(axis=1)
        ear[s:e] = rng.poisson(lam * T_early[None, :]).sum(axis=1)
    k_tot, k_ear = int(full.k.sum()), int(early.k.sum())

    def summ(x, obs):
        return dict(observed=obs, median=float(np.median(x)),
                    lo=float(np.percentile(x, 5)), hi=float(np.percentile(x, 95)),
                    p_at_least=float((x >= obs).mean()), p_at_most=float((x <= obs).mean()),
                    hist=np.bincount(x).tolist())
    return dict(total=summ(tot, k_tot), first_window=summ(ear, k_ear))


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 - consequences for the update
# ─────────────────────────────────────────────────────────────────────────────
def fit_summary(d: CalibrationData) -> dict:
    fit = fit_hyperparameters(d)
    return dict(N=int(d.n), K=int(d.k.sum()), E=float(d.mu.sum()),
                exposure_Mh=float(d.T.sum() / 1e6),
                c=[float(x) for x in fit.c], a0=[float(x) for x in fit.a0],
                p_c_below_1=float(fit.W[fit.C < 1.0].sum()))


def refits(fleet: Fleet) -> dict:
    sweep = [dict(boundary_d=0, **fit_summary(fleet.data()))]
    early = []
    for d in BOUNDARIES_D:
        tau = d * HOUR_PER_DAY
        sweep.append(dict(boundary_d=d, **fit_summary(fleet.data(window=(tau, None)))))
        e = fleet.data(window=(0.0, tau))
        K, E = int(e.k.sum()), float(e.mu.sum())
        lo, hi = jeffreys(K, E)
        early.append(dict(boundary_d=d, K=K, E=E, ratio=K / E, lo=lo, hi=hi))
    tau = REFERENCE_BOUNDARY_D * HOUR_PER_DAY
    lomo = []
    for m in fleet.missions:
        lomo.append(dict(omitted=m,
                         full=fit_summary(fleet.data(exclude=(m,))),
                         routine=fit_summary(fleet.data(window=(tau, None), exclude=(m,)))))
    return dict(sweep=sweep, early_phase=early, leave_one_mission_out=lomo)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)


def _cumulative(ax, fails, scale_h, xmax, label_fmt, annotate_days=None):
    ages = np.sort([f["t_h"] / scale_h for f in fails])
    missions = sorted({f["mission"] for f in fails})
    T = {f["mission"]: f["T_h"] / scale_h for f in fails}
    Km = {m: sum(f["mission"] == m for f in fails) for m in missions}
    grid = np.linspace(0, xmax, 1200)
    expected = sum(Km[m] * np.minimum(grid, T[m]) / T[m] for m in missions)
    # constant-rate envelope: each mission's failures placed uniformly in its window
    rng = np.random.default_rng(SEED)
    sims = np.zeros((N_BAND_REPS, grid.size), dtype=np.int16)
    for m in missions:
        uu = rng.uniform(0, T[m], size=(N_BAND_REPS, Km[m]))
        for j in range(Km[m]):
            sims += grid[None, :] >= uu[:, j:j + 1]
    lo, hi = np.percentile(sims, [5, 95], axis=0)
    n = len(ages)
    ax.fill_between(grid, lo, hi, color=MUTED, alpha=0.18, lw=0,
                    label="constant rate: 90% range")
    ax.plot(grid, expected, color=MUTED, lw=2, label="constant rate: expected")
    shown = ages[ages <= xmax]
    xs = np.concatenate([[0.0], np.repeat(shown, 2), [xmax]])
    ys = np.concatenate([[0, 0], np.repeat(np.arange(1, shown.size + 1), 2)])
    ax.plot(xs, ys, color=BLUE, lw=2, label="observed")
    if annotate_days is not None:
        q = annotate_days * HOUR_PER_DAY / scale_h
        k = int((ages <= q).sum())
        e = float(np.interp(q, grid, expected))
        ax.plot([q], [k], "o", ms=6, color=BLUE, mec="white", mew=1.6, zorder=5)
        ax.annotate(label_fmt.format(k=k, e=e), xy=(q, k), xytext=(0.35 * xmax, 0.93 * (n + 2)),
                    color=INK, fontsize=8.5, va="top", ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=2, shrinkB=4,
                                    relpos=(0.0, 0.0)))
    ax.set_xlim(0, xmax)
    _style(ax)


def _counts_by_band(fails):
    """Observed and constant-rate expected failure counts in each age band."""
    tops = [e * HOUR_PER_DAY if e is not None else None for e in BAND_EDGES_D]
    missions = sorted({f["mission"] for f in fails})
    T = {f["mission"]: f["T_h"] for f in fails}
    Km = {m: sum(f["mission"] == m for f in fails) for m in missions}
    tops = [t if t is not None else max(T.values()) for t in tops]
    obs, exp, lab = [], [], []
    for i in range(len(tops) - 1):
        t1, t2 = tops[i], tops[i + 1]
        # the first band also collects the one pre-launch record
        obs.append(sum(1 for f in fails
                       if (f["t_h"] < t2 if i == 0 else t1 <= f["t_h"] < t2)))
        exp.append(sum(Km[m] * (min(t2, T[m]) - min(t1, T[m])) / T[m] for m in missions))

        def fmt(d):
            return f"{d:.0f} d" if d < 365 else f"{d / 365.25:.0f} y"
        d1, d2 = BAND_EDGES_D[i], BAND_EDGES_D[i + 1]
        lab.append(f"0-{fmt(d2)}" if i == 0 else
                   (f"> {fmt(d1)}" if d2 is None else f"{fmt(d1)}-{fmt(d2)}"))
    return obs, exp, lab


def fig_ages(fails, out_pngs):
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 3.9),
                           gridspec_kw=dict(width_ratios=[1.3, 1, 1.05]))
    xmax_y = max(f["T_h"] for f in fails) / HOUR_PER_YEAR
    _cumulative(ax[0], fails, HOUR_PER_YEAR, xmax_y, "")
    ax[0].set_xlabel("years since launch", color=INK2)
    ax[0].set_ylabel("cumulative failures, all missions", color=INK2)
    ax[0].set_ylim(0, len(fails) + 2)
    _cumulative(ax[1], fails, HOUR_PER_DAY, 365.0,
                "{k} failures in the first 90 days" + chr(10)
                + "{e:.1f} expected at a constant rate",
                annotate_days=REFERENCE_BOUNDARY_D)
    ax[1].set_ylim(0, len(fails) + 2)
    ax[1].set_xlabel("days since launch", color=INK2)

    obs, exp, lab = _counts_by_band(fails)
    x = np.arange(len(obs))
    ax[2].bar(x - 0.19, obs, width=0.38, color=BLUE, label="observed", zorder=3)
    ax[2].bar(x + 0.19, exp, width=0.38, color=MUTED, alpha=0.65,
              label="constant rate", zorder=3)
    for xi, k in zip(x, obs):
        ax[2].annotate(str(k), xy=(xi - 0.19, k), xytext=(0, 2),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=8, color=INK2)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(lab, fontsize=8, color=INK2)
    ax[2].set_ylabel("failures in band", color=INK2)
    ax[2].set_ylim(0, max(max(obs), max(exp)) * 1.22)
    ax[2].legend(loc="upper right", frameon=False, fontsize=8.5)
    _style(ax[2])

    h, l = ax[0].get_legend_handles_labels()
    order = [l.index("observed"), l.index("constant rate: expected"),
             l.index("constant rate: 90% range")]
    leg = ax[0].legend([h[i] for i in order], [l[i] for i in order], loc="lower right",
                       frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


def _band_label(b):
    def fmt(d):
        if d is None:
            return "end"
        return f"{d:.0f} d" if d < 365 else f"{d / 365.25:.0f} y"
    return f"{fmt(b['start_d']) if b['start_d'] > 0 else '0'}–{fmt(b['end_d'])}"


def fig_intensity(bands, overall, out_pngs):
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(bands))
    r = np.array([b["ratio"] for b in bands])
    lo = np.array([b["lo"] for b in bands])
    hi = np.array([b["hi"] for b in bands])
    ax.axhline(1.0, color=INK2, lw=0.9)
    ax.text(len(bands) - 0.55, 1.08, "handbook rate", color=INK2, fontsize=8.5, ha="right",
            va="bottom")
    ax.axhline(overall, color=MUTED, lw=0.9)
    ax.text(len(bands) - 0.55, overall * 0.92, f"whole record, {overall:.2f}", color=INK2,
            fontsize=8.5, ha="right", va="top")
    pos = r > 0
    ax.vlines(x, np.where(pos, lo, np.nan), hi, color=BLUE, lw=2)
    ax.plot(x[pos], r[pos], "o", ms=7, color=BLUE, mec="white", mew=1.6, zorder=5)
    zero = ~pos
    ax.vlines(x[zero], 0.02, hi[zero], color=BLUE, lw=2)
    ax.plot(x[zero], np.full(zero.sum(), 0.02), "v", ms=7, color=BLUE, mec="white", mew=1.2)
    ax.set_yscale("log")
    ax.set_ylim(0.015, 40)
    ticks = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_band_label(b)}\n{b['K']} obs / {b['E']:.1f} exp" for b in bands],
                       fontsize=8.5)
    ax.set_xlabel("time since launch", color=INK2)
    ax.set_ylabel("observed ÷ handbook-expected failures", color=INK2)
    _style(ax)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


def fig_ppc(ppc, out_pngs):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for a, key in zip(ax, ("total", "first_window")):
        s = ppc[key]
        h = np.array(s["hist"], dtype=float)
        h /= h.sum()
        xs = np.arange(len(h))
        a.bar(xs, h, width=0.86, color=MUTED, alpha=0.55, lw=0, label="replicated by the model")
        a.axvline(s["observed"], color=BLUE, lw=2, label="observed")
        a.set_xlim(-0.8, max(len(h), s["observed"] + 3))
        a.set_xlabel("number of failures", color=INK2)
        _style(a)
    ax[0].set_ylabel("share of replicates", color=INK2)
    ax[1].set_xlim(-0.8, max(15, ppc["first_window"]["observed"] + 3))
    leg = ax[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


def fig_bias(ref, out_pngs):
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    a = ax[0]
    rows = ref["sweep"]
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        col = MUTED if r["boundary_d"] == 0 else BLUE
        a.hlines(y, r["c"][1], r["c"][2], color=col, lw=2)
        a.plot(r["c"][0], y, "o", ms=7, color=col, mec="white", mew=1.6, zorder=5)
    a.axvline(1.0, color=INK2, lw=0.9)
    a.set_yticks(ys)
    a.set_yticklabels(["whole record" if r["boundary_d"] == 0 else
                       f"after {r['boundary_d']} days" for r in rows], fontsize=9)
    a.set_xscale("log")
    a.set_xlim(0.2, 2.2)
    a.set_xticks([0.25, 0.5, 1, 2])
    a.set_xticklabels(["0.25", "0.5", "1", "2"])
    a.set_xlabel("handbook bias $c$ (median, 90% interval)", color=INK2)
    _style(a, grid_axis="x")

    a = ax[1]
    lomo = ref["leave_one_mission_out"]
    ys = np.arange(len(lomo))[::-1]
    off = 0.17
    for y, r in zip(ys, lomo):
        for d, col, lab in ((+off, MUTED, "whole record"),
                            (-off, BLUE, f"after {REFERENCE_BOUNDARY_D} days")):
            s = r["full" if col == MUTED else "routine"]
            a.hlines(y + d, s["c"][1], s["c"][2], color=col, lw=2)
            a.plot(s["c"][0], y + d, "o", ms=6, color=col, mec="white", mew=1.4, zorder=5,
                   label=lab if y == ys[0] else None)
    a.axvline(1.0, color=INK2, lw=0.9)
    a.set_yticks(ys)
    a.set_yticklabels([f"without {r['omitted']}" for r in lomo], fontsize=9)
    a.set_xscale("log")
    a.set_xlim(0.2, 2.2)
    a.set_xticks([0.25, 0.5, 1, 2])
    a.set_xticklabels(["0.25", "0.5", "1", "2"])
    a.set_xlabel("handbook bias $c$ (median, 90% interval)", color=INK2)
    _style(a, grid_axis="x")
    leg = a.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def report(p: dict) -> None:
    t = p["timing"]
    print("\n" + "=" * 86)
    print("SQ3 - FAILURE TIMING")
    print("=" * 86)
    for name, r in [("all failures", t["all"]), ("without pre-launch", t["without_pre_launch"]),
                    ("without launch-day records", t["without_launch_day"]),
                    (f"after {REFERENCE_BOUNDARY_D} days", t["after_boundary"]),
                    ("after one year", t["after_one_year"])]:
        print(f"  {name:<28} K={r['K']:>2}  mean u={r['mean_u']:.3f}  z={r['z']:+.2f}  "
              f"p={r['p_two_sided']:.1e}")
    zs = [v["z"] for v in t["leave_one_mission_out"].values()]
    print(f"  leave-one-mission-out z in [{min(zs):+.2f}, {max(zs):+.2f}]")
    for r in p["early_counts"]:
        print(f"  first {r['days']:>3} d: {r['observed']:>2} observed, {r['expected']:.2f} expected, "
              f"P(>=obs)={r['p_at_least']:.1e}, missions {r['missions_with_failure']}")
    print("  intensity by age band (observed / handbook-expected):")
    for b in p["intensity_by_age"]:
        print(f"    {_band_label(b):<10} K={b['K']:>2} E={b['E']:6.2f} ratio={b['ratio']:6.2f} "
              f"[{b['lo']:.2f}, {b['hi']:.2f}]  exposure {b['exposure_Mh']:.1f} Mh")
    for k in ("total", "first_window"):
        s = p["predictive_check"][k]
        print(f"  PPC {k:<13} observed {s['observed']:>2}, replicated {s['median']:.0f} "
              f"[{s['lo']:.0f}, {s['hi']:.0f}], P(>=obs)={s['p_at_least']:.4f}")
    print("  refits:")
    for r in p["refits"]["sweep"]:
        print(f"    boundary {r['boundary_d']:>3} d: N={r['N']} K={r['K']:>2} E={r['E']:6.2f} "
              f"c={r['c'][0]:.3f} [{r['c'][1]:.3f}, {r['c'][2]:.3f}] a0={r['a0'][0]:.3f} "
              f"P(c<1)={r['p_c_below_1']:.3f}")
    for r in p["refits"]["early_phase"]:
        print(f"    before {r['boundary_d']:>3} d: K={r['K']:>2} E={r['E']:.2f} "
              f"ratio={r['ratio']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]")
    for r in p["refits"]["leave_one_mission_out"]:
        f, q = r["full"], r["routine"]
        print(f"    without {r['omitted']}: whole c={f['c'][0]:.3f} [{f['c'][1]:.3f}, {f['c'][2]:.3f}]"
              f"   after {REFERENCE_BOUNDARY_D} d c={q['c'][0]:.3f} [{q['c'][1]:.3f}, {q['c'][2]:.3f}]")
    print("=" * 86)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="folder holding the workbook")
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--out-dir", default=None, help="default: results/")
    args = ap.parse_args(argv)

    wb = pipeline.find_workbook(args.dir) if not args.workbook else Path(args.workbook)
    missions = pipeline.list_missions(wb)
    out_dir = Path(args.out_dir) if args.out_dir else pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tau = REFERENCE_BOUNDARY_D * HOUR_PER_DAY

    fails = load_failures(wb, missions)
    with tempfile.TemporaryDirectory(prefix="sq2_") as tmp:
        fleet = Fleet(wb, missions, Path(tmp))
        full = fleet.data()
        assert int(full.k.sum()) == len(fails), "every failure record must reach the fit"
        early = fleet.data(window=(0.0, tau))
        payload = dict(
            workbook=wb.name, missions=missions,
            settings=dict(seed=SEED, reference_boundary_d=REFERENCE_BOUNDARY_D,
                          boundaries_d=BOUNDARIES_D, early_windows_d=EARLY_WINDOWS_D,
                          band_edges_d=BAND_EDGES_D, band_reps=N_BAND_REPS, ppc_draws=N_PPC),
            failures=fails,
            timing=timing_tests(fails, tau, missions),
            early_counts=early_counts(fails),
            intensity_by_age=intensity_by_age(fleet, fails, full),
            predictive_check=predictive_check(full, early),
            refits=refits(fleet),
        )
    payload["whole_record_ratio"] = float(full.k.sum() / full.mu.sum())
    report(payload)

    (out_dir / "sq3_timing.json").write_text(json.dumps(payload, indent=1))
    fig_ages(fails, [out_dir / "sq3_failure_ages.png"])
    fig_intensity(payload["intensity_by_age"], payload["whole_record_ratio"],
                  [out_dir / "sq3_intensity_by_age.png"])
    fig_ppc(payload["predictive_check"], [out_dir / "sq3_predictive_check.png"])
    fig_bias(payload["refits"], [out_dir / "sq3_routine_bias.png"])
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
