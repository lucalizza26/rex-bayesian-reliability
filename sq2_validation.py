"""SQ2 and SQ3: do updated failure rates predict a mission they have not seen better than
the handbook (SQ2), and does a separate early phase predict better still (SQ3)?

Leave-one-mission-out cross-validation. In each fold one mission is held out, the
shared scalars are fitted on the other seven with the SQ1 calibration, unchanged, and
the held-out mission's failure record is scored under the resulting prediction. The
held-out mission's own failures never enter its prediction: what is tested is the
fleet-level learning (c, alpha_0) that a new mission would inherit, not the
per-component conjugate update with a component's own record.

SCORE. The log predictive density of each held-out mission's failure times, summed over
the eight folds. Within a phase p a component's failures come at rate lambda_p times the
number of its units operating, so its contribution is

    log Gamma(a + k) - log Gamma(a) + a log p + k log(1 - p) - k log T     (Gamma rate)
    k log lambda_hb - mu                                                  (handbook)

with k, T, mu its failures, exposure and handbook-expected failures in that phase and
p = a / (a + c mu); the per-mission score is the log of the grid-posterior average of
the product over the mission's components. The number of units operating at each
failure is common to every model and is dropped, so scores are compared, never read
in isolation, and are reported relative to the handbook. Because every model is scored
on the same failure times, the split models and every boundary are comparable with
each other and with the single-phase update. Differences come with their per-mission
mean, paired standard error, the number of missions won, and the number of missions at
which the mean difference would reach two standard errors, 4 s^2 / mean^2.

MODELS

  handbook   Poisson(handbook rate x exposure); nothing is fitted
  sq1        the SQ1 update: c and alpha_0 learned on the other seven missions and
             integrated over (SQ2)
  split @ x  the SQ1 model fitted separately to the first x days of every mission and
             to the rest; each phase has its own constant rate, c and alpha_0
             (BOUNDARIES_D). The early-phase c grid is widened to C_RANGE_EARLY,
             since the early rate runs far above the SQ1 grid's ceiling of 6; the
             priors are unchanged (SQ3).
  selected   split, with the boundary (or no split) chosen inside each fold by the
             same cross-validation on the other seven missions only, so the held-out
             mission plays no part in choosing it (SQ3).

The split sweep is repeated without the failure records dated before or on launch day.

    python sq2_validation.py

writes results/sq2_validation.json and the figures sq2_validation.png and sq3_boundary.png.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.special import gammaln, logsumexp

import pipeline
from calibration import C_RANGE, CalibrationData, fit_hyperparameters
from sq3_failure_timing import AXIS, BLUE, GRID, HOUR_PER_DAY, INK, INK2, MUTED, Fleet

SEED = 11                                   # nothing here is random; kept for the record
BOUNDARIES_D = [7, 15, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730]
REFERENCE_BOUNDARY_D = 90                   # the routine-operations boundary of SQ3
C_RANGE_EARLY = (0.05, 300.0)               # early-phase c grid; SQ1 uses C_RANGE
# Posterior mass allowed on a grid edge. Fold fits put up to 0.7% on the SQ1 alpha_0
# grid's edge; refitting on (0.005, 200) moved no per-mission score by more than
# 0.006, so the SQ1 grid is kept and only a larger share is flagged.
EDGE_WARN = 1e-2
NONE = "none"                               # the no-split candidate in the selection
LAUNCH_DAY_H = 24.0                         # records dated before this age are "launch day"


# ─────────────────────────────────────────────────────────────────────────────
# Scores
# ─────────────────────────────────────────────────────────────────────────────
def _time_term(d: CalibrationData) -> float:
    """-sum k log T: places each failure inside its phase's exposure."""
    hit = d.k > 0
    assert np.all(d.T[hit] > 0), "a failure with no exposure in its phase"
    return float(-np.sum(d.k[hit] * np.log(d.T[hit])))


def score_handbook(d: CalibrationData) -> float:
    """k log lambda_hb - mu: the Poisson limit of score_fit, time term included
    (k log lambda = k log mu - k log T)."""
    hit = d.k > 0
    return float(np.sum(d.k[hit] * np.log(d.lam_prior[hit])) - d.mu.sum())


def score_fit(d: CalibrationData, fit) -> float:
    """Log predictive density of the table's failure times under a grid fit, joint
    over its components (they share the scalars)."""
    if d.n == 0:
        return 0.0
    w = fit.W.ravel()
    keep = w > 1e-14 * w.max()
    a0 = fit.A0.ravel()[keep][:, None]
    c = fit.C.ravel()[keep][:, None]
    k, mu = d.k[None, :], d.mu[None, :]
    p = a0 / (a0 + c * mu)
    ll = (gammaln(k + a0) - gammaln(a0) + a0 * np.log(p) + k * np.log1p(-p)).sum(axis=1)
    return float(logsumexp(ll + np.log(w[keep] / w[keep].sum()))) + _time_term(d)


def decompose(d: CalibrationData, a0: float, c: float) -> dict:
    """Where the gain over the handbook comes from, at fixed (a0, c): the difference
    in log score per component, summed over the components without and with a failure.
    The time term is common to both scores and cancels."""
    p = a0 / (a0 + c * d.mu)
    nb = gammaln(d.k + a0) - gammaln(a0) + a0 * np.log(p) + d.k * np.log1p(-p)
    pois = np.where(d.k > 0, d.k * np.log(np.maximum(d.mu, 1e-300)), 0.0) - d.mu
    diff = nb - pois
    fail = d.k > 0
    return dict(survivors=float(diff[~fail].sum()), failed=float(diff[fail].sum()),
                total=float(diff.sum()), a0=float(a0), c=float(c))


def edge_mass(fit) -> Dict[str, float]:
    """Posterior mass on the outermost grid nodes of each free axis."""
    W = fit.W
    out = {}
    if W.shape[1] > 1:
        out["c"] = float(W[:, 0].sum() + W[:, -1].sum())
    if W.shape[0] > 1:
        out["a0"] = float(W[0, :].sum() + W[-1, :].sum())
    return out


def compare(a: Dict[str, float], b: Dict[str, float], missions: List[str]) -> dict:
    """Model a against model b on the same held-out missions."""
    d = np.array([a[m] - b[m] for m in missions])
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    mean = float(d.mean())
    return dict(sum=float(d.sum()), mean=mean, sd=sd, se=float(se),
                z=float(mean / se) if se > 0 else float("nan"),
                wins=int((d > 0).sum()), n=len(d),
                missions_needed=(float(4 * sd ** 2 / mean ** 2) if mean > 0 else None),
                per_mission={m: float(x) for m, x in zip(missions, d)})


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
class Tables:
    """The whole-record table and the early / late tables of every boundary, with
    per-mission views and fits that leave a set of missions out."""

    def __init__(self, fleet: Fleet, boundaries_d: List[int]):
        self.missions = fleet.missions
        self.t = {NONE: fleet.data()}
        for x in boundaries_d:
            tau = x * HOUR_PER_DAY
            self.t[("early", x)] = fleet.data(window=(0.0, tau))
            self.t[("late", x)] = fleet.data(window=(tau, None))
            for m in self.missions:
                ke = self.of(("early", x), m).k.sum() + self.of(("late", x), m).k.sum()
                assert ke == self.of(NONE, m).k.sum(), \
                    f"{m}: the phases at {x} d do not partition the failure record"
        self.edges: Dict[str, float] = {}

    def of(self, key, m: str) -> CalibrationData:
        d = self.t[key]
        return d.subset(np.array(d.mission) == m)

    def fit_without(self, key, excluded, **kw):
        d = self.t[key]
        fit = fit_hyperparameters(d.subset(~np.isin(d.mission, list(excluded))), **kw)
        phase = "whole record" if key == NONE else key[0]
        for axis, v in edge_mass(fit).items():
            name = f"{axis}, {phase}"
            self.edges[name] = max(self.edges.get(name, 0.0), v)
        return fit

    def split_score(self, x: int, m: str, excluded) -> float:
        fe = self.fit_without(("early", x), excluded, c_range=C_RANGE_EARLY)
        fl = self.fit_without(("late", x), excluded)
        return score_fit(self.of(("early", x), m), fe) + score_fit(self.of(("late", x), m), fl)


def fleet_without_launch_day(wb: Path, missions: List[str], scratch: Path) -> Fleet:
    """The fleet with every failure record dated before or on launch day removed."""
    fleet = Fleet(wb, missions, scratch)
    for m in missions:
        ctx = fleet.ctx[m]
        kept = {}
        for key, dates in ctx.failure_map.items():
            late = [d for d in dates
                    if (d - ctx.launch_date).total_seconds() / 3600.0 >= LAUNCH_DAY_H]
            if late:
                kept[key] = late
        fleet.ctx[m] = replace(ctx, failure_map=kept)
    return fleet


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def outer_scores(tab: Tables, boundaries_d: List[int]) -> dict:
    """Every model's score on every held-out mission, fitted on the other seven."""
    S = {"handbook": {}, "sq1": {}}
    S.update({f"split_{x}": {} for x in boundaries_d})
    folds = {}
    for m in tab.missions:
        test = tab.of(NONE, m)
        f_sq1 = tab.fit_without(NONE, [m])
        S["handbook"][m] = score_handbook(test)
        S["sq1"][m] = score_fit(test, f_sq1)
        for x in boundaries_d:
            S[f"split_{x}"][m] = tab.split_score(x, m, [m])
        folds[m] = dict(K=int(test.k.sum()), E=float(test.mu.sum()), N=int(test.n),
                        ratio=float(test.k.sum() / test.mu.sum()),
                        sq1_c=f_sq1.c[0], sq1_a0=f_sq1.a0[0],
                        decomposition=decompose(test, f_sq1.a0[0], f_sq1.c[0]))
    return dict(scores=S, folds=folds)


def selected_split(tab: Tables, outer: dict, boundaries_d: List[int]) -> dict:
    """Choose no split or a boundary inside each fold, by leave-one-mission-out on the
    seven training missions only (every inner fit leaves out the outer mission too),
    and score the held-out mission at that choice."""
    candidates = [NONE] + list(boundaries_d)
    inner = {(m, j): {} for m in tab.missions for j in tab.missions if m != j}
    for m, j in itertools.combinations(tab.missions, 2):
        excl = [m, j]
        f = tab.fit_without(NONE, excl)
        inner[(m, j)][NONE] = score_fit(tab.of(NONE, j), f)
        inner[(j, m)][NONE] = score_fit(tab.of(NONE, m), f)
        for x in boundaries_d:
            fe = tab.fit_without(("early", x), excl, c_range=C_RANGE_EARLY)
            fl = tab.fit_without(("late", x), excl)
            for out, held in ((m, j), (j, m)):
                inner[(out, held)][x] = (score_fit(tab.of(("early", x), held), fe)
                                         + score_fit(tab.of(("late", x), held), fl))
    chosen, score, totals = {}, {}, {}
    for m in tab.missions:
        tot = {x: sum(inner[(m, j)][x] for j in tab.missions if j != m) for x in candidates}
        best = max(candidates, key=lambda x: tot[x])
        chosen[m] = best
        score[m] = (outer["scores"]["sq1"][m] if best == NONE
                    else outer["scores"][f"split_{best}"][m])
        totals[m] = {str(x): float(v - tot[NONE]) for x, v in tot.items()}
    return dict(chosen={m: (x if x == NONE else int(x)) for m, x in chosen.items()},
                scores=score, inner_gain_over_no_split=totals)


def phase_fits(tab: Tables, boundaries_d: List[int]) -> List[dict]:
    """The two phases fitted on all eight missions, for description."""
    rows = []
    for x in boundaries_d:
        fe = tab.fit_without(("early", x), [], c_range=C_RANGE_EARLY)
        fl = tab.fit_without(("late", x), [])
        e, l = tab.t[("early", x)], tab.t[("late", x)]
        rows.append(dict(boundary_d=x,
                         early=dict(K=int(e.k.sum()), E=float(e.mu.sum()),
                                    exposure_Mh=float(e.T.sum() / 1e6),
                                    c=list(fe.c), a0=list(fe.a0),
                                    p_c_above_1=float(fe.W[fe.C > 1.0].sum())),
                         late=dict(K=int(l.k.sum()), E=float(l.mu.sum()),
                                   exposure_Mh=float(l.T.sum() / 1e6),
                                   c=list(fl.c), a0=list(fl.a0),
                                   p_c_below_1=float(fl.W[fl.C < 1.0].sum()))))
    return rows


def summarise(outer: dict, sel: dict, missions: List[str], boundaries_d: List[int]) -> dict:
    S = outer["scores"]
    hb = S["handbook"]
    rel = {name: {m: s[m] - hb[m] for m in missions} for name, s in S.items()}
    rel["selected"] = {m: sel["scores"][m] - hb[m] for m in missions}
    sweep = [dict(boundary_d=x,
                  vs_sq1=compare(S[f"split_{x}"], S["sq1"], missions),
                  vs_handbook=compare(S[f"split_{x}"], hb, missions))
             for x in boundaries_d]
    best = max(sweep, key=lambda r: r["vs_sq1"]["sum"])["boundary_d"]
    for r in sweep:
        r["vs_best"] = compare(S[f"split_{r['boundary_d']}"], S[f"split_{best}"], missions)
    ladder = dict(
        sq1_vs_handbook=compare(S["sq1"], hb, missions),
        reference_split_vs_sq1=compare(S[f"split_{REFERENCE_BOUNDARY_D}"], S["sq1"], missions),
        best_split_vs_sq1=compare(S[f"split_{best}"], S["sq1"], missions),
        selected_vs_sq1=compare(sel["scores"], S["sq1"], missions),
        selected_vs_handbook=compare(sel["scores"], hb, missions),
    )
    return dict(relative_to_handbook=rel, ladder=ladder, sweep=sweep, best_boundary_d=best)


def sweep_without_launch_day(tab: Tables, boundaries_d: List[int]) -> dict:
    out = outer_scores(tab, boundaries_d)
    S, missions = out["scores"], tab.missions
    K = int(tab.t[NONE].k.sum())
    return dict(K=K, sq1_vs_handbook=compare(S["sq1"], S["handbook"], missions),
                sweep=[dict(boundary_d=x, vs_sq1=compare(S[f"split_{x}"], S["sq1"], missions))
                       for x in boundaries_d])


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=0)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for t in leg.get_texts():
        t.set_color(INK)


def fig_validation(p: dict, out_png: Path):
    """SQ2: gain of the SQ1 update over the handbook per held-out mission, split into
    the components without and with a failure."""
    missions = p["missions"]
    folds = p["folds"]
    rel = p["summary"]["relative_to_handbook"]
    fig, a = plt.subplots(figsize=(6.8, 4.2))
    ys = np.arange(len(missions))[::-1]
    h = 0.36
    for y, m in zip(ys, missions):
        dcp = folds[m]["decomposition"]
        a.barh(y + h / 2, dcp["survivors"], height=h, color=BLUE, zorder=3,
               label="components without a failure" if y == ys[0] else None)
        a.barh(y - h / 2, dcp["failed"], height=h, color=MUTED, zorder=3,
               label="components with a failure" if y == ys[0] else None)
        a.plot([rel["sq1"][m]], [y], "D", ms=6, color=INK, mec="white", mew=1.0, zorder=5,
               label="total" if y == ys[0] else None)
    a.axvline(0.0, color=INK2, lw=0.9)
    a.set_yticks(ys)
    a.set_yticklabels([f"{m}   {folds[m]['K']} / {folds[m]['E']:.1f}" for m in missions],
                      fontsize=9, color=INK)
    a.set_xlabel("gain in log score over the handbook", color=INK2)
    a.set_ylabel("mission left out, failures / expected", color=INK2)
    _style(a)
    _legend(a, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)


def fig_boundary(p: dict, out_png: Path):
    """SQ3: (a) the bias of each phase against the boundary, fitted on all missions;
    (b) the cross-validated gain of the split over the single-phase SQ1 update."""
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.0), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ph = p["phase_fits"]
    x = np.array([r["boundary_d"] for r in ph], dtype=float)
    a = ax[0]
    for key, col, lab in (("early", INK2, "early phase"), ("late", BLUE, "routine operations")):
        med = np.array([r[key]["c"][0] for r in ph])
        lo = np.array([r[key]["c"][1] for r in ph])
        hi = np.array([r[key]["c"][2] for r in ph])
        a.fill_between(x, lo, hi, color=col, alpha=0.14, lw=0)
        a.plot(x, med, "-o", color=col, lw=2, ms=5, mec="white", mew=1.2, label=lab)
    a.axhline(1.0, color=INK2, lw=0.9, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(x)
    a.set_xticklabels([f"{v:.0f}" for v in x], fontsize=8)
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    yt = [0.3, 0.5, 1, 2, 5, 10, 20, 30]
    a.set_yticks(yt)
    a.set_yticklabels([f"{v:g}" for v in yt])
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel("length of the early phase, days", color=INK2)
    a.set_ylabel("handbook bias of the phase", color=INK2)
    _style(a, grid_axis="y")
    _legend(a, loc="upper right")

    a = ax[1]
    sweep = p["summary"]["sweep"]
    missions = p["missions"]
    mean = np.array([r["vs_sq1"]["mean"] for r in sweep])
    se = np.array([r["vs_sq1"]["se"] for r in sweep])
    for i, m in enumerate(missions):
        a.plot(x, [r["vs_sq1"]["per_mission"][m] for r in sweep], color=MUTED, lw=0.9,
               alpha=0.7, zorder=2, label="one held-out mission" if i == 0 else None)
    a.fill_between(x, mean - 2 * se, mean + 2 * se, color=BLUE, alpha=0.14, lw=0, zorder=3,
                   label="±2 standard errors")
    a.plot(x, mean, "-o", color=BLUE, lw=2, ms=6, mec="white", mew=1.4, zorder=4,
           label="mean over missions")
    nl = p["without_launch_day"]["sweep"]
    a.plot(x, [r["vs_sq1"]["mean"] for r in nl], "--", color=INK, lw=1.3, zorder=4,
           label="mean, without launch-day records")
    a.axhline(0.0, color=INK2, lw=0.9)
    a.set_xscale("log")
    a.set_xticks(x)
    a.set_xticklabels([f"{v:.0f}" for v in x], fontsize=8)
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel("length of the early phase, days", color=INK2)
    a.set_ylabel("gain in log score per mission", color=INK2)
    _style(a, grid_axis="y")
    _legend(a, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def _line(name: str, c: dict) -> str:
    need = f"{c['missions_needed']:.0f}" if c["missions_needed"] else "-"
    return (f"  {name:<34} sum {c['sum']:+7.2f}  per mission {c['mean']:+.2f} +/- {c['se']:.2f}"
            f"  ({c['z']:+.1f} se)  won {c['wins']}/{c['n']}  missions for 2 se {need}")


def report(p: dict) -> None:
    s = p["summary"]
    print("\n" + "=" * 100)
    print("VALIDATION - leave-one-mission-out log score (higher is better)")
    print("=" * 100)
    print("  held-out    K      E  ratio  c (others)  a0     gain: survivors  failed  total  exact")
    for m, f in p["folds"].items():
        d = f["decomposition"]
        print(f"    {m:<6} {f['K']:>3} {f['E']:6.2f} {f['ratio']:6.2f}   {f['sq1_c']:.3f}   "
              f"{f['sq1_a0']:.3f}   {d['survivors']:+8.2f} {d['failed']:+8.2f} {d['total']:+8.2f} "
              f"{s['relative_to_handbook']['sq1'][m]:+7.2f}")
    L = s["ladder"]
    print(_line("SQ1 update vs handbook", L["sq1_vs_handbook"]))
    print(_line(f"split {REFERENCE_BOUNDARY_D} d vs SQ1 update", L["reference_split_vs_sq1"]))
    print(_line(f"split {s['best_boundary_d']} d (best) vs SQ1", L["best_split_vs_sq1"]))
    print(_line("split, selected vs SQ1 update", L["selected_vs_sq1"]))
    print(_line("split, selected vs handbook", L["selected_vs_handbook"]))
    print("  boundary chosen in each fold: "
          + ", ".join(f"{m} {x}" for m, x in p["selection"]["chosen"].items()))
    print("  sweep (split vs SQ1 update; vs best; without launch-day records):")
    nl = {r["boundary_d"]: r["vs_sq1"] for r in p["without_launch_day"]["sweep"]}
    for r in s["sweep"]:
        a, b, c = r["vs_sq1"], r["vs_best"], nl[r["boundary_d"]]
        print(f"    {r['boundary_d']:>4} d  sum {a['sum']:+7.2f}  per mission {a['mean']:+.2f} +/- "
              f"{a['se']:.2f}  won {a['wins']}/8   vs best {b['mean']:+.2f} +/- {b['se']:.2f}"
              f"   no launch day {c['mean']:+.2f} +/- {c['se']:.2f} won {c['wins']}/8")
    nlh = p["without_launch_day"]["sq1_vs_handbook"]
    print(f"  without launch-day records (K={p['without_launch_day']['K']}): "
          f"SQ1 vs handbook {nlh['mean']:+.2f} +/- {nlh['se']:.2f}")
    print("  phases fitted on all missions:")
    for r in p["phase_fits"]:
        e, l = r["early"], r["late"]
        print(f"    {r['boundary_d']:>4} d  early K={e['K']:>2} E={e['E']:5.2f} c={e['c'][0]:6.2f} "
              f"[{e['c'][1]:.2f}, {e['c'][2]:.2f}] a0={e['a0'][0]:.2f}   late K={l['K']:>2} "
              f"E={l['E']:6.2f} c={l['c'][0]:.3f} [{l['c'][1]:.3f}, {l['c'][2]:.3f}] "
              f"a0={l['a0'][0]:.3f}")
    print(f"  largest posterior mass on a grid edge: {p['grid_edge_mass']}")
    print("=" * 100)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="folder holding the workbook (default: data/)")
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--out-dir", default=None, help="default: results/")
    args = ap.parse_args(argv)

    wb = pipeline.find_workbook(args.dir) if not args.workbook else Path(args.workbook)
    missions = pipeline.list_missions(wb)
    out_dir = Path(args.out_dir) if args.out_dir else pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="val_") as tmp:
        (Path(tmp) / "all").mkdir()
        tab = Tables(Fleet(wb, missions, Path(tmp) / "all"), BOUNDARIES_D)
        full_fit = tab.fit_without(NONE, [])
        outer = outer_scores(tab, BOUNDARIES_D)
        sel = selected_split(tab, outer, BOUNDARIES_D)
        (Path(tmp) / "nl").mkdir()
        tab_nl = Tables(fleet_without_launch_day(wb, missions, Path(tmp) / "nl"), BOUNDARIES_D)
        payload = dict(
            workbook=wb.name, missions=missions,
            settings=dict(seed=SEED, score="log predictive density of held-out failure times",
                          scheme="leave one mission out", boundaries_d=BOUNDARIES_D,
                          reference_boundary_d=REFERENCE_BOUNDARY_D,
                          c_range=list(C_RANGE), c_range_early=list(C_RANGE_EARLY),
                          launch_day_h=LAUNCH_DAY_H),
            all_missions_fit=dict(c=list(full_fit.c), a0=list(full_fit.a0)),
            folds=outer["folds"],
            scores=outer["scores"],
            selection=sel,
            summary=summarise(outer, sel, missions, BOUNDARIES_D),
            phase_fits=phase_fits(tab, BOUNDARIES_D),
            without_launch_day=sweep_without_launch_day(tab_nl, BOUNDARIES_D),
            grid_edge_mass=tab.edges,
        )
    report(payload)
    for axis, v in payload["grid_edge_mass"].items():
        if v > EDGE_WARN:
            print(f"WARNING: up to {v:.1e} of a fit's posterior sits on the {axis} grid edge")

    (out_dir / "sq2_validation.json").write_text(json.dumps(payload, indent=1, default=float))
    fig_validation(payload, out_dir / "sq2_validation.png")
    fig_boundary(payload, out_dir / "sq3_boundary.png")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
