"""Appendix: robustness of the thesis conclusions to the assumptions the other analyses leave open.

    python appendix_robustness.py    # ~15 min; needs results/sq2_validation.json and results/sq3_timing.json

writes results/appendix_robustness.json and the figure appendix_robustness.png.

Four checks, each aimed at one conclusion and one weak point of the existing analysis:

1. Exact tests on the leave-one-mission-out gains (no refitting). The thesis judges a gain by
   mean / standard error over eight missions, read against two. With eight heavy-tailed
   per-mission gains that normal reading is approximate, so each headline comparison is also
   given an exact one-sided sign-flip test (2^8 assignments), a Wilcoxon signed-rank test and
   the range of mean / se with each mission dropped in turn. Read from results/sq2_validation.json.

2. Non-uniform exposure errors. The existing sensitivity suite scales every exposure by the
   same +/-20%, which a fleet-wide bias largely absorbs. Errors in reconstructed operating hours
   (duty cycles, standby status, unit counts) are component- or mission-specific, so each
   component's exposure is multiplied by an independent lognormal factor (median one), or each
   mission's, and the whole-record and routine-operations fits are repeated.

3. Clustered failures. The model treats every failure as independent evidence. As a bound on
   dependence, failures of one mission dated within CLUSTER_DAYS of each other (chained) are
   collapsed into their earliest member, and the fits, the trend test and the cross-validation
   are repeated.

4. Between-mission heterogeneity. Leave-one-mission-out validation assumes missions are
   exchangeable. A mixed predictive check draws (alpha_0, c) from the fleet posterior, fresh
   component rates from the population prior and each mission's failure count, and compares
   each mission's observed count with that range.

Seed 11 throughout; every fit is the unchanged calibration of calibration.py.
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
from scipy import stats

import pipeline
from calibration import CalibrationData, fit_hyperparameters
from sq3_failure_timing import (BLUE, HOUR_PER_DAY, INK, INK2, MUTED, REFERENCE_BOUNDARY_D,
                                Fleet, _style, trend_test)
import sq2_validation as val

SEED = 11
REPS = 100
EXPOSURE_SCENARIOS = [("component", 0.3), ("component", 0.7), ("mission", 0.3)]
CLUSTER_DAYS = 60.0
PPC_DRAWS = 20000
TAU = REFERENCE_BOUNDARY_D * HOUR_PER_DAY


def summary(fit) -> dict:
    return dict(c=[float(v) for v in fit.c], a0=[float(v) for v in fit.a0],
                p_below_1=float(fit.W[fit.C < 1.0].sum()))


# ── 1. exact tests on the cross-validation gains ─────────────────────────────
def gain_tests(g: Dict[str, float]) -> dict:
    x = np.array(list(g.values()), dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    p_flip = float(((signs * np.abs(x)).mean(axis=1) >= x.mean() - 1e-12).mean())
    drop = [np.delete(x, i).mean() / (np.delete(x, i).std(ddof=1) / np.sqrt(n - 1)) for i in range(n)]
    return dict(mean=float(x.mean()), se=float(se), z=float(x.mean() / se), wins=int((x > 0).sum()),
                p_signflip=p_flip,
                p_t=float(stats.ttest_1samp(x, 0.0, alternative="greater").pvalue),
                p_wilcoxon=float(stats.wilcoxon(x, alternative="greater").pvalue),
                z_drop_one=[float(min(drop)), float(max(drop))],
                weakest_drop=list(g)[int(np.argmin(drop))])


def lomo_tests(V: dict) -> dict:
    L, nl = V["summary"]["ladder"], V["without_launch_day"]
    nl30 = next(r for r in nl["sweep"] if r["boundary_d"] == 30)["vs_sq1"]["per_mission"]
    nl_hb = nl["sq1_vs_handbook"]["per_mission"]
    cases = {
        "update vs handbook": L["sq1_vs_handbook"]["per_mission"],
        "split 30 d vs single phase": L["best_split_vs_sq1"]["per_mission"],
        "split, in-fold boundary, vs single phase": L["selected_vs_sq1"]["per_mission"],
        "split, in-fold boundary, vs handbook": L["selected_vs_handbook"]["per_mission"],
        "no launch-day records: update vs handbook": nl_hb,
        "no launch-day records: split 30 d vs single phase": nl30,
        "no launch-day records: split 30 d vs handbook": {m: nl_hb[m] + nl30[m] for m in nl_hb},
    }
    return {k: gain_tests(v) for k, v in cases.items()}


# ── 2. exposure errors ───────────────────────────────────────────────────────
def perturb(d: CalibrationData, factor: Dict[tuple, float]) -> CalibrationData:
    f = np.array([factor[(m, l)] for m, l in zip(d.mission, d.label)])
    return CalibrationData(label=d.label, lam_prior=d.lam_prior, T=d.T * f, k=d.k, mission=d.mission)


def exposure_errors(full: CalibrationData, late: CalibrationData, rng) -> dict:
    keys = list(zip(full.mission, full.label))
    missions = sorted(set(full.mission))
    out = {}
    for level, sd in EXPOSURE_SCENARIOS:
        rows = []
        for _ in range(REPS):
            if level == "component":
                factor = dict(zip(keys, np.exp(sd * rng.standard_normal(len(keys)))))
            else:
                fm = dict(zip(missions, np.exp(sd * rng.standard_normal(len(missions)))))
                factor = {k: fm[k[0]] for k in keys}
            rows.append(dict(whole=summary(fit_hyperparameters(perturb(full, factor))),
                             routine=summary(fit_hyperparameters(perturb(late, factor)))))
        name = f"{level}, sd(log) {sd}"
        out[name] = dict(level=level, sd_log=sd, reps=rows, digest={
            w: dict(median_c=np.percentile([r[w]["c"][0] for r in rows], [5, 50, 95]).tolist(),
                    upper_c=np.percentile([r[w]["c"][2] for r in rows], [5, 50, 95]).tolist(),
                    share_median_below_1=float(np.mean([r[w]["c"][0] < 1 for r in rows])),
                    share_interval_below_1=float(np.mean([r[w]["c"][2] < 1 for r in rows])),
                    p_below_1=np.percentile([r[w]["p_below_1"] for r in rows], [5, 50, 95]).tolist())
            for w in ("whole", "routine")})
        print(f"[exposure] {name}: routine interval below one in "
              f"{out[name]['digest']['routine']['share_interval_below_1']:.0%}")
    return out


# ── 3. clustered failures ────────────────────────────────────────────────────
def clusters(failures: List[dict]) -> List[List[dict]]:
    """Chain each mission's failures that lie within CLUSTER_DAYS of the previous one."""
    out = []
    for m in sorted({f["mission"] for f in failures}):
        fs = sorted((f for f in failures if f["mission"] == m), key=lambda f: f["t_raw_h"])
        group = [fs[0]]
        for f in fs[1:]:
            if f["t_raw_h"] - group[-1]["t_raw_h"] <= CLUSTER_DAYS * HOUR_PER_DAY:
                group.append(f)
            else:
                out.append(group); group = [f]
        out.append(group)
    return [g for g in out if len(g) > 1]


def collapse(fleet: Fleet, drop: List[dict]) -> Fleet:
    for f in drop:
        ctx = fleet.ctx[f["mission"]]
        key = (f["subsystem"], f["component"])
        dates = ctx.failure_map[key]
        hours = [(d - ctx.launch_date).total_seconds() / 3600.0 for d in dates]
        j = int(np.argmin([abs(h - f["t_raw_h"]) for h in hours]))
        assert abs(hours[j] - f["t_raw_h"]) < 1.0, f
        kept = dates[:j] + dates[j + 1:]
        fm = dict(ctx.failure_map)
        if kept:
            fm[key] = kept
        else:
            del fm[key]
        fleet.ctx[f["mission"]] = replace(ctx, failure_map=fm)
    return fleet


def clustered(wb: Path, missions: List[str], failures: List[dict], scratch: Path) -> dict:
    groups = clusters(failures)
    drop = [f for g in groups for f in g[1:]]
    kept = [f for f in failures if f not in drop]
    fleet = collapse(Fleet(wb, missions, scratch), drop)
    full, late = fleet.data(), fleet.data(window=(TAU, None))
    assert int(full.k.sum()) == len(kept)
    tab = val.Tables(fleet, [30])
    S = val.outer_scores(tab, [30])["scores"]
    g = lambda a, b: {m: S[a][m] - S[b][m] for m in missions}
    early = [f for f in kept if f["t_h"] <= TAU]
    return dict(
        rule=f"failures of one mission within {CLUSTER_DAYS:.0f} days of the previous one, chained, "
             "collapsed into the earliest",
        clusters=[[f"{f['mission']} {f['component']} d{f['t_raw_h'] / 24:.0f}" for f in g] for g in groups],
        K=len(kept), K_first_90_d=len(early),
        whole=summary(fit_hyperparameters(full)), routine=summary(fit_hyperparameters(late)),
        trend_all=trend_test([f["u"] for f in kept]),
        cv={"update vs handbook": gain_tests(g("sq1", "handbook")),
            "split 30 d vs single phase": gain_tests(g("split_30", "sq1")),
            "split 30 d vs handbook": gain_tests(g("split_30", "handbook"))})


# ── 4. between-mission heterogeneity ────────────────────────────────────────
def mission_check(d: CalibrationData, rng) -> dict:
    fit = fit_hyperparameters(d)
    w = fit.W.ravel() / fit.W.sum()
    idx = rng.choice(w.size, size=PPC_DRAWS, p=w)
    a0, c = fit.A0.ravel()[idx], fit.C.ravel()[idx]
    mission = np.array(d.mission)
    out = {}
    for m in sorted(set(d.mission)):
        sel = mission == m
        lam_hb, T = d.lam_prior[sel], d.T[sel]
        lam = rng.gamma(a0[:, None], (c[:, None] * lam_hb[None, :]) / a0[:, None])
        K = rng.poisson((lam * T[None, :]).sum(axis=1))
        k = int(d.k[sel].sum())
        out[m] = dict(K=k, E=float(d.mu[sel].sum()), replicated=np.percentile(K, [5, 50, 95]).tolist(),
                      p_at_most=float((K <= k).mean()), p_at_least=float((K >= k).mean()))
    return out


# ── figure ───────────────────────────────────────────────────────────────────
def figure(res: dict, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    names = list(res["exposure"])
    ys = np.arange(len(names) + 1)[::-1]
    base = res["baseline"]["routine"]["c"]
    a.hlines(ys[0], base[1], base[2], color=INK2, lw=2)
    a.plot(base[0], ys[0], "o", ms=7, color=INK2, mec="white", mew=1.4, zorder=5, label="as reconstructed: median and 90% interval")
    for y, n in zip(ys[1:], names):
        dg = res["exposure"][n]["digest"]["routine"]
        a.hlines(y, dg["upper_c"][0], dg["upper_c"][2], color=BLUE, lw=6, alpha=0.35,
                 label="upper 90% limit, 5–95% of replicates" if y == ys[1] else None)
        a.hlines(y, dg["median_c"][0], dg["median_c"][2], color=BLUE, lw=2,
                 label="median, 5–95% of replicates" if y == ys[1] else None)
        a.plot(dg["median_c"][1], y, "o", ms=6, color=BLUE, mec="white", mew=1.2, zorder=5)
    a.axvline(1.0, color=INK2, lw=0.9)
    a.set_yticks(ys)
    a.set_yticklabels(["no error"] + [n.replace("sd(log)", "error sd") for n in names], fontsize=9)
    a.set_xscale("log"); a.set_xlim(0.3, 2.0)
    a.set_xticks([0.3, 0.5, 1, 2]); a.set_xticklabels(["0.3", "0.5", "1", "2"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel(f"routine-operations bias $c^r$ (after {REFERENCE_BOUNDARY_D} days)", color=INK2)
    _style(a, grid_axis="x")
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    b = ax[1]
    pc = res["mission_check"]["whole"]
    ms = list(pc)
    ys = np.arange(len(ms))[::-1]
    for y, m in zip(ys, ms):
        lo, med, hi = pc[m]["replicated"]
        b.hlines(y, lo, hi, color=MUTED, lw=6, alpha=0.45,
                 label="replicated, 90% range" if y == ys[0] else None)
        b.plot(med, y, "|", ms=12, mew=1.6, color=INK2)
        b.plot(pc[m]["K"], y, "o", ms=7, color=BLUE, mec="white", mew=1.4, zorder=5,
               label="observed" if y == ys[0] else None)
    b.set_yticks(ys); b.set_yticklabels(ms, fontsize=9)
    b.set_xlabel("failures of the mission, whole record", color=INK2)
    _style(b, grid_axis="x")
    b.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for x in ax:
        for t in x.get_legend().get_texts():
            t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from the stored JSON")
    args = ap.parse_args(argv)
    wb = pipeline.find_workbook(None)
    missions = pipeline.list_missions(wb)
    out_dir = Path(args.out_dir) if args.out_dir else pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        figure(json.loads((out_dir / "appendix_robustness.json").read_text(encoding="utf-8")),
               out_dir / "appendix_robustness.png")
        return
    rng = np.random.default_rng(SEED)

    V = json.loads((out_dir / "sq2_validation.json").read_text(encoding="utf-8"))
    failures = json.loads((out_dir / "sq3_timing.json").read_text(encoding="utf-8"))["failures"]
    with tempfile.TemporaryDirectory(prefix="rob_") as tmp:
        (Path(tmp) / "a").mkdir(); (Path(tmp) / "b").mkdir()
        fleet = Fleet(wb, missions, Path(tmp) / "a")
        full, late = fleet.data(), fleet.data(window=(TAU, None))
        res = dict(
            settings=dict(seed=SEED, reps=REPS, exposure_scenarios=EXPOSURE_SCENARIOS,
                          cluster_days=CLUSTER_DAYS, ppc_draws=PPC_DRAWS,
                          routine_boundary_d=REFERENCE_BOUNDARY_D),
            baseline=dict(whole=summary(fit_hyperparameters(full)),
                          routine=summary(fit_hyperparameters(late))),
            lomo_tests=lomo_tests(V),
            mission_check=dict(whole=mission_check(full, rng), routine=mission_check(late, rng)),
            clustered=clustered(wb, missions, failures, Path(tmp) / "b"),
            exposure=exposure_errors(full, late, rng),
        )
    assert abs(res["baseline"]["whole"]["c"][0] - 0.9635) < 1e-3, "baseline must reproduce the thesis"
    (out_dir / "appendix_robustness.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res, out_dir / "appendix_robustness.png")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
