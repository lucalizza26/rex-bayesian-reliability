"""Appendix (exploratory): does handbook accuracy differ between classes of component?

CLASSIFICATION  data/component_classes.csv, maintained by hand. Every subsystem of every mission gets a
                functional class: AOCS, data handling, communications, power, propulsion,
                thermal or payload. Where a declared subsystem mixes functions, individual
                components are reassigned by function; a blank class leaves a component
                unclassified (harness, a boom hinge, a fluidic system whose function the
                record does not state), and unclassified components stay out of this analysis.

EVIDENCE        the whole record: every failure of every mission against the failures its
                handbook rates predict over the whole exposure. The question is how many
                failures each class recorded against its expectation, not when they occurred
                (that is SQ3), so no part of a mission is set aside. The failures within the
                first 90 days are counted per class for reference only.

MODELS          the SQ1 calibration with the bias allowed to differ by class. alpha_0 is
                shared by every class in every model. All integrals are deterministic grids
                with Gauss-Hermite quadrature; no MCMC.
                  pooled        one bias c for every class (the SQ1 model)
                  partial       log c_l = log c + omega * eps_l, eps_l ~ N(0, 1),
                                omega ~ HalfNormal(1): classes drawn towards a common bias
                  independent   log c_l ~ N(0, 1) independently for every class

FRAMING         the per-class CHECK against the single SQ1 correction, read together with
                each class's own bias (independent model) in one table, `class_table`; the
                DATA REQUIREMENT (`missions_needed`) says how many missions each grouping
                would need to beat the single correction. Partial pooling and the grouping
                comparison are supporting material.

CHECK           class failure totals replicated from the pooled posterior against the
                observed totals, with the class's failures in the first 90 days alongside.

COMPARISON      the marginal likelihood (evidence) Z of every model, and the leave-one-
                mission-out predictive density  log p(D_m | D_-m) = log Z(D) - log Z(D_-m),
                summed over missions, for five groupings of the same components: pooled,
                platform / payload, function by declared subsystem, function at component
                level (the classification above) and mission.

DATA NEEDED     a grouping beats the single correction by two standard errors on leave-one-
                mission-out once the number of missions n satisfies n g > 2 s sqrt(n), i.e.
                n > 4 s^2 / g^2, with g and s the mean and standard deviation of the per-
                mission predictive gain. Undefined (None) when g <= 0.

    python appendix_classes.py

writes results/appendix_classes.json and three figures: appendix_class_overview.png (check
and own bias per class), appendix_class_estimates.png (partial pooling) and
appendix_grouping_comparison.png.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma as gamma_dist, norm

import pipeline
from sq3_failure_timing import (Fleet, HOUR_PER_DAY, REFERENCE_BOUNDARY_D,
                                SEED, INK, INK2, MUTED, GRID, AXIS, BLUE, _style)

CLASS_FILE = pipeline.DATA_DIR / "component_classes.csv"
CLASS_ORDER = ["Data handling", "Power", "Payload", "AOCS", "Communications", "Propulsion",
               "Thermal"]
PRIMARY = "function, component level"
SCHEMES = ["pooled", "platform / payload", "function, declared subsystem", PRIMARY, "mission"]

# grids: shared strength, common bias, class spread, class bias (fine), class deviation
LA = np.linspace(np.log(0.05), np.log(20.0), 50)
LC = np.linspace(np.log(0.05), np.log(6.0), 60)
OM = np.linspace(0.02, 2.5, 45)
Y = np.linspace(-20.0, 20.0, 801)
YP = (Y > -7.0) & (Y < 5.0)                  # support used for class posteriors
XQ, WQ = np.polynomial.hermite_e.hermegauss(40)
LWQ = np.log(WQ / np.sqrt(2.0 * np.pi))
OMEGA_SCALE = 1.0
OMEGA_SENSITIVITY = [0.5, 2.0]
ALPHA0_FIXED = 1.0
N_CHECK = 20000


def _mass(logdens: np.ndarray) -> np.ndarray:
    """Log prior masses on a grid, renormalized over the grid."""
    return logdens - logsumexp(logdens)


LP_A = _mass(norm.logpdf(LA))
LP_C = _mass(norm.logpdf(LC))
LP_Y = _mass(norm.logpdf(Y))


def lp_omega(scale: float) -> np.ndarray:
    return _mass(norm.logpdf(OM, 0.0, scale))


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
def load_classes(path: Path) -> Tuple[Dict, Dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("mission", "subsystem", "component", "class"):
        if col not in df.columns:
            raise SystemExit(f"{path.name}: missing column '{col}'")
    bad = sorted(set(df["class"]) - set(CLASS_ORDER) - {""})
    if bad:
        raise SystemExit(f"{path.name}: unknown classes {bad}")
    default, override = {}, {}
    for _, r in df.iterrows():
        key = (r.mission.strip(), r.subsystem.strip())
        if r.component.strip():
            k3 = key + (r.component.strip(),)
            if k3 in override:
                raise SystemExit(f"{path.name}: duplicate row for {k3}")
            override[k3] = r["class"].strip()
        else:
            if key in default:
                raise SystemExit(f"{path.name}: duplicate row for {key}")
            default[key] = r["class"].strip()
    return default, override


def classify(data, default: Dict, override: Dict) -> pd.DataFrame:
    """One row per fitted component: functional class, declared-subsystem class, evidence."""
    rows, present = [], set()
    for m, label, k, mu, T in zip(data.mission, data.label, data.k, data.mu, data.T):
        sub, comp = label.split("/", 1)
        present.add((m, sub))
        if (m, sub) not in default:
            raise SystemExit(f"{CLASS_FILE.name}: no class row for subsystem {(m, sub)}")
        rows.append(dict(mission=m, subsystem=sub, component=comp, k=int(k), mu=float(mu),
                         T=float(T), declared=default[(m, sub)],
                         func=override.get((m, sub, comp), default[(m, sub)])))
    df = pd.DataFrame(rows)
    stale = [k for k in default if k not in present]
    stale += [k for k in override
              if not ((df.mission == k[0]) & (df.subsystem == k[1]) & (df.component == k[2])).any()]
    if stale:
        raise SystemExit(f"{CLASS_FILE.name}: rows that match no fitted component: {stale}")
    return df


def group_of(row, scheme: str) -> str:
    if scheme == "pooled":
        return "fleet"
    if scheme == "platform / payload":
        return "Payload" if row.func == "Payload" else "Platform"
    if scheme == "function, declared subsystem":
        return row.declared
    if scheme == PRIMARY:
        return row.func
    if scheme == "mission":
        return row.mission
    raise ValueError(scheme)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic class models
# ─────────────────────────────────────────────────────────────────────────────
def ll_table(k: np.ndarray, mu: np.ndarray, la: np.ndarray = LA) -> np.ndarray:
    """sum_i log NegBin(k_i | alpha_0, c' mu_i) on (la x Y)."""
    out = np.zeros((la.size, Y.size))
    if k.size == 0:
        return out
    kk = k[None, :].astype(float)
    cm = np.exp(Y)[:, None] * mu[None, :]
    for ia, a0 in enumerate(np.exp(la)):
        lp = (gammaln(kk + a0) - gammaln(a0) - gammaln(kk + 1.0)
              + a0 * (np.log(a0) - np.log(a0 + cm)) + kk * (np.log(cm) - np.log(a0 + cm)))
        out[ia] = lp.sum(axis=1)
    return out


def at(tab: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (A, Y) table at log-bias values y of any shape."""
    pos = (y - Y[0]) / (Y[1] - Y[0])
    i0 = np.clip(np.floor(pos).astype(int), 0, Y.size - 2)
    f = np.clip(pos - i0, 0.0, 1.0)
    return tab[:, i0] * (1.0 - f) + tab[:, i0 + 1] * f


def evidences(tabs: List[np.ndarray], omega_scale: float = OMEGA_SCALE,
              lp_a: np.ndarray = LP_A) -> dict:
    """Log evidence of the pooled, partial-pooling and independent models."""
    tot = sum(at(t, LC) for t in tabs)                                   # (A, B)
    post_p = tot + lp_a[:, None] + LP_C[None, :]
    logZ_P = logsumexp(post_p)
    y = LC[:, None, None] + OM[None, :, None] * XQ[None, None, :]        # (B, W, Q)
    logI = [logsumexp(at(t, y) + LWQ, axis=3) for t in tabs]             # (A, B, W) each
    acc = sum(logI)
    base = lp_a[:, None, None] + LP_C[None, :, None] + lp_omega(omega_scale)[None, None, :]
    logZ_H = logsumexp(acc + base)
    ind = sum(logsumexp(t + LP_Y[None, :], axis=1) for t in tabs)        # (A,)
    logZ_I = logsumexp(ind + lp_a)
    return dict(P=float(logZ_P), H=float(logZ_H), I=float(logZ_I),
                post_p=post_p - logZ_P, logI=logI, base=base, acc=acc)


def wq(values: np.ndarray, logw: np.ndarray, qs=(0.5, 0.05, 0.95)) -> List[float]:
    w = np.exp(logw - logw.max()).ravel()
    v = values.ravel()
    o = np.argsort(v)
    cdf = np.cumsum(w[o]) / w.sum()
    return [float(np.interp(q, cdf, v[o])) for q in qs]


def omega_summary(ev: dict) -> List[float]:
    lw = logsumexp(ev["acc"] + ev["base"], axis=(0, 1))
    return wq(OM, lw)


def class_posteriors(tabs: List[np.ndarray], ev: dict, lp_a: np.ndarray = LP_A) -> Tuple[list, list]:
    """Posterior of each class bias under the partial and the independent models:
    [median, q05, q95, P(c_l < 1)]."""
    yp = Y[YP]
    nrm = norm.logpdf(yp[None, None, :], LC[:, None, None], OM[None, :, None])   # (B, W, Y)
    total = ev["acc"] + ev["base"]
    ind_parts = [logsumexp(t + LP_Y[None, :], axis=1) for t in tabs]
    part, indep = [], []
    for j, t in enumerate(tabs):
        rest = total - ev["logI"][j]                                     # (A, B, W)
        dens = np.full(yp.size, -np.inf)
        for ia in range(t.shape[0]):
            dens = np.logaddexp(dens, logsumexp(rest[ia][:, :, None] + nrm, axis=(0, 1)) + t[ia, YP])
        p = np.exp(dens - dens.max()); p /= p.sum()
        cdf = np.cumsum(p)
        part.append([float(np.exp(np.interp(q, cdf, yp))) for q in (0.5, 0.05, 0.95)]
                    + [float(p[yp < 0].sum())])
        rest_i = sum(ind_parts) - ind_parts[j] + lp_a
        dens_i = logsumexp(rest_i[:, None] + t[:, YP] + LP_Y[None, YP], axis=0)
        p = np.exp(dens_i - dens_i.max()); p /= p.sum()
        cdf = np.cumsum(p)
        indep.append([float(np.exp(np.interp(q, cdf, yp))) for q in (0.5, 0.05, 0.95)]
                     + [float(p[yp < 0].sum())])
    return part, indep


def scheme_tables(df: pd.DataFrame, scheme: str, missions: List[str]):
    g = df.apply(lambda r: group_of(r, scheme), axis=1)
    groups = [c for c in CLASS_ORDER if c in set(g)] + sorted(set(g) - set(CLASS_ORDER))
    tabs = {}
    for grp in groups:
        for m in missions:
            sel = ((g == grp) & (df.mission == m)).values
            if sel.any():
                tabs[(grp, m)] = ll_table(df.k.values[sel], df.mu.values[sel])
    return groups, g, tabs


def combine(tabs: dict, groups: List[str], drop: str = None) -> List[np.ndarray]:
    zero = np.zeros((LA.size, Y.size))
    return [sum((t for (grp, m), t in tabs.items() if grp == gg and m != drop), zero)
            for gg in groups]


def compare(df: pd.DataFrame, scheme: str, missions: List[str], omega_scale: float = OMEGA_SCALE) -> dict:
    groups, g, tabs = scheme_tables(df, scheme, missions)
    full = combine(tabs, groups)
    ev = evidences(full, omega_scale)
    per = {}
    for m in missions:
        ev_m = evidences(combine(tabs, groups, drop=m), omega_scale)
        per[m] = {key: ev[key] - ev_m[key] for key in ("P", "H", "I")}
    out = dict(scheme=scheme, groups=groups, n_groups=len(groups),
               logZ={key: ev[key] for key in ("P", "H", "I")},
               lomo={key: float(sum(per[m][key] for m in missions)) for key in ("P", "H", "I")},
               lomo_per_mission=per, omega=omega_summary(ev))
    for key in ("H", "I"):
        d = np.array([per[m][key] - per[m]["P"] for m in missions])
        out[f"lomo_gain_{key}"] = float(d.sum())
        out[f"lomo_gain_{key}_se"] = float(d.std(ddof=1) * np.sqrt(d.size))
        out[f"missions_needed_{key}"] = missions_needed(d)
    return out, groups, full, ev


def missions_needed(gains: np.ndarray):
    """Missions at which the summed leave-one-mission-out gain reaches two standard errors,
    if every further mission behaved like the ones in the record: n > 4 s^2 / g^2."""
    g = float(gains.mean())
    if g <= 0.0:
        return None
    return float(4.0 * gains.var(ddof=1) / g ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Descriptives and the pooled check
# ─────────────────────────────────────────────────────────────────────────────
def jeffreys(K: int, E: float) -> Tuple[float, float]:
    lo, hi = gamma_dist.ppf([0.05, 0.95], K + 0.5, scale=1.0 / E)
    return float(lo), float(hi)


def describe(df: pd.DataFrame) -> List[dict]:
    rows = []
    for cls in CLASS_ORDER:
        sub = df[df.func == cls]
        K, E = int(sub.k.sum()), float(sub.mu.sum())
        lo, hi = jeffreys(K, E)
        rows.append(dict(cls=cls, types=int(len(sub)), missions=int(sub.mission.nunique()),
                         exposure_Mh=float(sub["T"].sum() / 1e6), E=E, K=K,
                         ratio=K / E, lo=lo, hi=hi))
    return rows


def pooled_check(df: pd.DataFrame, groups: List[str], ev: dict, rng) -> Dict[str, dict]:
    """Class totals replicated from the pooled posterior of (alpha_0, c)."""
    w = np.exp(ev["post_p"]).ravel()
    idx = rng.choice(w.size, size=N_CHECK, p=w / w.sum())
    a0 = np.exp(LA)[idx // LC.size]
    c = np.exp(LC)[idx % LC.size]
    k, mu = df.k.values.astype(float), df.mu.values
    member = {grp: (df.func == grp).values for grp in groups}
    reps = {grp: np.empty(N_CHECK, dtype=int) for grp in groups}
    for s in range(0, N_CHECK, 2000):
        e = min(s + 2000, N_CHECK)
        aa, cc = a0[s:e, None], c[s:e, None]
        lam_T = rng.gamma(aa + k[None, :], 1.0 / (aa / (cc * mu[None, :]) + 1.0))
        rep = rng.poisson(lam_T)
        for grp in groups:
            reps[grp][s:e] = rep[:, member[grp]].sum(axis=1)
    out = {}
    for grp in groups:
        r, obs = reps[grp], int(df.k.values[member[grp]].sum())
        out[grp] = dict(observed=obs, median=float(np.median(r)), lo=float(np.percentile(r, 5)),
                        hi=float(np.percentile(r, 95)), p_at_most=float((r <= obs).mean()),
                        p_at_least=float((r >= obs).mean()))
    return out


MIN_EXPECTED = 1.0      # below this many handbook-expected failures a class cannot inform
LEAN = 0.90             # one-sided posterior probability at which an own bias 'leans'


def class_table(res: dict) -> List[dict]:
    """One row per class: the check against the single correction, the class's early failures,
    and its own bias (independent model), with a reading of the three together.

    check   'inside' / 'above' / 'below' the 90% range a single correction replicates;
            tail_p is the replicated probability of a count at least as far from the median
            in the observed direction.
    own     'no information'  if fewer than MIN_EXPECTED failures were expected,
            'above' / 'below' if the class's own 90% interval excludes 1,
            'leans above' / 'leans below' if P(c_l > 1) or P(c_l < 1) reaches LEAN,
            'consistent'      otherwise."""
    rows = []
    for c in res["classes"]:
        chk = res["pooled_check"][c["cls"]]
        med, lo, hi, p_below = res["class_posteriors"][c["cls"]]["independent"]
        obs = chk["observed"]
        if obs > chk["hi"]:
            check = "above"
        elif obs < chk["lo"]:
            check = "below"
        else:
            check = "inside"
        tail = (chk["p_at_least"] if obs > chk["median"] else
                chk["p_at_most"] if obs < chk["median"] else None)
        if c["E"] < MIN_EXPECTED:
            own = "no information"
        elif hi < 1.0:
            own = "below"
        elif lo > 1.0:
            own = "above"
        elif 1.0 - p_below >= LEAN:
            own = "leans above"
        elif p_below >= LEAN:
            own = "leans below"
        else:
            own = "consistent"
        rows.append(dict(cls=c["cls"], types=c["types"], K=obs, E=c["E"], ratio=c["ratio"],
                         K_before_boundary=c["K_before_boundary"],
                         predicted=[chk["median"], chk["lo"], chk["hi"]], check=check, tail_p=tail,
                         own_bias=[med, lo, hi], p_above=1.0 - p_below, own=own))
    return rows


def analyse(df: pd.DataFrame, missions: List[str], rng) -> dict:
    schemes, primary = [], None
    for scheme in SCHEMES:
        res, groups, full, ev = compare(df, scheme, missions)
        schemes.append(res)
        print(f"[sq3]   {scheme:<30} groups={res['n_groups']}  logZ P={res['logZ']['P']:.2f} "
              f"H={res['logZ']['H']:.2f} I={res['logZ']['I']:.2f}  LOMO gain H={res['lomo_gain_H']:+.2f} "
              f"(se {res['lomo_gain_H_se']:.2f}) I={res['lomo_gain_I']:+.2f} (se {res['lomo_gain_I_se']:.2f})")
        if scheme == PRIMARY:
            primary = (groups, full, ev)
    groups, full, ev = primary
    part, indep = class_posteriors(full, ev)
    fleet_c = wq(np.exp(LC)[None, :] * np.ones((LA.size, 1)), ev["post_p"])
    return dict(
        classes=describe(df),
        schemes=schemes,
        fleet_c=fleet_c,
        class_posteriors={grp: dict(partial=part[j], independent=indep[j]) for j, grp in enumerate(groups)},
        pooled_check=pooled_check(df, groups, ev, rng),
    )


def sensitivity(df: pd.DataFrame, missions: List[str]) -> dict:
    out = {"omega_scale": {}, "alpha0_fixed": {}}
    for scale in OMEGA_SENSITIVITY:
        res, groups, full, ev = compare(df, PRIMARY, missions, omega_scale=scale)
        part, _ = class_posteriors(full, ev)
        out["omega_scale"][str(scale)] = dict(
            logZ_H_minus_P=res["logZ"]["H"] - res["logZ"]["P"], lomo_gain_H=res["lomo_gain_H"],
            lomo_gain_H_se=res["lomo_gain_H_se"], omega=res["omega"],
            partial={grp: part[j] for j, grp in enumerate(groups)})
    # alpha_0 pinned: rebuild the tables on a single strength value
    la = np.array([np.log(ALPHA0_FIXED)])
    lp_a = np.array([0.0])
    gcol = df.apply(lambda r: group_of(r, PRIMARY), axis=1)
    groups = [c for c in CLASS_ORDER if c in set(gcol)]
    tabs = {}
    for grp in groups:
        for m in missions:
            sel = ((gcol == grp) & (df.mission == m)).values
            if sel.any():
                tabs[(grp, m)] = ll_table(df.k.values[sel], df.mu.values[sel], la)
    zero = np.zeros((1, Y.size))

    def comb(drop=None):
        return [sum((t for (gg, m), t in tabs.items() if gg == grp and m != drop), zero) for grp in groups]

    ev = evidences(comb(), lp_a=lp_a)
    gains = []
    for m in missions:
        ev_m = evidences(comb(m), lp_a=lp_a)
        gains.append((ev["H"] - ev_m["H"]) - (ev["P"] - ev_m["P"]))
    part, _ = class_posteriors(comb(), ev, lp_a=lp_a)
    out["alpha0_fixed"] = dict(alpha0=ALPHA0_FIXED, logZ_P=ev["P"], logZ_H=ev["H"],
                               lomo_gain_H=float(np.sum(gains)),
                               lomo_gain_H_se=float(np.std(gains, ddof=1) * np.sqrt(len(gains))),
                               partial={grp: part[j] for j, grp in enumerate(groups)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _log_axis(ax, lo, hi, ticks):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_estimates(res: dict, out_pngs):
    classes = res["classes"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    ys = np.arange(len(classes))[::-1]
    a = ax[0]
    for y, c in zip(ys, classes):
        lo = max(c["lo"], 0.012)
        a.hlines(y, lo, c["hi"], color=MUTED, lw=2)
        if c["K"] > 0:
            a.plot(c["ratio"], y, "o", ms=7, color=MUTED, mec="white", mew=1.6, zorder=5)
        else:
            a.plot(0.012, y, "<", ms=7, color=MUTED, zorder=5)
        a.text(9.5, y, f"{c['K']} / {c['E']:.1f}", va="center", ha="right", fontsize=8.5, color=INK2)
    a.axvline(1.0, color=INK2, lw=0.9)
    _log_axis(a, 0.01, 10, [0.01, 0.03, 0.1, 0.3, 1, 3])
    a.set_yticks(ys)
    a.set_yticklabels([c["cls"] for c in classes], fontsize=9)
    a.set_xlabel("observed ÷ handbook-expected failures", color=INK2)
    _style(a, grid_axis="x")

    a = ax[1]
    f_med, f_lo, f_hi = res["fleet_c"]
    a.axvspan(f_lo, f_hi, color=MUTED, alpha=0.15, lw=0, label="single correction, 90% interval")
    a.axvline(f_med, color=MUTED, lw=1.2)
    for y, c in zip(ys, classes):
        p = res["class_posteriors"][c["cls"]]
        ind = p["independent"]
        a.hlines(y - 0.16, ind[1], ind[2], color=INK2, lw=1.2)
        a.plot(ind[0], y - 0.16, "o", ms=5.5, mfc="white", mec=INK2, mew=1.2, zorder=5,
               label="classes independent" if y == ys[0] else None)
        par = p["partial"]
        a.hlines(y + 0.16, par[1], par[2], color=BLUE, lw=2)
        a.plot(par[0], y + 0.16, "o", ms=7, color=BLUE, mec="white", mew=1.6, zorder=6,
               label="classes partially pooled" if y == ys[0] else None)
    a.axvline(1.0, color=INK2, lw=0.9)
    _log_axis(a, 0.05, 10, [0.1, 0.3, 1, 3, 10])
    a.set_xlabel("handbook bias of the class (median, 90% interval)", color=INK2)
    _style(a, grid_axis="x")
    h, l = a.get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3,
                     bbox_to_anchor=(0.62, 0.0))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


def fig_overview(res: dict, out_pngs):
    """Main-text figure: per class, (a) the observed failures against the range a single
    correction replicates, with the class's failures in the first 90 days, and (b) the class's
    own bias with classes estimated independently, against the single correction."""
    rows = res["class_table"]
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
    ys = np.arange(len(rows))[::-1]
    a = ax[0]
    xmax = max(max(r["predicted"][2], r["K"]) for r in rows) + 4.5
    for y, r in zip(ys, rows):
        med, lo, hi = r["predicted"]
        a.hlines(y, lo, hi, color=MUTED, lw=6, alpha=0.45,
                 label="replicated by a single correction, 90% range" if y == ys[0] else None)
        a.plot(med, y, "|", ms=12, mew=1.6, color=INK2)
        a.plot(r["K"], y, "o", ms=7, color=BLUE, mec="white", mew=1.6, zorder=5,
               label="observed" if y == ys[0] else None)
        a.text(xmax - 0.2, y, f"{r['K_before_boundary']} of {r['K']}", va="center", ha="right",
               fontsize=8.5, color=INK2)
    a.text(xmax - 0.2, ys[0] + 0.62, f"first {REFERENCE_BOUNDARY_D} d", va="bottom", ha="right",
           fontsize=8, color=MUTED)
    a.set_yticks(ys)
    a.set_yticklabels([r["cls"] for r in rows], fontsize=9)
    a.set_xlim(-0.5, xmax)
    a.set_ylim(ys[-1] - 0.6, ys[0] + 1.0)
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a.set_xlabel("failures in the class", color=INK2)
    _style(a, grid_axis="x")

    b = ax[1]
    f_med, f_lo, f_hi = res["fleet_c"]
    b.axvspan(f_lo, f_hi, color=MUTED, alpha=0.15, lw=0, label="single correction, 90% interval")
    b.axvline(f_med, color=MUTED, lw=1.2)
    for y, r in zip(ys, rows):
        med, lo, hi = r["own_bias"]
        b.hlines(y, lo, hi, color=BLUE, lw=2)
        b.plot(med, y, "o", ms=7, color=BLUE, mec="white", mew=1.6, zorder=6,
               label="own class bias, classes independent" if y == ys[0] else None)
    b.axvline(1.0, color=INK2, lw=0.9)
    _log_axis(b, 0.05, 10, [0.1, 0.3, 1, 3, 10])
    b.set_xlabel("handbook bias of the class (median, 90% interval)", color=INK2)
    b.yaxis.set_tick_params(labelleft=True)     # class names on both panels
    _style(b, grid_axis="x")

    # one legend under each panel, so each panel stands alone as a subfigure
    for x in ax:
        leg = x.legend(frameon=False, fontsize=8.3, loc="upper center", ncol=1,
                       bbox_to_anchor=(0.5, -0.17))
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


def fig_groupings(res: dict, out_pngs):
    schemes = [s for s in SCHEMES if s != "pooled"]
    rows = {s["scheme"]: s for s in res["schemes"]}
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ys = np.arange(len(schemes))[::-1]
    for key, off, col, lw, lab, marker in (
            ("I", -0.16, INK2, 1.2, "groups independent", dict(ms=5.5, mfc="white", mec=INK2, mew=1.2)),
            ("H", 0.16, BLUE, 2, "groups partially pooled", dict(ms=7, color=BLUE, mec="white", mew=1.6))):
        for y, sc in zip(ys, schemes):
            g, se = rows[sc][f"lomo_gain_{key}"], rows[sc][f"lomo_gain_{key}_se"]
            ax.hlines(y + off, g - se, g + se, color=col, lw=lw)
            ax.plot(g, y + off, "o", zorder=5, label=lab if y == ys[0] else None, **marker)
    ax.axvline(0.0, color=INK2, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(schemes, fontsize=9)
    ax.set_xlabel("predictive gain over a single correction, missions left out in turn [log units]",
                  color=INK2, fontsize=9)
    _style(ax, grid_axis="x")
    leg = ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    for p in out_pngs:
        fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
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
    default, override = load_classes(CLASS_FILE)
    rng = np.random.default_rng(SEED)

    with tempfile.TemporaryDirectory(prefix="sq3_") as tmp:
        fleet = Fleet(wb, missions, Path(tmp))
        df_all = classify(fleet.data(), default, override)
        df = df_all[df_all.func != ""].reset_index(drop=True)
        coverage = dict(
            types=int(len(df_all)), classified=int(len(df)),
            E=float(df_all.mu.sum()), E_classified=float(df.mu.sum()),
            K=int(df_all.k.sum()), K_classified=int(df.k.sum()),
            unclassified=[f"{r.mission}: {r.subsystem}/{r.component}"
                          for r in df_all[df_all.func == ""].itertuples()])
        print(f"[sq3] {len(df)} of {len(df_all)} types classified, "
              f"K {int(df.k.sum())} of {int(df_all.k.sum())}, E {df.mu.sum():.2f} of {df_all.mu.sum():.2f}")
        # descriptive only: the failures after the boundary, to count each class's early ones
        late = classify(fleet.data(window=(REFERENCE_BOUNDARY_D * HOUR_PER_DAY, None)), default, override)
        results = analyse(df, missions, rng)
        results["sensitivity"] = sensitivity(df, missions)

    for c in results["classes"]:
        c["K_before_boundary"] = c["K"] - int(late.k[late.func == c["cls"]].sum())
    results["class_table"] = class_table(results)

    payload = dict(workbook=wb.name, missions=missions, class_file=CLASS_FILE.name,
                   settings=dict(seed=SEED, evidence="whole record",
                                 reference_boundary_d=REFERENCE_BOUNDARY_D,
                                 min_expected=MIN_EXPECTED, lean=LEAN,
                                 omega_scale=OMEGA_SCALE, omega_sensitivity=OMEGA_SENSITIVITY,
                                 alpha0_fixed=ALPHA0_FIXED, grid=dict(alpha0=LA.size, c=LC.size,
                                 omega=OM.size, class_bias=Y.size, quadrature=XQ.size),
                                 check_draws=N_CHECK),
                   coverage=coverage, results=results)
    (out_dir / "appendix_classes.json").write_text(json.dumps(payload, indent=1))

    fig_overview(results, [out_dir / "appendix_class_overview.png"])
    fig_estimates(results, [out_dir / "appendix_class_estimates.png"])
    fig_groupings(results, [out_dir / "appendix_grouping_comparison.png"])

    r = results
    print("\n" + "=" * 90)
    print("CLASSES (whole record)")
    print("=" * 90)
    for c in r["classes"]:
        p = r["class_posteriors"][c["cls"]]
        print(f"  {c['cls']:<15} types={c['types']:>3} K={c['K']:>2} ({c['K_before_boundary']} in {REFERENCE_BOUNDARY_D} d) "
              f"E={c['E']:6.2f} ratio={c['ratio']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]  "
              f"partial {p['partial'][0]:.2f} [{p['partial'][1]:.2f}, {p['partial'][2]:.2f}] P<1={p['partial'][3]:.2f}  "
              f"indep {p['independent'][0]:.2f} [{p['independent'][1]:.2f}, {p['independent'][2]:.2f}]")
    print(f"  single correction c = {r['fleet_c'][0]:.3f} [{r['fleet_c'][1]:.3f}, {r['fleet_c'][2]:.3f}]")
    for cls, s in r["pooled_check"].items():
        print(f"  check {cls:<15} observed {s['observed']:>2}, replicated {s['median']:.0f} "
              f"[{s['lo']:.0f}, {s['hi']:.0f}]  P(<=)={s['p_at_most']:.2f} P(>=)={s['p_at_least']:.2f}")
    print("  class table:")
    for t in r["class_table"]:
        tail = "-" if t["tail_p"] is None else f"{t['tail_p']:.2f}"
        print(f"    {t['cls']:<15} {t['K']:>2} vs {t['predicted'][0]:.0f} [{t['predicted'][1]:.0f}, "
              f"{t['predicted'][2]:.0f}] check={t['check']:<6} tail={tail:<5} early {t['K_before_boundary']}  "
              f"own {t['own_bias'][0]:.2f} [{t['own_bias'][1]:.2f}, {t['own_bias'][2]:.2f}] "
              f"P(>1)={t['p_above']:.2f} -> {t['own']}")
    for s in r["schemes"][1:]:
        fmt = lambda v: "n/a" if v is None else f"{v:.0f}"
        print(f"  missions needed  {s['scheme']:<30} independent {fmt(s['missions_needed_I'])}, "
              f"partial {fmt(s['missions_needed_H'])}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
