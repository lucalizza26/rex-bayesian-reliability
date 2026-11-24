"""
calibration.py — Conjugate (Gamma-Poisson) empirical-Bayes calibration of the
handbook priors produced by pipeline.py.

This is the working method for SQ1 (prior specification / sensitivity). It takes,
for every component i:

    lambda_prior_i  — the handbook failure rate, = FIT_i * 1e-9  (prior mean rate)
    T_i             — operational hours           (failure-censored exposure)
    k_i             — observed failures

and learns, from the whole fleet at once, how much to trust the handbook and by how
much it is globally biased — then returns a calibrated Gamma posterior for every
component.

The "fleet" may span SEVERAL MISSIONS: pass a list of components CSVs (see sq1_update.py)
and one shared (alpha_0, c) is calibrated on all of them together. Components are
never merged across missions — a component carries its mission tag, so two missions
that both have an "AOCS/Reaction Wheel" stay two separate entries with their own
exposure, failures and handbook FIT.

────────────────────────────────────────────────────────────────────────────────
The model
────────────────────────────────────────────────────────────────────────────────
Per component, a conjugate Gamma-Poisson update with a shared prior:

    prior       lambda_i ~ Gamma(alpha_0, beta_i),  beta_i = alpha_0 / (c * lambda_prior_i)
    likelihood  k_i ~ Poisson(lambda_i * T_i)
    posterior   lambda_i | k_i ~ Gamma(alpha_0 + k_i, beta_i + T_i)      (conjugate)

Two shared scalars carry all the learning:
    alpha_0  — the prior STRENGTH  (pseudo-failures; small = weak, uncertain prior)
    c        — a global BIAS factor (prior mean = c * lambda_prior; c < 1 = handbook
               pessimistic)

Integrating lambda_i out makes each count Negative-Binomial:

    k_i ~ NegBin(alpha_0, p_i),   p_i = alpha_0 / (alpha_0 + c * mu_i),   mu_i = lambda_prior_i * T_i

────────────────────────────────────────────────────────────────────────────────
Why a grid and not a point estimate
────────────────────────────────────────────────────────────────────────────────
With only a handful of failures, a plain maximum-likelihood point estimate of
(alpha_0, c) is not identifiable: the marginal likelihood is a long, nearly flat
ridge that a single outlier can drag to a degenerate, pathologically weak prior.
The fix is to REGULARISE the two scalars with weak priors and integrate over them:

    log alpha_0 ~ Normal(0, 1),   log c ~ Normal(0, 1)

evaluated on a small 2-D log-grid. Each component's posterior is then a mixture of
Gammas, weighted by the grid posterior of (alpha_0, c). No MCMC is required — only a
2-D numerical integration; the per-component layer stays closed-form Gamma.

Usage:
    python calibration.py [components_csv ...] [--name NAME] [--out-dir DIR] [--no-show]
A single CSV writes the four figures and <mission>_calibration.json next to it; several
CSVs pool them and write <NAME>_*.png / <NAME>_calibration.json into results/ (this is what
sq1_update.py drives).
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.stats import nbinom, gamma as gamma_dist
from scipy.special import gammaln
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline


FIT = 1e9  # 1 FIT = 1e-9 failures / hour

# ── Grid + regularisation settings (the method's only knobs) ──────────────────
A0_RANGE = (0.05, 20.0)   # strength alpha_0 grid range
C_RANGE = (0.05, 6.0)     # bias c grid range
NGRID = 90                # points per axis
PRIOR_SD = 1.0            # sd of the weak priors  log alpha_0, log c ~ N(0, PRIOR_SD^2)
N_DRAWS = 20000           # posterior draws per component
SEED = 11
N_SAMPLES = 6             # components highlighted in the per-component figures / tables


# ─────────────────────────────────────────────────────────────────────────────
# Data container + loader
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CalibrationData:
    """Per-component inputs the calibration needs (one entry per component).

    Entries from several missions may be pooled into one CalibrationData (see
    `pool` / `load_pooled`): the fit then learns ONE (alpha_0, c) from all of them.
    `mission[i]` tags which mission entry i came from, and is part of the component's
    identity everywhere — so components sharing a name across missions are never
    merged or matched with each other.
    """
    label:     List[str]              # "subsystem/component", e.g. "AOCS/Reaction Wheel"
    lam_prior: np.ndarray             # handbook prior mean rate  (failures / hour)
    T:         np.ndarray             # operational hours (failure-censored exposure)
    k:         np.ndarray             # observed failures (counts)
    mission:   Optional[List[str]] = None   # mission id per entry; defaults to ""

    def __post_init__(self):
        if self.mission is None:
            self.mission = [""] * len(self.label)

    @property
    def n(self) -> int:
        return len(self.k)

    @property
    def mu(self) -> np.ndarray:
        """Handbook-expected failure counts mu_i = lambda_prior_i * T_i."""
        return self.lam_prior * self.T

    @property
    def missions(self) -> List[str]:
        """Distinct missions pooled here, in first-seen order."""
        return list(dict.fromkeys(self.mission))

    def key(self, i: int) -> Tuple[str, str]:
        """Identity of entry i: (mission, 'subsystem/component')."""
        return (self.mission[i], self.label[i])

    def subset(self, mask) -> "CalibrationData":
        """New dataset keeping only the entries where mask is True."""
        m = np.asarray(mask, dtype=bool)
        return CalibrationData(
            label=[l for j, l in enumerate(self.label) if m[j]],
            lam_prior=self.lam_prior[m], T=self.T[m], k=self.k[m],
            mission=[s for j, s in enumerate(self.mission) if m[j]])

    def with_extra(self, mission: str, label: str, lam: float, T: float,
                   k: int) -> "CalibrationData":
        """New dataset with one extra component appended."""
        return CalibrationData(
            label=self.label + [label], lam_prior=np.append(self.lam_prior, lam),
            T=np.append(self.T, T), k=np.append(self.k, k),
            mission=self.mission + [mission])


def pool(datasets: List[CalibrationData]) -> CalibrationData:
    """Concatenate per-mission datasets into one fleet-wide dataset.

    Entries are kept exactly as they are — no grouping, no de-duplication — so the
    single (alpha_0, c) fit sees every mission's components as distinct units.
    """
    if not datasets:
        raise ValueError("pool() needs at least one CalibrationData")
    return CalibrationData(
        label=[l for d in datasets for l in d.label],
        lam_prior=np.concatenate([d.lam_prior for d in datasets]) if datasets else np.array([]),
        T=np.concatenate([d.T for d in datasets]),
        k=np.concatenate([d.k for d in datasets]).astype(int),
        mission=[m for d in datasets for m in d.mission])


def load_calibration_data(components_csv,
                          mission: Optional[str] = None) -> CalibrationData:
    """Build CalibrationData from a pipeline components CSV.

    lambda_prior is the handbook FIT read straight from the workbook, converted
    only by its unit definition: FIT = failures per 1e9 hours, so

        lambda_prior [1/h] = FIT * 1e-9

    No mission duration enters this — the workbook states the rate directly, and
    the reliability column is never used.

    Components are excluded when they have no handbook prior (FIT NA or <= 0) or
    no exposure (operational_hours <= 0, i.e. cold-standby-only types) — both
    carry zero likelihood information.

    Real missions lack handbook data for some components, leaving the FIT at 0 or
    filled with a placeholder ('-', 'tbd', ...). A single non-numeric cell makes
    pandas read the whole column as strings, so the numeric inputs are coerced
    first (unparseable -> NaN); such components are then EXCLUDED like any other
    prior-less component rather than crashing the comparisons.
    """
    df = pd.read_csv(components_csv)
    if mission is None:
        # normally one mission per file, but read the column per row so a CSV that
        # happens to hold several missions is tagged correctly
        row_mission = df['mission'].astype(str).str.strip()
    else:
        row_mission = pd.Series([str(mission)] * len(df), index=df.index)

    fit = pd.to_numeric(df['fit'], errors='coerce')
    hours = pd.to_numeric(df['operational_hours'], errors='coerce')
    fails = pd.to_numeric(df['n_failures'], errors='coerce')

    labels, lam, T, k, miss = [], [], [], [], []
    skipped, zero_exposure = [], []
    for i, c in df.iterrows():
        f = fit[i]
        name = f"{c['subsystem_name']}/{c['component_name']}"
        if pd.isna(f) or f <= 0:
            skipped.append(name)
            continue
        T_i = hours[i]
        if pd.isna(T_i) or T_i <= 0:
            zero_exposure.append(name)
            continue
        labels.append(name)
        miss.append(row_mission[i])
        lam.append(float(f) * 1e-9)
        T.append(float(T_i))
        k.append(int(fails[i]) if pd.notna(fails[i]) else 0)

    tag = "/".join(row_mission.unique())
    if skipped:
        print(f"[load] {tag}: excluded {len(skipped)} component(s) with no valid prior: "
              + ", ".join(skipped))
    if zero_exposure:
        print(f"[load] {tag}: excluded {len(zero_exposure)} component(s) with zero "
              f"exposure (operational_hours = 0): " + ", ".join(zero_exposure))

    return CalibrationData(
        label=labels,
        lam_prior=np.asarray(lam, dtype=float),
        T=np.asarray(T, dtype=float),
        k=np.asarray(k, dtype=int),
        mission=miss,
    )


def load_pooled(components_csvs) -> CalibrationData:
    """Load one or several components CSVs into a single fleet-wide dataset.

    Every entry keeps its mission tag, so same-named components from different
    missions remain separate (own exposure, own failures, own handbook FIT).
    """
    csvs = as_csv_list(components_csvs)
    data = pool([load_calibration_data(c) for c in csvs])
    dupes = len(data.label) - len(set(zip(data.mission, data.label)))
    if dupes:
        print(f"[load] warning: {dupes} duplicated (mission, component) key(s) — "
              f"per-component scenario matching may be ambiguous")
    return data


def as_csv_list(components_csvs) -> List[Path]:
    """Accept a single path or an iterable of paths; always return a list of Paths."""
    if isinstance(components_csvs, (str, Path)):
        return [Path(components_csvs)]
    return [Path(c) for c in components_csvs]


def default_components_csvs() -> List[Path]:
    """Standalone-CLI fallback: the components CSV of every mission the workbook
    declares. sq1_update.py is the entry point that selects a subset."""
    base = Path(__file__).parent / "results" / "components"
    return [base / m / f"{m}_components.csv" for m in pipeline.list_missions()]


def resolve_output(csvs: List[Path], data: CalibrationData,
                   name: Optional[str], out_dir) -> Tuple[str, Path]:
    """Decide the output NAME and directory for a run.

    One mission  -> keep the historical behaviour: named after the mission, written
                    next to its CSV.
    Several      -> named `name` (default "FLEET"), written to results/.
    """
    missions = data.missions
    if name is None:
        name = missions[0] if len(missions) == 1 else "FLEET"
    if out_dir is None:
        out_dir = (csvs[0].parent if len(csvs) == 1
                   else Path(__file__).parent / "results")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return name, out_dir


# ─────────────────────────────────────────────────────────────────────────────
# The conjugate model
# ─────────────────────────────────────────────────────────────────────────────
def log_marginal(data: CalibrationData, a0: float, c: float) -> float:
    """Negative-Binomial marginal log-likelihood of the observed counts at (a0, c)."""
    p = a0 / (a0 + c * data.mu)
    return float(np.sum(nbinom.logpmf(data.k, a0, p)))


def log_marginal_grid(data: CalibrationData, A0: np.ndarray, C: np.ndarray) -> np.ndarray:
    """log_marginal evaluated at every grid point at once (same NegBin pmf,
    written out via gammaln so the whole grid is one broadcast)."""
    a0 = A0.reshape(-1, 1)
    c = C.reshape(-1, 1)
    k, mu = data.k[None, :], data.mu[None, :]
    p = a0 / (a0 + c * mu)
    ll = (gammaln(k + a0) - gammaln(a0) - gammaln(k + 1)
          + a0 * np.log(p) + k * np.log1p(-p))
    return ll.sum(axis=1).reshape(A0.shape)


@dataclass
class GridFit:
    """Posterior of the two shared scalars on the log-grid."""
    A0: np.ndarray            # grid of alpha_0 values          (NGRID x NGRID)
    C:  np.ndarray            # grid of c values                (NGRID x NGRID)
    W:  np.ndarray            # normalised posterior weights    (NGRID x NGRID)
    a0: Tuple[float, float, float]   # (median, q05, q95)
    c:  Tuple[float, float, float]   # (median, q05, q95)

    @property
    def a0_med(self) -> float:
        return self.a0[0]

    @property
    def c_med(self) -> float:
        return self.c[0]


def _wquantile(v: np.ndarray, w: np.ndarray, p: float) -> float:
    o = np.argsort(v)
    return float(np.interp(p, np.cumsum(w[o]), v[o]))


def fit_hyperparameters(data: CalibrationData, *,
                        a0_range: Tuple[float, float] = A0_RANGE,
                        c_range: Tuple[float, float] = C_RANGE,
                        ngrid: int = NGRID,
                        prior_sd_a0: float = PRIOR_SD,
                        prior_sd_c: float = PRIOR_SD,
                        prior_mean_la0: float = 0.0,
                        prior_mean_lc: float = 0.0,
                        fix_a0: Optional[float] = None,
                        fix_c: Optional[float] = None) -> GridFit:
    """Fit (alpha_0, c) by regularised numerical integration on a log-grid.

    logpost = marginal log-likelihood + log Normal(mean, sd^2) priors on
    (log a0, log c), exponentiated and normalised to a posterior over the grid.

    Every keyword is a sensitivity knob (defaults = the module constants used by
    the baseline analysis): the grid ranges/resolution, the sd and centre of each
    weak prior, and fix_a0 / fix_c which pin a scalar to one value (its axis
    collapses to a single grid point and its prior term is dropped).
    """
    la = (np.array([np.log(fix_a0)]) if fix_a0 is not None else
          np.linspace(np.log(a0_range[0]), np.log(a0_range[1]), ngrid))
    lc = (np.array([np.log(fix_c)]) if fix_c is not None else
          np.linspace(np.log(c_range[0]), np.log(c_range[1]), ngrid))
    LA, LC = np.meshgrid(la, lc, indexing="ij")
    A0, C = np.exp(LA), np.exp(LC)

    logpost = log_marginal_grid(data, A0, C)
    if fix_a0 is None:
        logpost = logpost - 0.5 * ((LA - prior_mean_la0) / prior_sd_a0) ** 2
    if fix_c is None:
        logpost = logpost - 0.5 * ((LC - prior_mean_lc) / prior_sd_c) ** 2
    W = np.exp(logpost - logpost.max())
    W /= W.sum()

    a0f, cf, wf = A0.ravel(), C.ravel(), W.ravel()
    a0 = (_wquantile(a0f, wf, .5), _wquantile(a0f, wf, .05), _wquantile(a0f, wf, .95))
    c = (_wquantile(cf, wf, .5), _wquantile(cf, wf, .05), _wquantile(cf, wf, .95))
    return GridFit(A0=A0, C=C, W=W, a0=a0, c=c)


def component_posteriors(data: CalibrationData, fit: GridFit,
                         n_draws: int = N_DRAWS, seed: int = SEED) -> List[dict]:
    """Posterior for every component: a mixture of Gammas over the (a0, c) grid.
    The MEAN is exact, sum_g W_g (a0_g + k_i) / (beta_ig + T_i); the median and the
    5th/95th percentiles come from Monte-Carlo draws of the mixture. The thesis
    summarises each component by its mean (medians only for quantities aggregated across
    components); the exact mean has no sampling noise, which would otherwise flip
    low-evidence components, sitting at E[c] x handbook = 0.98 of it, to either side of
    the handbook rate. Rows are sorted by handbook FIT (desc)."""
    rng = np.random.default_rng(seed)
    a0f, cf, wf = fit.A0.ravel(), fit.C.ravel(), fit.W.ravel()
    gi = rng.choice(len(wf), size=n_draws, p=wf)
    a0s, cs = a0f[gi], cf[gi]

    rows = []
    for i in range(data.n):
        beta = a0s / (cs * data.lam_prior[i])
        draws = rng.gamma(a0s + data.k[i], 1.0 / (beta + data.T[i])) * FIT
        beta_g = a0f / (cf * data.lam_prior[i])
        mean_exact = float(np.sum(wf * (a0f + data.k[i]) / (beta_g + data.T[i]))) * FIT
        sub, comp = (data.label[i].split("/", 1) + [""])[:2]
        rows.append(dict(
            idx=i, mission=data.mission[i], sub=sub.strip(), name=comp.strip(),
            k=int(data.k[i]), T=float(data.T[i]),
            hb=float(data.lam_prior[i] * FIT), med=float(np.median(draws)),
            lo=float(np.percentile(draws, 5)), hi=float(np.percentile(draws, 95)),
            mean=mean_exact))
    rows.sort(key=lambda r: -r["hb"])
    return rows


def evidence_weights(data: CalibrationData, fit: GridFit,
                     chunk: int = 64) -> np.ndarray:
    """Posterior-mean evidence weight for every component.

        w_i = T_i / (beta_i + T_i)

    is the share of the posterior mean attributable to OBSERVED evidence rather
    than to the handbook prior; 1 - w_i is the shrinkage towards the handbook.
    Because beta_i = alpha_0 / (c * lambda_prior_i) depends on the two shared
    scalars, w_i is not a single number: it is averaged over the grid posterior
    rather than evaluated at a point estimate, consistently with every other
    quantity the framework reports.

    Returned in the component order of `data`, i.e. indexable by row['idx'].
    """
    a0 = fit.A0.ravel()[None, :]
    c = fit.C.ravel()[None, :]
    gw = fit.W.ravel()[None, :]
    out = np.empty(data.n)
    for s in range(0, data.n, chunk):
        e = min(s + chunk, data.n)
        T = data.T[s:e, None]
        beta = a0 / (c * data.lam_prior[s:e, None])
        out[s:e] = ((T / (beta + T)) * gw).sum(axis=1)
    return out


def row_key(r: dict) -> Tuple[str, str, str]:
    """Identity of a result row: (mission, subsystem, component).

    The mission is part of the key so that a scenario's rows are matched against the
    baseline's per MISSION — components sharing a name across missions never collide.
    """
    return (r.get("mission", ""), r["sub"], r["name"])


def pick_samples(rows: List[dict], n: int = N_SAMPLES) -> List[dict]:
    """Up to n representative component rows to highlight: the failed components
    first, then survivors spread evenly across the handbook-FIT range (rows come in
    sorted by FIT). Data-driven, so it works for any mission and never assumes a
    particular component exists.

    When several missions are pooled, the survivor slots are shared out between the
    missions so the figure is not dominated by whichever mission is largest."""
    picks = [r for r in rows if r["k"] > 0][:n]
    survivors = [r for r in rows if r["k"] == 0]
    need = n - len(picks)
    if need > 0 and survivors:
        by_mission: dict = {}
        for r in survivors:
            by_mission.setdefault(r.get("mission", ""), []).append(r)
        ms = list(by_mission)
        quota = {m: need // len(ms) for m in ms}
        for m in ms[:need % len(ms)]:
            quota[m] += 1
        for m in ms:
            lst, q = by_mission[m], quota[m]
            if q > 0 and lst:
                pos = np.unique(np.linspace(0, len(lst) - 1, q).round().astype(int))
                picks.extend(lst[int(j)] for j in pos)
    picks.sort(key=lambda r: -r["hb"])
    return picks[:n]


def prior_weights(data: CalibrationData, fit: GridFit,
                  sample_rows: List[dict]) -> List[dict]:
    """Prior pseudo-exposure beta_i and prior weight beta_i/(beta_i+T_i) at the
    posterior-median (a0, c), for the highlighted sample components."""
    a0m, cm = fit.a0_med, fit.c_med
    out = []
    for r in sample_rows:
        i = r["idx"]
        beta = a0m / (cm * data.lam_prior[i])
        out.append(dict(mission=data.mission[i], name=r["name"],
                        hb=float(data.lam_prior[i] * FIT), k=int(data.k[i]),
                        beta_Mh=float(beta / 1e6), T_Mh=float(data.T[i] / 1e6),
                        mu=float(data.mu[i]), wt=float(beta / (beta + data.T[i]) * 100)))
    return out


def point_mle(data: CalibrationData) -> Tuple[float, float, float]:
    """Unregularised (no-prior) point MLE of (alpha_0, c) and the log-likelihood
    gain from allowing bias, delta = max_a0 LL(a0, c*) - max_a0 LL(a0, 1).
    Used only to illustrate why point estimation is ill-posed here."""
    neg = lambda x: -log_marginal(data, np.exp(x[0]), np.exp(x[1]))
    res = minimize(neg, x0=[np.log(0.5), np.log(0.5)], method="Nelder-Mead")
    a0_mle, c_mle = float(np.exp(res.x[0])), float(np.exp(res.x[1]))

    ag = np.logspace(-2, 1.5, 200)
    ll_bias = max(log_marginal(data, a, c_mle) for a in ag)
    ll_nobias = max(log_marginal(data, a, 1.0) for a in ag)
    return a0_mle, c_mle, float(ll_bias - ll_nobias)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_fit_surface(data: CalibrationData, mle: Tuple[float, float, float],
                    out_png: Path) -> None:
    """Why regularise: the marginal likelihood is a long flat ridge, so a single
    point estimate runs to a degenerate weak prior. How little the bias adds to the
    fit (dll) is returned by point_mle."""
    a0_mle, c_mle, dll = mle
    fig, a = plt.subplots(figsize=(7.4, 5.2))
    laa = np.linspace(np.log(0.02), np.log(30), 150)
    lcc = np.linspace(np.log(0.05), np.log(6), 150)
    Z = np.array([[log_marginal(data, np.exp(x), np.exp(y)) for x in laa] for y in lcc])
    cf = a.contourf(np.exp(laa), np.exp(lcc), Z, levels=40, cmap="viridis")
    a.plot(a0_mle, c_mle, "*", ms=20, color="red", mec="white", mew=0.8)
    # label x in axes fraction (clear of the left spine), y in data
    a.annotate(f"point MLE\n$\\alpha_0$={a0_mle:.3f}, c={c_mle:.2f}",
               xy=(a0_mle, c_mle), xytext=(0.03, 0.35),
               textcoords=a.get_yaxis_transform(), color="red", fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color="red", lw=1))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"strength $\alpha_0$"); a.set_ylabel("bias c")
    fig.colorbar(cf, ax=a, label="log marginal likelihood")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def fig_hyper(fit: GridFit, mission: str, out_png: Path) -> None:
    """Joint posterior of (alpha_0, c) and its two marginals (the 'optimisation')."""
    A0, C, W = fit.A0, fit.C, fit.W
    a0med, a0lo, a0hi = fit.a0
    cmed, clo, chi = fit.c
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    cf = a.contourf(A0, C, W, levels=30, cmap="viridis")
    a.plot(a0med, cmed, "*", ms=18, color="red", mec="white",
           label=f"median ({a0med:.2f}, {cmed:.2f})")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"strength $\alpha_0$"); a.set_ylabel("bias c")
    a.legend(fontsize=8, loc="upper right")
    cb = fig.colorbar(cf, ax=a)

    wa, wc = W.sum(axis=1), W.sum(axis=0)
    a = ax[1]
    a.plot(A0[:, 0], wa / wa.sum(), color="C0")
    a.axvline(a0med, color="C3", ls="--", label=f"median {a0med:.2f}")
    a.axvspan(a0lo, a0hi, color="C3", alpha=.12, label=f"90% [{a0lo:.2f},{a0hi:.2f}]")
    a.set_xscale("log"); a.set_xlabel(r"strength $\alpha_0$")
    a.set_ylabel("posterior weight")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[2]
    a.plot(C[0, :], wc / wc.sum(), color="C2")
    a.axvline(cmed, color="C3", ls="--", label=f"median {cmed:.2f}")
    a.axvspan(clo, chi, color="C3", alpha=.12, label=f"90% [{clo:.2f},{chi:.2f}]")
    a.axvline(1, color="k", ls=":", lw=1, label="c=1")
    a.set_xscale("log"); a.set_xlabel("bias c")
    a.set_ylabel("posterior weight")
    a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_gamma(data: CalibrationData, fit: GridFit, sample_rows: List[dict],
              mission: str, out_png: Path, multi: bool = False) -> None:
    """Conjugate Gamma prior -> posterior for the highlighted sample components, as
    mixtures over the (a0, c) grid, on a log rate axis."""
    a0f, cf, wf = fit.A0.ravel(), fit.C.ravel(), fit.W.ravel()
    o = np.argsort(-wf); cum = np.cumsum(wf[o])
    keep = o[:np.searchsorted(cum, 0.999) + 1]
    a0k, ck, wk = a0f[keep], cf[keep], wf[keep] / wf[keep].sum()

    FITLO, FITHI = 1e-2, 5e3
    fg = np.logspace(np.log10(FITLO), np.log10(FITHI), 700)
    lg = fg * 1e-9
    ncol = min(3, max(1, len(sample_rows)))
    nrow = max(1, int(np.ceil(len(sample_rows) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.5 * nrow), squeeze=False)
#    fig.suptitle(f"{mission} — conjugate model: Gamma prior -> posterior "
#                 r"(mixture over the $\alpha_0$,c grid)  [log rate]",
#                 fontsize=12.5, fontweight="bold")
    axflat = axes.flat
    for ax_, r in zip(axflat, sample_rows):
        i = r["idx"]
        beta = a0k / (ck * data.lam_prior[i])
        pr = np.zeros_like(lg); po = np.zeros_like(lg)
        for a0v, bv, wv in zip(a0k, beta, wk):
            pr += wv * gamma_dist.pdf(lg, a0v, scale=1 / bv)
            po += wv * gamma_dist.pdf(lg, a0v + data.k[i], scale=1 / (bv + data.T[i]))
        pr *= lg; po *= lg
        m = max(pr.max(), po.max()) or 1
        ax_.plot(fg, pr / m, color="C0", ls="--", lw=1.8, label="prior (mixture)")
        ax_.plot(fg, po / m, color="C3", lw=2.2, label="posterior (mixture)")
        ax_.axvline(data.lam_prior[i] * FIT, color="C7", ls=":", lw=1.1)
        ax_.axvline(r["mean"], color="C3", ls=":", lw=1.1)
        failed = data.k[i] > 0
        tag = f"[{r['mission']}] " if multi and r.get("mission") else ""
        ax_.set_title(f"{tag}{r['name']} — {f'FAILED (k={data.k[i]})' if failed else 'k=0'}\n"
                      f"handbook {data.lam_prior[i]*FIT:.1f} -> post mean {r['mean']:.1f} FIT",
                      color="C3" if failed else "#222", fontsize=10, fontweight="bold")
        ax_.set_xscale("log"); ax_.set_xlim(FITLO, FITHI); ax_.set_ylim(0, 1.1)
        ax_.set_xlabel("failure rate [FIT]"); ax_.set_ylabel("rel. density")
        ax_.grid(alpha=.3, which="both"); ax_.legend(fontsize=8, loc="upper left")
    for ax_ in list(axflat)[len(sample_rows):]:   # hide any unused panels
        ax_.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130); plt.close(fig)


def fig_all(rows: List[dict], mission: str, out_png: Path) -> None:
    """Every component: how far the update moves it from its handbook rate.

    At 642 components a per-component spine is unreadable, so each component is
    plotted against its own handbook rate instead of against an index. Every point
    below the diagonal is a component revised downwards, and the spread around the
    diagonal is the revision itself; panel (b) is that revision as a distribution.

    The posterior MEAN is plotted, not the median. The handbook rate is
    a mean and the prior is built so that its mean is c x handbook, so a mean against
    the handbook compares like with like, and means are what add up to a predicted
    failure count: summed over the fleet these give back the 35 observed. A median
    against the handbook mixes in the skew of the posterior, which at alpha_0 = 0.206
    is large, and reads as a ninefold downward revision of every component."""
    hb = np.array([r["hb"] for r in rows], dtype=float)
    md = np.array([r["mean"] for r in rows], dtype=float)
    ww = np.array([float(r.get("w", np.nan)) for r in rows], dtype=float)
    failed = np.array([r["k"] > 0 for r in rows])
    pos = (hb > 0) & (md > 0)
    ratio = np.full(hb.shape, np.nan)
    ratio[pos] = md[pos] / hb[pos]

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4),
                           gridspec_kw=dict(width_ratios=[1.3, 1]))

    a = ax[0]
    lo_lim = min(hb[pos].min(), md[pos].min()) * 0.4
    hi_lim = max(hb[pos].max(), md[pos].max()) * 2.5
    a.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color="C7", lw=1.2, zorder=2,
           label="no revision")
    sc = a.scatter(hb[pos & ~failed], md[pos & ~failed], c=ww[pos & ~failed],
                   s=13, cmap="viridis", vmin=0, vmax=1, linewidths=0,
                   alpha=0.85, zorder=3)
    a.scatter(hb[pos & failed], md[pos & failed], marker="^", s=46, facecolor="C3",
              edgecolor="white", linewidths=0.7, zorder=5, label="observed failure")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(lo_lim, hi_lim); a.set_ylim(lo_lim, hi_lim)
    a.set_xlabel("handbook rate [FIT]")
    a.set_ylabel("posterior mean [FIT]")
    a.grid(alpha=.3, which="both")
    a.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    cb = fig.colorbar(sc, ax=a, pad=0.02)
    cb.set_label("evidence weight $w_i$", fontsize=9)

    b = ax[1]
    good = np.isfinite(ratio) & (ratio > 0)
    bins = np.logspace(np.log10(ratio[good].min()), np.log10(ratio[good].max()), 46)
    b.hist(ratio[good & ~failed], bins=bins, color="C0", alpha=.85,
           label="no observed failure")
    b.hist(ratio[good & failed], bins=bins, color="C3", alpha=.9,
           label="observed failure")
    med_surv = float(np.median(ratio[good & ~failed]))
    b.axvline(1.0, color="C7", lw=1.2)
    b.axvline(med_surv, color="C0", lw=1.5, ls="--",
              label=f"median survivor, {med_surv:.3f}")
    b.set_xscale("log")
    b.set_xlabel("posterior mean / handbook rate")
    b.set_ylabel("components")
    b.grid(alpha=.3, axis="y")
    b.legend(fontsize=8.5, loc="upper left", framealpha=.95)

    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def mission_breakdown(data: CalibrationData) -> List[dict]:
    """Per-mission contribution to the pooled fit (components, failures, exposure)."""
    out = []
    for m in data.missions:
        sel = np.array([s == m for s in data.mission])
        out.append(dict(mission=m, N=int(sel.sum()), K=int(data.k[sel].sum()),
                        E=round(float(data.mu[sel].sum()), 2),
                        total_Mh=round(float(data.T[sel].sum()) / 1e6, 1)))
    return out


def run(components_csv, show: bool = False, save: bool = True,
        name: Optional[str] = None, out_dir=None) -> dict:
    """Fit the conjugate model on one mission or on several pooled together, write
    the four figures + a results JSON, and return the summary statistics.

    `components_csv` is a single components CSV or a list of them. With a list, ONE
    (alpha_0, c) is calibrated on all of their components at once while each
    component keeps its own mission identity.
    """
    csvs = as_csv_list(components_csv)
    data = load_pooled(csvs)

    if data.n == 0:
        raise ValueError(
            f"No components with a valid prior in {', '.join(str(c) for c in csvs)}: every "
            "component has a missing/placeholder FIT (NA or <=0) or zero "
            "exposure — nothing to calibrate.")

    name, out_dir = resolve_output(csvs, data, name, out_dir)
    missions = data.missions
    multi = len(missions) > 1

    fit = fit_hyperparameters(data)
    rows = component_posteriors(data, fit)
    samples = pick_samples(rows, n=max(N_SAMPLES, min(2 * len(missions), 12)))
    wrows = prior_weights(data, fit, samples)
    mle = point_mle(data)

    # each component summarised by its posterior MEAN; medians only across components
    mean_all = np.array([r["mean"] for r in rows])
    hb_all = np.array([r["hb"] for r in rows])
    surv = [r for r in rows if r["k"] == 0 and r["hb"] > 0]
    surv_ratio = float(np.median([r["mean"] / r["hb"] for r in surv]))
    ratio_pct = [round(float(v), 3) for v in np.percentile(mean_all / hb_all, [5, 25, 50, 75, 95])]
    failed_ratio = float(np.median([r["mean"] / r["hb"] for r in rows if r["k"] > 0]))

    wvec = evidence_weights(data, fit)
    for r in rows:
        r["w"] = round(float(wvec[r["idx"]]), 4)
    wq = [round(float(v), 4) for v in np.percentile(wvec, [5, 25, 50, 75, 95])]

    stats = dict(
        mission=name, missions=missions, by_mission=mission_breakdown(data),
        N=data.n, K=int(data.k.sum()), E=round(float(data.mu.sum()), 2),
        total_Mh=round(float(data.T.sum()) / 1e6, 1),
        a0=[round(v, 3) for v in fit.a0], c=[round(v, 3) for v in fit.c],
        n_below=int((mean_all < hb_all).sum()), surv_ratio=round(surv_ratio, 3),
        failed_ratio=round(failed_ratio, 2), ratio_pct=ratio_pct,
        sum_mean_T=round(float(np.sum(mean_all / FIT * np.array([r["T"] for r in rows]))), 2),
        cv=round(1 / np.sqrt(fit.a0_med), 2),
        w_pct=wq, med_w=wq[2],
        n_w_half=int((wvec > 0.5).sum()), n_w_tenth=int((wvec > 0.1).sum()),
        mle=dict(a0=round(mle[0], 4), c=round(mle[1], 3), dll=round(mle[2], 3)),
    )

    print("=" * 74)
    print(f"CONJUGATE EMPIRICAL-BAYES CALIBRATION — {name}  (n = {data.n})")
    if multi:
        print(f"  pooled over {len(missions)} missions: {', '.join(missions)}"
              f"  (one shared alpha_0 and c)")
        for b in stats["by_mission"]:
            print(f"    {b['mission']:<10} N={b['N']:<4} K={b['K']:<3} "
                  f"E={b['E']:<8} exposure {b['total_Mh']} Mh")
    print("=" * 74)
    print(f"  strength alpha_0 = {stats['a0'][0]}  90% CI [{stats['a0'][1]}, {stats['a0'][2]}]")
    print(f"  bias     c       = {stats['c'][0]}  90% CI [{stats['c'][1]}, {stats['c'][2]}]")
    print(f"  K={stats['K']} failures, E={stats['E']} expected, exposure {stats['total_Mh']} Mh")
    print(f"  {stats['n_below']}/{stats['N']} posterior means below handbook; median survivor "
          f"mean/handbook {stats['surv_ratio']}; failed components x{stats['failed_ratio']}; "
          f"sum of means x T = {stats['sum_mean_T']}")
    print(f"  evidence weight w = T/(beta+T): median {stats['med_w']}, "
          f"5-95% [{stats['w_pct'][0]}, {stats['w_pct'][4]}]")
    print(f"  {stats['n_w_half']}/{stats['N']} components majority-evidence (w>0.5), "
          f"{stats['n_w_tenth']}/{stats['N']} above w=0.1")
    print(f"  (point MLE would give alpha_0={stats['mle']['a0']}, c={stats['mle']['c']}, "
          f"dLL={stats['mle']['dll']} — ill-posed)")
    print("=" * 74)

    if save:
        fig_fit_surface(data, mle, out_dir / f"{name}_eb_fit.png")
        fig_hyper(fit, name, out_dir / f"{name}_eb_hyper.png")
        fig_gamma(data, fit, samples, name, out_dir / f"{name}_eb_gamma.png", multi=multi)
        fig_all(rows, name, out_dir / f"{name}_eb_all.png")
        results = dict(stats=stats, rows=rows, wrows=wrows)
        with open(out_dir / f"{name}_calibration.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"figures + {name}_calibration.json written to {out_dir}")
    if show:
        plt.show()
    return dict(stats=stats, rows=rows, wrows=wrows, fit=fit, name=name, out_dir=out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Conjugate Gamma-Poisson empirical-Bayes prior calibration. "
                    "Give several components CSVs to calibrate one shared "
                    "(alpha_0, c) across missions.")
    ap.add_argument("components_csv", nargs="*",
                    help="pipeline components CSV(s) (default: every mission in "
                         "the workbook)")
    ap.add_argument("--name", default=None,
                    help="output name for a pooled run (default: FLEET)")
    ap.add_argument("--out-dir", default=None, help="where to write figures + JSON")
    ap.add_argument("--no-show", action="store_true",
                    help="save figures but do not open a window (default anyway)")
    args = ap.parse_args(argv)
    run(args.components_csv or default_components_csvs(), show=not args.no_show,
        name=args.name, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
