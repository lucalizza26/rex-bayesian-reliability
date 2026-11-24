"""
sq1_data_ladder.py — SQ1: how prior sensitivity varies with the amount of in-orbit data.

SQ1 has two clauses. calibration.py and sensitivity.py address the first ("how
sensitive are posterior reliability estimates to the choice and specification of
prior distributions"); this script addresses the second — "**and how does this
sensitivity vary with the amount of available in-orbit data**".

The method is a ladder. The fleet is refitted on every subset of missions of size
1, 2, ... M; on each subset the SQ1 specification axes are re-run and their effect
measured against that subset's OWN baseline. Plotting effect against the subset's
total expected failures

    mu_tot = sum_i lambda_prior_i * T_i

gives a decay curve: how much does the specification of the component prior still
matter, once this much evidence has accumulated. mu_tot rather than mission count
is the x-axis because it is the quantity that actually buys precision — the
relative error on the bias c has a floor of 1/sqrt(mu_tot) — and because it makes
fleets of different size and lifetime comparable.

The axes swept are the ones that set the COMPONENT prior directly (SQ1 proper),
not the hyper-priors on (log alpha_0, log c), which regularise the estimator
rather than specify the prior:

    strength  alpha_0 pinned at 0.1 / 1 / 10 instead of learned from the fleet
    location  c pinned at 1, i.e. the handbook assumed unbiased
    location  every handbook FIT scaled by 2 and by 1/2 (global miscalibration)

INPUT IS THE WORKBOOK, NOTHING ELSE. The per-mission RBD and components CSVs are
rebuilt in a temporary directory and thrown away, so this never reads — or
writes — the derived CSVs sitting in the mission folders, and it can be pointed
at any workbook, including one held outside this repository.

Usage:
    python sq1_data_ladder.py                       # the thesis setting (up to 35 subsets per rung)
    python sq1_data_ladder.py --list                # what the workbook declares
    python sq1_data_ladder.py --missions SC1 SC2 --repeats 4

writes results/sq1_ladder.json and the figures sq1_ladder_sensitivity.png and
sq1_ladder_precision.png.
"""

import argparse
import itertools
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
import calibration
from calibration import (CalibrationData, load_calibration_data, pool,
                         fit_hyperparameters, component_posteriors)
from sensitivity import scale_priors, compare_components, shrink_of

# Fewer draws than calibration.N_DRAWS: this runs one posterior per axis per
# subset, so the cost is ~7x the number of subsets. The metric is a DIFFERENCE
# against a baseline drawn with the same seed, so the Monte-Carlo error largely
# cancels (sensitivity.py measures ~1% residual at 20 000 draws); the effects
# being resolved here are 5-100%. Raise with --draws if the curves look ragged.
N_DRAWS = 4000
SEED = 11

# ── the SQ1 axes: specification of the COMPONENT prior ───────────────────────
# (label, kind, fit_hyperparameters kwargs) — these pin a parameter of the prior
# instead of learning it from the fleet.
FIT_AXES: List[Tuple[str, str, dict]] = [
    ("a0 = 0.1",   "strength", dict(fix_a0=0.1)),
    ("a0 = 1",     "strength", dict(fix_a0=1.0)),
    ("a0 = 10",    "strength", dict(fix_a0=10.0)),
    ("c = 1",      "location", dict(fix_c=1.0)),
]
# (label, kind, factor) — these rescale the handbook rate the prior is centred on.
DATA_AXES: List[Tuple[str, str, float]] = [
    ("handbook x2", "handbook", 2.0),
    ("handbook /2", "handbook", 0.5),
]
AXIS_NAMES = [a[0] for a in FIT_AXES] + [a[0] for a in DATA_AXES]
AXIS_KIND = {n: k for n, k, _ in FIT_AXES + DATA_AXES}
AXIS_COLOUR = {"a0 = 0.1": "C0", "a0 = 1": "C9", "a0 = 10": "C1", "c = 1": "C3",
               "handbook x2": "C2", "handbook /2": "C8"}


# ─────────────────────────────────────────────────────────────────────────────
# Workbook resolution — the only input
# ─────────────────────────────────────────────────────────────────────────────
def resolve_workbook(directory: Optional[str], workbook: Optional[str]) -> Path:
    """Locate the workbook from --workbook, or find it in --dir (default: the repo).

    Discovery is pipeline.find_workbook, so this script and sq1_update.py agree on which
    workbook is "the" workbook — real or anonymised, the name does not matter.
    """
    if workbook:
        wb = Path(workbook).expanduser()
        if not wb.exists():
            raise SystemExit(f"workbook not found: {wb}")
        return wb.resolve()
    return pipeline.find_workbook(directory)


def build_missions(workbook: Path, missions: Optional[Sequence[str]],
                   scratch: Path) -> Dict[str, CalibrationData]:
    """Rebuild every mission's components from the workbook into `scratch`.

    Returns one CalibrationData per mission. The intermediates live only in the
    temporary directory, so the mission folders on disk are neither read nor
    written and the workbook stays the single source of truth.
    """
    declared = pipeline.list_missions(workbook)
    if missions:
        unknown = [m for m in missions if m not in declared]
        if unknown:
            raise SystemExit(f"not in {workbook.name}: {', '.join(unknown)}\n"
                             f"declared: {', '.join(declared)}")
        wanted = list(missions)
    else:
        wanted = declared

    out: Dict[str, CalibrationData] = {}
    for m in wanted:
        rbd_csv = scratch / m / f"{m}_rbd.csv"
        comp_csv = scratch / m / f"{m}_components.csv"
        pipeline.convert(m, rbd_csv, workbook)
        pipeline.cluster(rbd_csv, comp_csv, workbook=workbook)
        data = load_calibration_data(comp_csv, mission=m)
        if data.n == 0:
            print(f"[ladder] {m}: no usable components (no handbook FIT) — skipped")
            continue
        out[m] = data
    if not out:
        raise SystemExit("no mission produced usable components")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One rung of the ladder
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subset(names: Sequence[str], per_mission: Dict[str, CalibrationData],
                    *, ngrid: int, draws: int, seed: int) -> dict:
    """Fit one subset of missions and measure every SQ1 axis against its own
    baseline. Each subset is self-contained: the axes are compared to the fit
    that subset's data supports, never to the full fleet's."""
    data = pool([per_mission[n] for n in names])
    base_fit = fit_hyperparameters(data, ngrid=ngrid)
    base_rows = component_posteriors(data, base_fit, n_draws=draws, seed=seed)

    axes = {}
    for label, kind, kw in FIT_AXES:
        f = fit_hyperparameters(data, ngrid=ngrid, **kw)
        rows = component_posteriors(data, f, n_draws=draws, seed=seed)
        axes[label] = dict(kind=kind, **compare_components(rows, base_rows),
                           c=[round(v, 3) for v in f.c])
    for label, kind, factor in DATA_AXES:
        d2 = scale_priors(data, factor)
        f = fit_hyperparameters(d2, ngrid=ngrid)
        rows = component_posteriors(d2, f, n_draws=draws, seed=seed)
        axes[label] = dict(kind=kind, **compare_components(rows, base_rows),
                           c=[round(v, 3) for v in f.c])

    mu_tot = float(data.mu.sum())
    return dict(
        missions=list(names), n_missions=len(names), n_components=data.n,
        mu_tot=round(mu_tot, 3), K=int(data.k.sum()),
        c=[round(v, 4) for v in base_fit.c], a0=[round(v, 4) for v in base_fit.a0],
        # the SQ1 conclusion is "c < 1"; does this much data still support it?
        c_excludes_1=bool(base_fit.c[2] < 1.0),
        ci_ratio=round(float(base_fit.c[2] / base_fit.c[1]), 3),
        rse_floor=round(float(1.0 / np.sqrt(mu_tot)), 4) if mu_tot > 0 else None,
        shrink=round(shrink_of(base_rows), 3),
        axes=axes)


def subsets_for(missions: Sequence[str], size: int, repeats: int,
                rng: np.random.Generator) -> List[Tuple[str, ...]]:
    """All mission combinations of this size, or `repeats` of them sampled without
    replacement — so a rung is never represented by one arbitrary subset."""
    combos = list(itertools.combinations(missions, size))
    if len(combos) <= repeats:
        return combos
    idx = rng.choice(len(combos), size=repeats, replace=False)
    return [combos[i] for i in sorted(idx)]


def run_ladder(per_mission: Dict[str, CalibrationData], *, repeats: int,
               ngrid: int, draws: int, seed: int) -> List[dict]:
    names = list(per_mission)
    rng = np.random.default_rng(seed)
    results = []
    for size in range(1, len(names) + 1):
        combos = subsets_for(names, size, repeats, rng)
        print(f"[ladder] rung {size}/{len(names)} missions - {len(combos)} subset(s)")
        for combo in combos:
            results.append(evaluate_subset(combo, per_mission, ngrid=ngrid,
                                           draws=draws, seed=seed))
    return results


def aggregate(results: List[dict]) -> List[dict]:
    """Collapse the subsets of each rung to a median and an interquartile band."""
    rungs = []
    for size in sorted({r["n_missions"] for r in results}):
        grp = [r for r in results if r["n_missions"] == size]
        row = dict(n_missions=size, n_subsets=len(grp),
                   mu_tot=round(float(np.median([r["mu_tot"] for r in grp])), 3),
                   K=round(float(np.median([r["K"] for r in grp])), 1),
                   n_components=int(np.median([r["n_components"] for r in grp])),
                   c_med=round(float(np.median([r["c"][0] for r in grp])), 4),
                   c_lo=round(float(np.median([r["c"][1] for r in grp])), 4),
                   c_hi=round(float(np.median([r["c"][2] for r in grp])), 4),
                   excl_1=sum(r["c_excludes_1"] for r in grp),
                   axes={})
        for a in AXIS_NAMES:
            v = [r["axes"][a]["move_med"] for r in grp]
            row["axes"][a] = dict(med=round(float(np.median(v)), 1),
                                  q25=round(float(np.percentile(v, 25)), 1),
                                  q75=round(float(np.percentile(v, 75)), 1))
        rungs.append(row)
    return rungs


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def by_evidence(rungs: List[dict]) -> List[dict]:
    """Rungs in ascending mu_tot — the order the figures must connect them in.

    aggregate() returns rungs by SUBSET SIZE, which is the right order for the
    table but not for the plots: the x-axis is mu_tot, and a rung sits at the
    MEDIAN mu_tot of the subsets sampled for it. Because --repeats samples only a
    handful of the combinations at each size, that median is not monotonic in
    size — a 5-mission rung drawn from short missions can carry less evidence
    than a 4-mission rung drawn from long ones. Connecting rungs in size order
    then draws a line that doubles back on itself. Sorting by the quantity
    actually on the axis removes the artefact without touching the numbers.
    """
    return sorted(rungs, key=lambda g: g["mu_tot"])


def fig_sensitivity(results: List[dict], rungs: List[dict], tag: str,
                    out_png: Path) -> None:
    """THE SQ1 figure: effect of misspecifying the component prior, against the
    amount of evidence. A decaying curve says the data is taking over; a flat one
    says the prior still dictates the answer."""
    rungs = by_evidence(rungs)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
#    fig.suptitle(f"{tag} — does the specification of the component prior still "
#                 f"matter as REX evidence accumulates?", fontsize=13, fontweight="bold")

    for a, kinds in ((ax[0], {"strength"}), (ax[1], {"location", "handbook"})):
        for name in (n for n in AXIS_NAMES if AXIS_KIND[n] in kinds):
            col = AXIS_COLOUR[name]
            xs = [r["mu_tot"] for r in results]
            ys = [r["axes"][name]["move_med"] for r in results]
            a.scatter(xs, ys, s=14, color=col, alpha=.28, zorder=2)
            mx = [g["mu_tot"] for g in rungs]
            a.fill_between(mx, [g["axes"][name]["q25"] for g in rungs],
                           [g["axes"][name]["q75"] for g in rungs],
                           color=col, alpha=.12, zorder=1)
            a.plot(mx, [g["axes"][name]["med"] for g in rungs], "o-", color=col,
                   lw=2, label=name, zorder=3)
        a.set_xlabel(r"evidence  $\mu_{tot}$ = expected failures  "
                     "(pooled over the subset)")
        a.set_ylabel("movement of component posterior means vs baseline  [%]")
        a.legend(fontsize=9); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_precision(results: List[dict], rungs: List[dict], tag: str,
                  out_png: Path) -> None:
    """What the same evidence buys the headline result: the bias c and its CI."""
    rungs = by_evidence(rungs)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
#    fig.suptitle(f"{tag} — precision of the learned bias c vs accumulated evidence",
#                 fontsize=13, fontweight="bold")

    a = ax[0]
    mx = [g["mu_tot"] for g in rungs]
    a.fill_between(mx, [g["c_lo"] for g in rungs], [g["c_hi"] for g in rungs],
                   color="C2", alpha=.18, label="90% CI (rung median)")
    a.plot(mx, [g["c_med"] for g in rungs], "o-", color="C2", lw=2, label="c median")
    a.scatter([r["mu_tot"] for r in results], [r["c"][0] for r in results],
              s=14, color="C2", alpha=.35, zorder=3, label="individual subsets")
    a.axhline(1.0, color="k", ls=":", lw=1.2, label="c = 1 (handbook unbiased)")
    a.set_yscale("log")
    a.set_xlabel(r"evidence  $\mu_{tot}$"); a.set_ylabel("bias c")
    a.legend(fontsize=9); a.grid(alpha=.3, which="both")

    a = ax[1]
    a.plot(mx, [g["c_hi"] / g["c_lo"] for g in rungs], "o-", color="C0", lw=2,
           label="CI width ratio  q95/q05")
    mu = np.array([m for m in mx if m > 0])
    if len(mu):
        # 1/sqrt(mu_tot) is the Poisson floor on the relative error of c; shown
        # normalised to the widest rung so only the SHAPE is being compared.
        ref = 1.0 / np.sqrt(mu)
        scale = (rungs[0]["c_hi"] / rungs[0]["c_lo"]) / ref[0]
        a.plot(mu, ref * scale, "--", color="C7", lw=1.5,
               label=r"$1/\sqrt{\mu_{tot}}$ reference (shape only)")
    a.set_xlabel(r"evidence  $\mu_{tot}$"); a.set_ylabel("q95 / q05 of c")
    a.legend(fontsize=9); a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def report(rungs: List[dict], tag: str) -> None:
    head = (f"{'miss':>4} {'sub':>4} {'comp':>5} {'mu_tot':>8} {'K':>5} "
            f"{'c med':>7} {'90% CI':>15} {'c<1':>5} | "
            + " ".join(f"{a:>11}" for a in AXIS_NAMES))
    print("=" * len(head))
    print(f"SQ1 DATA LADDER - {tag}   (axis movement = % shift of component means)")
    print("=" * len(head))
    print(head)
    for g in rungs:
        ci = f"[{g['c_lo']:.2f}, {g['c_hi']:.2f}]"
        print(f"{g['n_missions']:>4} {g['n_subsets']:>4} {g['n_components']:>5} "
              f"{g['mu_tot']:>8.2f} {g['K']:>5.1f} {g['c_med']:>7.3f} {ci:>15} "
              f"{g['excl_1']:>3}/{g['n_subsets']:<1} | "
              + " ".join(f"{g['axes'][a]['med']:>11.1f}" for a in AXIS_NAMES))
    print("=" * len(head))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SQ1: how sensitivity to the component prior varies "
                    "with the amount of in-orbit data. Reads the workbook only.")
    ap.add_argument("--dir", default=None,
                    help="folder holding the workbook (default: data/)")
    ap.add_argument("--workbook", default=None,
                    help="path to the workbook, if --dir holds more than one")
    ap.add_argument("--missions", nargs="*", default=None,
                    help="restrict to these mission ids (default: all declared)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write results (default: results/)")
    ap.add_argument("--repeats", type=int, default=35,
                    help="max mission subsets evaluated per rung (default 35, all of them)")
    ap.add_argument("--draws", type=int, default=N_DRAWS,
                    help=f"posterior MC draws per component (default {N_DRAWS})")
    ap.add_argument("--ngrid", type=int, default=calibration.NGRID,
                    help=f"hyperparameter grid points per axis "
                         f"(default {calibration.NGRID})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--list", action="store_true",
                    help="show what the workbook declares, then exit")
    args = ap.parse_args(argv)

    wb = resolve_workbook(args.dir, args.workbook)
    print(f"[ladder] workbook: {wb}")
    if args.list:
        declared = pipeline.list_missions(wb)
        sheets = pipeline.sheet_names(wb)
        print(f"[ladder] missions declared ({len(declared)}): {', '.join(declared)}")
        missing = [m for m in declared if m not in sheets]
        if missing:
            print(f"[ladder] WARNING no RBD sheet for: {', '.join(missing)}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sq1_ladder_") as tmp:
        per_mission = build_missions(wb, args.missions, Path(tmp))
        tag = f"{len(per_mission)} missions"
        print(f"[ladder] {tag}: " + ", ".join(
            f"{m} (n={d.n}, mu={d.mu.sum():.2f}, k={int(d.k.sum())})"
            for m, d in per_mission.items()))
        results = run_ladder(per_mission, repeats=args.repeats, ngrid=args.ngrid,
                             draws=args.draws, seed=args.seed)

    rungs = aggregate(results)
    report(rungs, tag)

    fig_sensitivity(results, rungs, tag, out_dir / "sq1_ladder_sensitivity.png")
    fig_precision(results, rungs, tag, out_dir / "sq1_ladder_precision.png")
    payload = dict(workbook=wb.name, missions=list(per_mission),
                   settings=dict(repeats=args.repeats, draws=args.draws,
                                 ngrid=args.ngrid, seed=args.seed),
                   axes=[dict(name=n, kind=k) for n, k, _ in FIT_AXES]
                        + [dict(name=n, kind=k) for n, k, _ in DATA_AXES],
                   rungs=rungs, subsets=results)
    with open(out_dir / "sq1_ladder.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"figures + sq1_ladder.json written to {out_dir}")


if __name__ == "__main__":
    main()
