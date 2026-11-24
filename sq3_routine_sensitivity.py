"""SQ3, routine operations: is the bias after the commissioning boundary robust to the prior
specification?

sq3_failure_timing.py refits the SQ1 calibration, unchanged, on the exposure and failures
after the 90-day boundary and finds the handbook conservative there. This script runs the
full SQ1 sensitivity suite, sensitivity.py unchanged (28 perturbations of hyper-priors, data,
structure and numerics, the hyper-prior width sweep, three failure injections and the
leave-one-out jackknife), on that same after-90-day evidence, so the routine-operations
estimate is stress-tested exactly as the whole-record estimate is.

    python sq3_routine_sensitivity.py

writes results/sq3_routine_sensitivity.json (the sensitivity.py payload plus a `window` and a
`summary` block) and four sq3_routine_sens_*.png figures. About half an hour, mostly the
jackknife.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

import pipeline
import sensitivity
from calibration import load_pooled
from sq3_failure_timing import Fleet, HOUR_PER_DAY, REFERENCE_BOUNDARY_D

NAME = "sq3_routine"
# deliberate corruptions and structural assertions, not plausible specifications
NOT_PLAUSIBLE_GROUPS = {"structure", "injection"}
NOT_PLAUSIBLE_NAMES = {"handbook rates x2", "handbook rates /2"}


def window_tables(fleet: Fleet, tau: float, scratch: Path) -> List[Path]:
    """After-boundary component tables, one per mission, built with the same two calls
    sq3_failure_timing.Fleet.data makes for its reference refit."""
    paths = []
    for m in fleet.missions:
        path = scratch / f"{m}_after_{REFERENCE_BOUNDARY_D}d.csv"
        ctx = replace(fleet.ctx[m], window=(tau, None))
        pipeline.build_clusters(fleet.rbd[m], ctx).to_csv(path, index=False)
        paths.append(path)
    return paths


def summarize(res: dict) -> dict:
    """Does the routine-operations bias stay below one under every perturbation?"""
    scen = res["scenarios"]
    below = [s["name"] for s in scen if s["c"][0] < 1.0]
    excl = [s["name"] for s in scen if s["c"][2] < 1.0]
    plausible = [s for s in scen
                 if s["group"] not in NOT_PLAUSIBLE_GROUPS and s["name"] not in NOT_PLAUSIBLE_NAMES]
    worst = max(plausible, key=lambda s: s["c"][2])
    loo_c = np.array([x["c"] for x in res["loo"]])
    loo_base = res["loo_base"]["c"][0]
    inj = [s["c"][0] for s in res["injections"]]
    return dict(
        n_scenarios=len(scen),
        median_below_1=len(below),
        interval_below_1=len(excl),
        median_not_below_1=[s["name"] for s in scen if s["c"][0] >= 1.0],
        interval_reaching_1=[s["name"] for s in scen if s["c"][2] >= 1.0],
        plausible=dict(n=len(plausible),
                       median_range=[min(s["c"][0] for s in plausible),
                                     max(s["c"][0] for s in plausible)],
                       highest_upper_limit=worst["c"][2], highest_upper_limit_scenario=worst["name"],
                       all_intervals_below_1=all(s["c"][2] < 1.0 for s in plausible)),
        loo=dict(c_range=[float(loo_c.min()), float(loo_c.max())], reference=loo_base,
                 max_rel_move_pct=float(np.max(np.abs(loo_c - loo_base)) / loo_base * 100)),
        injection_c_range=[min(inj), max(inj)],
    )


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
    tau = REFERENCE_BOUNDARY_D * HOUR_PER_DAY

    with tempfile.TemporaryDirectory(prefix="sq3_routine_") as tmp:
        scratch = Path(tmp)
        fleet = Fleet(wb, missions, scratch)
        csvs = window_tables(fleet, tau, scratch)
        # the suite must see exactly the evidence of the routine-operations refit
        ours, ref = load_pooled(csvs), fleet.data(window=(tau, None))
        assert ours.label == ref.label and ours.mission == ref.mission
        assert np.array_equal(ours.k, ref.k) and np.allclose(ours.T, ref.T)
        assert np.allclose(ours.lam_prior, ref.lam_prior)
        res = sensitivity.run(csvs, name=NAME, out_dir=out_dir)

    summary = summarize(res)
    path = out_dir / f"{NAME}_sensitivity.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["window"] = dict(boundary_d=REFERENCE_BOUNDARY_D, start_h=tau, end_h=None,
                             N=int(ours.n), K=int(ours.k.sum()), E=float(ours.mu.sum()))
    payload["summary"] = summary
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    p = summary["plausible"]
    print(f"\nROUTINE OPERATIONS (after {REFERENCE_BOUNDARY_D} d): N={ours.n} K={int(ours.k.sum())} "
          f"E={ours.mu.sum():.2f}, baseline c={res['baseline']['c']}")
    print(f"  median below 1 in {summary['median_below_1']} of {summary['n_scenarios']} scenarios; "
          f"not below: {summary['median_not_below_1']}")
    print(f"  90% interval below 1 in {summary['interval_below_1']} of {summary['n_scenarios']}; "
          f"reaching 1: {summary['interval_reaching_1']}")
    print(f"  plausible scenarios ({p['n']}): median {p['median_range'][0]:.3f} to "
          f"{p['median_range'][1]:.3f}, highest upper limit {p['highest_upper_limit']:.3f} "
          f"({p['highest_upper_limit_scenario']}), all below 1: {p['all_intervals_below_1']}")
    lo = summary["loo"]
    print(f"  leave-one-out c in [{lo['c_range'][0]:.3f}, {lo['c_range'][1]:.3f}] about "
          f"{lo['reference']:.3f}, max move {lo['max_rel_move_pct']:.1f}%")
    print(f"  one injected failure: c from {summary['injection_c_range'][0]:.3f} to "
          f"{summary['injection_c_range'][1]:.3f}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
