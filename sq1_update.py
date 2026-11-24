"""SQ1: update the handbook failure rate of every component with the fleet's in-orbit record.

    python sq1_update.py                   # every mission in the workbook (the thesis setting)
    python sq1_update.py SC1 SC2           # a subset of missions
    python sq1_update.py --list            # what the workbook declares, then exit
    python sq1_update.py --no-sensitivity  # calibration only (minutes instead of half an hour)

The input is one workbook in data/ (pipeline.find_workbook): a `mission_list` sheet, a
`failures` sheet and one reliability-model sheet per mission, named after its mission id.

  1. DATA         for every mission, rebuild results/components/<M>/<M>_components.csv with
                  pipeline.py: one row per component type, identical units within the
                  mission merged, failure-censored exposure and observed failures.
  2. CALIBRATE    pool all missions' components into one dataset and integrate the two
                  shared scalars, the bias c and the prior strength alpha_0, over a 90 x 90
                  logarithmic grid (calibration.py); every component gets its posterior.
  3. SENSITIVITY  rerun the calibration under 28 perturbations of the specification, the
                  hyper-prior width sweep, one-failure injections and the leave-one-out
                  jackknife (sensitivity.py).

Components are pooled, never merged across missions: SC1's and SC2's "AOCS/Reaction Wheel"
stay two components with their own exposure, failures, handbook rate and posterior; they
share only the two fleet-level scalars. Within a mission, identical units of one component
type are merged into one row with their operating hours added up.

writes results/sq1_calibration.json, results/sq1_sensitivity.json and their figures.
"""

import argparse
from pathlib import Path

import calibration
import pipeline
import sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# The missions to run together; "all" is every mission the workbook declares.
# Every id must have a row in the workbook's `mission_list` sheet and a reliability-model
# sheet of the same name. Run `python sq1_update.py --list` to see what the workbook offers.
# ─────────────────────────────────────────────────────────────────────────────
MISSIONS = ["all"]

# Prefix of every output file of the pooled analysis.
FLEET_NAME = "sq1"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — per-mission data preparation
# ─────────────────────────────────────────────────────────────────────────────
def resolve_missions(missions) -> list:
    """Validate the requested mission ids against the workbook.

    Accepts "all" (or the single item "all") to mean every declared mission. Fails
    loudly and early on an unknown id, listing what the workbook does declare —
    cheaper than a confusing KeyError halfway through a pooled fit.
    """
    declared = pipeline.list_missions()
    requested = list(missions or MISSIONS)
    if [str(m).lower() for m in requested] == ["all"]:
        return declared

    unknown = [m for m in requested if m not in declared]
    if unknown:
        raise ValueError(
            f"Unknown mission id(s): {', '.join(unknown)}. The '{pipeline.MISSION_SHEET}' "
            f"sheet of {pipeline.workbook_path().name} declares: {', '.join(declared) or '(none)'}.")

    sheets = pipeline.sheet_names()
    missing_rbd = [m for m in requested if m not in sheets]
    if missing_rbd:
        raise ValueError(
            f"Mission(s) {', '.join(missing_rbd)} have no RBD sheet in "
            f"{pipeline.workbook_path().name} (a sheet must be named exactly as the mission id).")
    return requested


def prepare_mission(mission: str, *, reuse: bool = False) -> Path:
    """Build (or locate) <MISSION>/<MISSION>_components.csv and return its path.

    The workbook is the single source of truth, so the default is to REGENERATE:
    the components CSV is a derived artefact, not an input to hand-edit. It is also
    the file to skim for input typos — one row per component type, duplicate units
    merged, operational hours cumulated across them.

    Pass reuse=True to keep an existing file (e.g. to re-run only the calibration
    on unchanged data).
    """
    mdir = pipeline.COMPONENTS_DIR / mission
    rbd_csv = mdir / f"{mission}_rbd.csv"
    comp_csv = mdir / f"{mission}_components.csv"

    if reuse and comp_csv.exists():
        print(f"[{mission}] --reuse: keeping existing {comp_csv.name}")
        return comp_csv

    print(f"[{mission}] building from sheet '{mission}' of {pipeline.workbook_path().name}")
    pipeline.convert(mission, rbd_csv)
    pipeline.cluster(rbd_csv, comp_csv)
    return comp_csv


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run(missions=None, *, name: str = FLEET_NAME, reuse: bool = False,
        do_calibration: bool = True, do_sensitivity: bool = True) -> dict:
    """Run the full pooled analysis and return the calibration + sensitivity results."""
    missions = resolve_missions(missions)
    if not missions:
        raise ValueError("No missions to run — populate MISSIONS in sq1_update.py.")

    print("#" * 74)
    print(f"# POOLED SQ1 ANALYSIS — {len(missions)} mission(s): {', '.join(missions)}")
    print(f"# source: {pipeline.workbook_path().name}")
    print(f"# one shared alpha_0 and c; components kept separate per mission")
    print("#" * 74)

    # 1 — data
    csvs = [prepare_mission(m, reuse=reuse) for m in missions]

    # a single mission keeps its own name; several pool under <name>
    out_name = missions[0] if len(missions) == 1 else name
    out_dir = pipeline.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2 + 3 — calibration and sensitivity on the pooled fleet
    cal = sens = None
    if do_calibration:
        print()
        cal = calibration.run(csvs, name=out_name, out_dir=out_dir)
    if do_sensitivity:
        print()
        sens = sensitivity.run(csvs, name=out_name, out_dir=out_dir)

    print(f"\nDone. Pooled results for {', '.join(missions)} are in {out_dir}")
    return dict(missions=missions, name=out_name, out_dir=out_dir,
                components_csv=csvs, calibration=cal, sensitivity=sens)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the pipeline, the pooled conjugate calibration and the "
                    "sensitivity analysis over several missions at once.")
    ap.add_argument("missions", nargs="*",
                    help=f"mission ids, or 'all' (default: {', '.join(MISSIONS)})")
    ap.add_argument("--missions", dest="missions_opt", nargs="+", default=None,
                    help="same as the positional form; --missions all runs every "
                         "mission declared in the workbook")
    ap.add_argument("--list", action="store_true",
                    help="print the missions the workbook declares, then exit")
    ap.add_argument("--name", default=FLEET_NAME,
                    help=f"name of the pooled analysis (default: {FLEET_NAME})")
    ap.add_argument("--reuse", action="store_true",
                    help="keep the existing <M>_components.csv instead of "
                         "regenerating it from the workbook (default: rebuild)")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the scenario suite (much faster)")
    args = ap.parse_args(argv)

    if args.list:
        sheets = pipeline.sheet_names()
        print(f"{pipeline.workbook_path().name} declares:")
        for m in pipeline.list_missions():
            row = pipeline.mission_row(m)
            flag = "" if m in sheets else "   [!] no RBD sheet"
            print(f"  {m:<8} {str(row['mission name']).strip():<24} "
                  f"survived {pipeline.get_survived_hours(m):>10,.0f} h{flag}")
        extra = [s for s in sheets
                 if s not in (pipeline.MISSION_SHEET, pipeline.FAILURES_SHEET)
                 and s not in pipeline.list_missions()]
        if extra:
            print(f"  [!] RBD sheet(s) with no mission_list row: {', '.join(extra)}")
        return

    run(args.missions_opt or args.missions or MISSIONS, name=args.name,
        reuse=args.reuse,
        do_calibration=not args.no_calibration,
        do_sensitivity=not args.no_sensitivity)


if __name__ == "__main__":
    main()
