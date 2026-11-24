"""
pipeline.py — Satellite reliability data pipeline (RBD → per-component clusters).

Prepares the inputs the conjugate calibration (calibration.py) consumes. Stages,
run in order by run():

    Step 1  build_rbd / convert
            mission RBD sheet  →  structured CSV with hierarchical IDs
    Step 2  build_clusters / cluster
            structured CSV  →  per-component clusters with failure-censored
            operational hours (using the `failures` sheet), and the handbook
            failure rate (FIT, read straight from the workbook).
            This <mission>_components.csv is the input to calibration.py.

────────────────────────────────────────────────────────────────────────────────
The workbook is the single source of truth
────────────────────────────────────────────────────────────────────────────────
Everything comes from ONE Excel file — whichever single .xlsx sits beside these
scripts, so the real workbook and the anonymised one are interchangeable:

    sheet "mission_list"  one row per mission — id, name, launch date, nominal
                          duration (h), survived duration (h), status
    sheet "failures"      one row per observed failure — mission, subsystem,
                          component, failure_date
    sheet "<MISSION ID>"  that mission's RBD, one sheet per mission, named exactly
                          as its `mission id` (e.g. "SC1"). Columns: Name (indented
                          with " - " to encode tree depth), FIT, Reliability, SPF,
                          Redundancy, Number of Elements, Req. Elements.

FIT is taken DIRECTLY from the workbook (failures per 1e9 hours) — the Reliability
column is never read, and no reliability→rate conversion happens anywhere in the
pipeline. A FIT of 0 means "no handbook data" and is treated as a missing prior.

────────────────────────────────────────────────────────────────────────────────

Each stage is split in two:
  • a PURE CORE  (build_rbd / build_clusters) — takes DataFrames plus explicit
    parameters and returns a DataFrame. No globals, no file I/O.
  • a FILE WRAPPER (convert / cluster) — loads inputs from disk, calls the core,
    writes the output CSV. Used by the default pipeline (run()).

Per-mission parameters (durations, launch date, failures) are bundled in an
injectable MissionContext, so a sweep just builds the upstream DataFrame once and
re-runs a core with a tweaked context. For example, sweeping the survived duration
(re-runs clustering only):

    rbd_df = build_rbd("SC1")
    ctx    = MissionContext.load("SC1")
    for ts in (50000, 52618, 55000):
        comps = build_clusters(rbd_df, replace(ctx, t_survived=ts))
"""

import re
import difflib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"                  # the workbook and the class map
RESULTS_DIR = BASE_DIR / "results"            # result JSONs and figures
COMPONENTS_DIR = RESULTS_DIR / "components"   # per-mission tables, rebuilt on every run

# The workbook is found, not hard-coded: the anonymised copy and the real one have
# the same format and differ only in filename, and only ever one of them sits in a
# given folder. WORKBOOK = None means "auto-detect in DATA_DIR at first use";
# assign a path to it (or pass workbook=... to any function) to override.
WORKBOOK = None
WORKBOOK_NAMES = ("Mission Data.xlsx", "Mission Data anon8.xlsx")

MISSION_SHEET  = "mission_list"   # one row per mission
FAILURES_SHEET = "failures"       # one row per observed failure

DATE_FORMAT = "%d-%m-%Y"   # date strings in the mission_list / failures sheets


# ─────────────────────────────────────────────────────────────────────────────
# Workbook access
# ─────────────────────────────────────────────────────────────────────────────
# Every sheet is read at most once per process: the mission list and the failures
# sheet are queried repeatedly (once per mission, and again by every sensitivity
# scenario), and re-parsing a 770-row workbook each time is pure waste.
_SHEET_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def find_workbook(root=None) -> Path:
    """Locate the mission workbook in `root` (default DATA_DIR).

    The real workbook and the anonymised one have identical formats and differ
    only in name, so the folder — not the filename — is the thing to point at.
    Exactly one workbook is expected; if several are present, one of the known
    names wins, and anything still ambiguous is an error rather than a guess.
    """
    root = Path(root).expanduser() if root is not None else DATA_DIR
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    # "~$..." are Excel lock files left behind by an open workbook, not workbooks
    found = sorted(p for p in root.glob("*.xlsx") if not p.name.startswith("~$"))
    if not found:
        raise SystemExit(
            f"no .xlsx found in {root}\n"
            f"expected one of {', '.join(WORKBOOK_NAMES)} (any name works, as long "
            f"as it is the only workbook in the folder).")
    if len(found) > 1:
        known = [p for p in found if p.name in WORKBOOK_NAMES]
        if len(known) != 1:
            listing = "\n  ".join(p.name for p in found)
            raise SystemExit(
                f"{len(found)} workbooks in {root}; leave only one, or name it one "
                f"of {', '.join(WORKBOOK_NAMES)}:\n  {listing}")
        return known[0].resolve()
    return found[0].resolve()


_AUTO_WORKBOOK: List[Path] = []   # one-slot cache for the auto-detected workbook


def _wb(workbook=None) -> Path:
    """Resolve the workbook argument, defaulting to the module-level WORKBOOK.

    Read at call time, NOT captured as a default argument value, so that
    reassigning pipeline.WORKBOOK (to point at a different workbook) actually
    takes effect everywhere. WORKBOOK = None auto-detects in DATA_DIR, and the
    result is cached so the folder is scanned once per process.
    """
    if workbook is not None:
        return Path(workbook)
    if WORKBOOK is not None:
        return Path(WORKBOOK)
    if not _AUTO_WORKBOOK:
        _AUTO_WORKBOOK.append(find_workbook())
    return _AUTO_WORKBOOK[0]


def workbook_path(workbook=None) -> Path:
    """The workbook that will actually be read — for messages and logging."""
    return _wb(workbook)


def read_sheet(sheet: str, workbook=None) -> pd.DataFrame:
    """Read one sheet of the mission workbook (cached), with headers stripped."""
    workbook = _wb(workbook)
    key = (str(workbook.resolve()), sheet)
    if key not in _SHEET_CACHE:
        if not workbook.exists():
            raise FileNotFoundError(f"Mission workbook not found: {workbook}")
        try:
            df = pd.read_excel(workbook, sheet_name=sheet, engine='openpyxl')
        except ValueError as exc:
            raise ValueError(
                f"Sheet '{sheet}' not found in {workbook.name}. Available sheets: "
                + ", ".join(sheet_names(workbook))) from exc
        df.columns = [str(c).strip() for c in df.columns]
        _SHEET_CACHE[key] = df
    return _SHEET_CACHE[key].copy()


def sheet_names(workbook=None) -> List[str]:
    """All sheet names in the workbook, in file order."""
    return pd.ExcelFile(_wb(workbook), engine='openpyxl').sheet_names


def list_missions(workbook=None) -> List[str]:
    """Mission ids declared in the `mission_list` sheet, in sheet order."""
    df = read_sheet(MISSION_SHEET, workbook)
    return [str(m).strip() for m in df['mission id'].dropna()]


def mission_row(mission: str, workbook=None) -> pd.Series:
    """The `mission_list` row for one mission id."""
    df = read_sheet(MISSION_SHEET, workbook)
    df['mission id'] = df['mission id'].astype(str).str.strip()
    row = df[df['mission id'] == str(mission).strip()]
    if row.empty:
        raise ValueError(
            f"Mission '{mission}' has no row in the '{MISSION_SHEET}' sheet of "
            f"{_wb(workbook).name}. Declared missions: {', '.join(list_missions(workbook))}")
    return row.iloc[0]


def _parse_date(value) -> pd.Timestamp:
    """Parse a workbook date cell — Excel hands back either a real datetime or a
    dd-mm-yyyy string (often with stray whitespace)."""
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, 'year'):
        return pd.Timestamp(value)
    return pd.to_datetime(str(value).strip(), format=DATE_FORMAT)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — RBD parsing:  Excel RBD sheet → structured rows with IDs & duty cycles
# ═════════════════════════════════════════════════════════════════════════════
#
# Each component receives an ID of the form ABC.X.X.X… where ABC is the mission
# code and each .X is the 1-indexed position within the parent's children.
#
# duty_cycle rules:
#   • Non-leaf nodes (have children): always 1.0
#   • Leaf whose parent is Serial or Parallel Active: 1.0
#   • Leaf whose parent is Parallel Passive:
#         position ≤ parent_m  →  1.0  (active units)
#         position >  parent_m  →  0.0  (cold-standby units)
# 
# Note: actual duty cycles not yet present in input data, to be inserted manually in later step.

@dataclass
class RBDNode:
    name:       str
    redundancy: str
    n:          int
    m:          int
    fit:        float                                    # failures per 1e9 h
    children:   List['RBDNode'] = field(default_factory=list)

    @property
    def is_leaf(self):
        return not self.children


def _count_dashes(name: str) -> int:
    """Count leading '-' characters to determine tree depth."""
    count = 0
    for ch in name:
        if ch == '-':   count += 1
        elif ch == ' ': continue
        else:           break
    return count


def _clean_name(name: str) -> str:
    return re.sub(r'^[\s\-]+', '', name).strip()


def _normalise_redundancy(r: str) -> str:
    r_l = r.lower()
    if 'active'  in r_l: return 'Parallel Active'
    if 'passive' in r_l: return 'Parallel Passive'
    # 'Series' and 'Serial' both occur across the workbook's sheets; they mean
    # the same thing, and downstream code tests the normalised label exactly.
    if 'serial'  in r_l or 'series' in r_l: return 'Serial'
    return r.strip()


def parse_rbd_sheet(df) -> List[RBDNode]:
    """Parse the flat indented RBD table into a tree of RBDNode objects."""
    raw_depths = [_count_dashes(str(row['Name'])) for _, row in df.iterrows()]
    fit_col = pd.to_numeric(df['FIT'], errors='coerce')
    nodes = [
        RBDNode(
            name       = _clean_name(str(row['Name'])),
            redundancy = _normalise_redundancy(str(row['Redundancy']))
                         if pd.notna(row['Redundancy']) else 'Serial',
            n          = int(row['Number of Elements'])
                         if pd.notna(row['Number of Elements']) else 0,
            m          = int(row['Req. Elements'])
                         if pd.notna(row['Req. Elements']) else 0,
            fit        = float(fit_col.iloc[i]) if pd.notna(fit_col.iloc[i]) else None,
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]

    # Build parent-child relationships using a depth stack
    stack: list = []   # list of (depth, node)
    roots: List[RBDNode] = []
    for node, depth in zip(nodes, raw_depths):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)
        else:
            roots.append(node)
        stack.append((depth, node))

    return roots


def build_rows(
    node:        RBDNode,
    parent_id:   str,           # e.g. "FFF.1.2"
    position:    int,           # 1-indexed position among siblings
    parent_type: str,           # children_type of the parent node
    parent_m:    int,           # required elements of the parent node
    forced_zero: bool = False,  # True when an ancestor already has duty_cycle=0
) -> List[dict]:
    """Recursively build one output row per node (assigns ID and duty cycle)."""
    node_id       = f"{parent_id}.{position}"
    children_type = node.redundancy          # how THIS node's children connect

    if forced_zero:
        duty_cycle = 0.0                     # ancestor is off → whole subtree off
    elif parent_type == 'Parallel Passive':
        duty_cycle = 1.0 if position <= parent_m else 0.0
    else:
        duty_cycle = 1.0                     # Serial / Parallel Active

    row = {
        'id':            node_id,
        'name':          node.name,
        'children_type': children_type,
        'n':             node.n,
        'm':             node.m,
        'duty_cycle':    duty_cycle,
        'fit':           node.fit,
    }

    child_forced_zero = duty_cycle == 0.0
    rows = [row]
    for i, child in enumerate(node.children, start=1):
        rows.extend(build_rows(
            child, node_id, i, children_type, node.m,
            forced_zero=child_forced_zero,
        ))
    return rows


def build_rbd(mission_code: str, workbook=None, sheet: str = None) -> pd.DataFrame:
    """PURE CORE: read the mission's RBD sheet and return the structured table.

    The RBD sheet is named after the mission id (pass `sheet` to override).
    """
    df    = read_sheet(sheet or mission_code, workbook)
    roots = parse_rbd_sheet(df)

    # Top-level nodes are connected in series (the system is a serial chain of
    # subsystems), so every root inherits duty_cycle = 1.0.
    all_rows = []
    for i, root in enumerate(roots, start=1):
        all_rows.extend(build_rows(root, mission_code, i, 'Serial', len(roots)))
    return pd.DataFrame(all_rows)


def structural_defects(rbd_df: pd.DataFrame) -> pd.DataFrame:
    """Nodes that declare child elements (n > 0) but carry no child rows.

    Such a node is not a leaf (leaves are n == 0), so build_clusters never sees its
    units: the component contributes no exposure and never reaches the fit. It is
    reported rather than repaired, because treating it as a leaf would change the
    fitted population.
    """
    parents = {i.rsplit('.', 1)[0] for i in rbd_df['id']}
    return rbd_df[(rbd_df['n'] > 0) & (~rbd_df['id'].isin(parents))]


def convert(mission_code: str, output_path=None, workbook=None,
            sheet: str = None) -> pd.DataFrame:
    """FILE WRAPPER: build the RBD table and write it to <mission>_rbd.csv."""
    if output_path is None:
        output_path = COMPONENTS_DIR / mission_code / f"{mission_code}_rbd.csv"
    output_path = Path(output_path)
    out = build_rbd(mission_code, workbook, sheet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Written {len(out)} components to {output_path}")
    for _, r in structural_defects(out).iterrows():
        print(f"[{mission_code}] WARNING: node {r['id']} ({r['name']}) declares "
              f"{int(r['n'])} elements but has no child rows; it is not a leaf, so its "
              f"units never reach the fit.")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Shared — per-mission parameters (durations, launch date, failures)
# ═════════════════════════════════════════════════════════════════════════════

def get_survived_hours(mission: str, workbook=None) -> float:
    """Return the survived duration (h) from the `mission_list` sheet.

    This is the mission's exposure window: how long it has actually flown. It is
    the only duration the model needs — the prior comes from the FIT column, so
    the nominal duration plays no part in any rate conversion.
    """
    row = mission_row(mission, workbook)
    survived = pd.to_numeric(row['survived duration (h)'], errors='coerce')
    if pd.isna(survived):
        raise ValueError(
            f"Mission '{mission}' has no numeric 'survived duration (h)' in the "
            f"'{MISSION_SHEET}' sheet.")
    return float(survived)


def get_nominal_hours(mission: str, workbook=None) -> float:
    """Return the nominal design duration (h) from the `mission_list` sheet.

    Reporting only (it is the window the handbook FITs were specified against);
    nothing in the calibration depends on it.
    """
    row = mission_row(mission, workbook)
    nominal = pd.to_numeric(row.get('nominal duration (h)'), errors='coerce')
    return float(nominal) if pd.notna(nominal) else float('nan')


def get_launch_date(mission: str, workbook=None) -> pd.Timestamp:
    """Return the launch date (as a Timestamp) from the `mission_list` sheet."""
    return _parse_date(mission_row(mission, workbook)['launch date'])


def load_failures(mission: str, workbook=None
                  ) -> Dict[Tuple[str, str], List[pd.Timestamp]]:
    """Read the `failures` sheet (one row per failure)
    → {(subsystem, component): [dates]} for one mission."""
    failure_map: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
    f = read_sheet(FAILURES_SHEET, workbook)
    if f.empty:
        return failure_map
    f['mission'] = f['mission'].astype(str).str.strip()
    for _, row in f[f['mission'] == str(mission).strip()].iterrows():
        key = (str(row['subsystem']).strip(), str(row['component']).strip())
        failure_map.setdefault(key, []).append(_parse_date(row['failure_date']))
    return failure_map


@dataclass
class MissionContext:
    """All per-mission parameters that the clustering & Bayesian stages depend on.

    Bundling these makes sensitivity sweeps easy: build a context with .load(),
    then pass a tweaked copy to a stage, e.g. dataclasses.replace(ctx, t_survived=…).
    """
    mission:     str
    t_survived:  float
    launch_date: pd.Timestamp
    t_nominal:   float = float('nan')   # reporting only — no rate conversion uses it
    failure_map: Dict[Tuple[str, str], List[pd.Timestamp]] = field(default_factory=dict)
    # Exposure window (start, end) in hours since launch; None = the whole mission,
    # launch to the reference date, which is what every analysis except the phase fits of SQ3 uses.
    # A window restricts BOTH the exposure and the failures counted to one phase of
    # the mission, e.g. (2160, None) = everything after the first 90 days. end=None
    # means the reference date.
    window:      Optional[Tuple[float, Optional[float]]] = None

    def exposure_window(self) -> Tuple[float, float]:
        """The window actually applied, clipped to [0, t_survived]."""
        if self.window is None:
            return 0.0, self.t_survived
        start, end = self.window
        a = max(0.0, float(start))
        b = self.t_survived if end is None else min(float(end), self.t_survived)
        return min(a, b), b

    @classmethod
    def load(cls, mission: str, workbook=None) -> "MissionContext":
        return cls(
            mission     = mission,
            t_survived  = get_survived_hours(mission, workbook),
            launch_date = get_launch_date(mission, workbook),
            t_nominal   = get_nominal_hours(mission, workbook),
            failure_map = load_failures(mission, workbook),
        )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clustering:  structured RBD rows → per-component clusters
# ═════════════════════════════════════════════════════════════════════════════

def canonical_name(name: str) -> str:
    """Strip trailing numeric suffixes so instances group together.

        'Battery Cell.1' -> 'Battery Cell'
        'Battery Cell 1' -> 'Battery Cell'
        'PCU.1.2'        -> 'PCU'
        'Star Tracker'   -> 'Star Tracker'
    """
    return re.sub(r'([.\s]+\d+)+$', '', name).strip()


def subsystem_id(component_id: str) -> str:
    """Return the top-2-level ID: 'FFF.1.2.3' -> 'FFF.1'."""
    parts = component_id.split('.')
    return '.'.join(parts[:2])


def single_fit(values):
    """Collapse a component type's per-unit FITs to one PER-UNIT prior rate.

    Each cluster is one component TYPE, and the likelihood pairs one rate with the
    cluster's CUMULATED operational hours, so the representative value must be a
    per-unit rate — an average, never a sum. Hence:
      - drop placeholder 0 entries when real (> 0) FITs are present, so one
        undocumented unit does not dilute the type's handbook rate;
      - average any remaining values (units of a type may carry slightly
        different handbook FITs);
      - all-zero or a single value pass through unchanged; empty -> NA.

    An all-zero cluster keeps FIT = 0, which calibration.py reads as "no handbook
    prior" and excludes.
    """
    vals = values.dropna()
    if len(vals) == 0:
        return pd.NA
    real = vals[vals > 0]
    use = real if len(real) > 0 else vals
    return round(float(use.mean()), 6)


def build_clusters(rbd_df: pd.DataFrame, ctx: MissionContext) -> pd.DataFrame:
    """PURE CORE: group leaf components and compute failure-censored exposure.

    Only leaf nodes (n == 0) are physical components; non-leaves are assemblies.

    Every unit of the same component type inside a subsystem is MERGED into one
    row: n_present counts the units, operational_hours is their CUMULATED exposure
    and `fit` is the single per-unit handbook rate (see single_fit). The resulting
    <mission>_components.csv is both the calibration input and the sheet to
    eyeball for input typos — a name misspelt in the RBD shows up as a stray
    one-unit cluster next to its correctly-spelt siblings.
    """
    leaves = rbd_df[rbd_df['n'] == 0].copy()
    leaves['subsystem_id']   = leaves['id'].apply(subsystem_id)
    leaves['component_name'] = leaves['name'].apply(canonical_name)

    # Subsystem names from top-level nodes (IDs with exactly one dot: FFF.1, …)
    top_level = rbd_df[rbd_df['id'].str.count(r'\.') == 1][['id', 'name']].rename(
        columns={'id': 'subsystem_id', 'name': 'subsystem_name'}
    )
    leaves = leaves.merge(top_level, on='subsystem_id', how='left')

    grouped = (
        leaves
        .groupby(['subsystem_id', 'subsystem_name', 'component_name'], sort=False)
        .agg(
            n_present =('id',         'count'),
            n_active  =('duty_cycle', lambda x: (x == 1.0).sum()),
            fit       =('fit',        single_fit),
        )
        .reset_index()
    )
    grouped.insert(0, 'mission', ctx.mission)

    # ── Failure lookup ────────────────────────────────────────────────────────
    # Keyed by (subsystem, CANONICAL component), so the failures sheet may name
    # either the component type ("Battery Cell") or the specific unit that failed
    # ("Battery Cell.2") — both resolve to the cluster that unit belongs to.
    # ALL rows for a component accumulate: a type that failed N times gets N dates.
    lookup: Dict[Tuple[str, str], List[pd.Timestamp]] = {}
    for (sub, comp), dates in ctx.failure_map.items():
        key = (str(sub).strip(), canonical_name(str(comp).strip()))
        lookup.setdefault(key, []).extend(dates)

    def failure_dates(row):
        return lookup.get(
            (str(row['subsystem_name']).strip(), str(row['component_name']).strip()), []
        )

    # ── Failure-censored operational hours, with cold-standby takeover ────────
    # Exposure is the time-integral of the number of OPERATING units of the type.
    # n_active units run from launch. A failure removes one operating unit; if a
    # cold spare is left (n_present - n_active of them), it takes over at that
    # instant and the operating count is restored, so no exposure is lost. Once
    # the spares are used up, each further failure permanently costs the remaining
    # window (t_survived - t_failure).
    #
    #     exposure = n_active * T  -  sum over the failures beyond the spares of
    #                                 (T - t_failure)
    #
    # A PRE-LAUNCH failure clamps to t = 0: the unit never flew, and its spare
    # took over from the start, so the type still accrues the full window. That is
    # what lets a pre-launch failure carry k = 1 with real exposure instead of the
    # degenerate k = 1, T = 0 that the Poisson likelihood cannot represent.
    #
    # With no spares (n_present == n_active) this reduces exactly to the plain
    # failure-censored sum: surviving units get the full window, each failed unit
    # only launch -> failure.
    #
    # EXPOSURE WINDOW. With ctx.window = (a, b) the same integral is taken over
    # [a, b] only, and only the failures inside it are counted ([0, b] when a = 0,
    # (a, b] otherwise, so consecutive windows partition the record). A failure
    # before a still uses up a spare, or removes its slot for the whole window if
    # none is left. The default window [0, T] reproduces the formula above exactly.
    a, b = ctx.exposure_window()

    def failure_times(row):
        # hours since launch, clamped to [0, T]; earliest first, so the spares
        # cover the failures that happened soonest
        T = ctx.t_survived
        return sorted(
            max(0.0, min((d - ctx.launch_date).total_seconds() / 3600.0, T))
            for d in failure_dates(row)
        )

    def operational_hours(row):
        n_act   = int(row['n_active'])
        n_spare = max(int(row['n_present']) - n_act, 0)

        hours = n_act * (b - a)
        for i, t in enumerate(failure_times(row)):
            if i < n_spare:
                continue                 # a cold spare takes over: nothing lost
            if t < b:
                hours -= b - max(t, a)   # no spare left: that slot stops accruing
        return max(0.0, hours)

    def failures_in_window(row):
        times = failure_times(row)
        if a == 0.0:
            return sum(1 for t in times if t <= b)
        return sum(1 for t in times if a < t <= b)

    grouped.insert(
        grouped.columns.get_loc('n_active') + 1,
        'operational_hours',
        grouped.apply(operational_hours, axis=1),
    )
    grouped.insert(
        grouped.columns.get_loc('operational_hours') + 1,
        'n_failures',
        grouped.apply(failures_in_window, axis=1),
    )

    # ── Input check: every failure row must land on a cluster ─────────────────
    # A failure whose (subsystem, component) matches nothing in the RBD is
    # otherwise dropped in silence — the failure simply never reaches the
    # likelihood. With failures this scarce, losing one to a spelling mismatch is
    # the single most damaging input error there is, so say so loudly and name the
    # nearest RBD component to make the fix obvious.
    present = set(zip(grouped['subsystem_name'].astype(str).str.strip(),
                      grouped['component_name'].astype(str).str.strip()))
    for key in (k for k in lookup if k not in present):
        sub, comp = key
        near = difflib.get_close_matches(
            comp, [c for s, c in present if s == sub], n=3, cutoff=0.6)
        hint = (f" Did you mean: {', '.join(near)}?" if near
                else (f" (no subsystem named '{sub}' in the RBD — subsystems are: "
                      f"{', '.join(sorted({s for s, _ in present}))})"
                      if not any(s == sub for s, _ in present) else ""))
        print(f"[{ctx.mission}] WARNING: {len(lookup[key])} failure(s) on "
              f"'{sub}' / '{comp}' match no component in the RBD and are IGNORED."
              f"{hint}")

    # matched failures over the whole record, whatever window is applied
    n_used = int(grouped.apply(lambda row: len(failure_dates(row)), axis=1).sum())
    n_given = sum(len(v) for v in lookup.values())
    if n_used != n_given:
        print(f"[{ctx.mission}] WARNING: {n_given - n_used} of {n_given} failure "
              f"row(s) were not matched to any component.")

    # `fit` is already the handbook prior rate (failures per 1e9 h), straight from
    # the workbook — nothing to derive. Move it to the end so the CSV reads
    # identity → counts → exposure → evidence → prior.
    return grouped[[c for c in grouped.columns if c != 'fit'] + ['fit']]


def cluster(input_csv, output_csv=None, ctx: MissionContext = None,
            workbook=None) -> pd.DataFrame:
    """FILE WRAPPER: cluster an RBD CSV into <mission>_components.csv.

    Pass an explicit ctx to override the mission parameters (e.g. for a sweep);
    otherwise it is loaded from the workbook.
    """
    rbd_df  = pd.read_csv(input_csv)
    mission = rbd_df['id'].iloc[0].split('.')[0]
    if ctx is None:
        ctx = MissionContext.load(mission, workbook)

    grouped = build_clusters(rbd_df, ctx)

    if output_csv is None:
        output_csv = Path(input_csv).parent / f"{mission}_components.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    merged = int(grouped['n_present'].sum()) - len(grouped)
    print(f"Written {len(grouped)} component clusters to '{output_csv}' "
          f"({merged} duplicate unit(s) merged, "
          f"{int((grouped['n_failures'] > 0).sum())} with failures)")
    return grouped


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════

def run(mission_id: str, *, do_rbd: bool = True, do_cluster: bool = True,
        workbook=None) -> Path:
    """Run the data pipeline for one mission, writing the two intermediate CSVs
    (<mission>/<mission>_rbd.csv and <mission>/<mission>_components.csv) and
    returning the components CSV path.

    Any stage can be switched off to reuse existing intermediates (e.g.
    do_rbd=False to re-run only the clustering step). Calibration of the priors
    and the Gamma-Poisson posteriors is done downstream in calibration.py.
    """
    mission_dir = COMPONENTS_DIR / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)

    rbd_csv        = mission_dir / f"{mission_id}_rbd.csv"
    components_csv = mission_dir / f"{mission_id}_components.csv"

    if do_rbd:
        convert(mission_id, rbd_csv, workbook)
    if do_cluster:
        cluster(rbd_csv, components_csv, workbook=workbook)
    return components_csv


if __name__ == '__main__':
    # Standalone: rebuild the CSVs for every mission declared in the workbook.
    # sq1_update.py is the entry point that selects which missions to analyse.
    for mission in list_missions():
        run(mission)
