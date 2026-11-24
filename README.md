# Bayesian updating of handbook failure rates with in-orbit experience

Analysis code and data for the MSc thesis **Bayesian Updating of Reliability Predictions Using
Return of Experience Data** (L. Lizza, Delft University of Technology, carried out at ESA ESTEC,
2026). Every number in the thesis is produced by the scripts in this repository, which write their
result files to `results/` when they run.

## Research question

*How can Bayesian inference be applied to in-orbit Return of Experience data to systematically
assess and update the handbook failure rates used in spacecraft reliability predictions?*

- **SQ1.** How can in-orbit failure records update handbook failure rates at component level
  through Bayesian inference, and how robust are the updated rates to the specification of the
  prior and to the amount of in-orbit data?
- **SQ2.** To what extent do handbook failure rates updated with in-orbit experience predict the
  failures of a new mission more accurately than the handbook rates alone?
- **SQ3.** To what extent does treating the early part of a mission as a separate phase with its
  own constant failure rate improve the prediction of in-orbit failures, and how long should that
  early phase be?

## Method in brief

Each component type `i` has a handbook rate `lambda_hb,i` (FIT x 1e-9 per hour), an exposure
`T_i` (operating hours, failure-censored, cold spares accounted for) and `k_i` observed failures.

```
prior       lambda_i ~ Gamma(alpha_0, beta_i),   beta_i = alpha_0 / (c * lambda_hb,i)
likelihood  k_i      ~ Poisson(lambda_i * T_i)
posterior   lambda_i | k_i ~ Gamma(alpha_0 + k_i, beta_i + T_i)
```

Two scalars are shared by the whole fleet: the bias `c` (prior mean `c * lambda_hb,i`) and the
prior strength `alpha_0`. Both get weak priors (`log c, log alpha_0 ~ Normal(0, 1)`) and are
integrated over a 90 x 90 logarithmic grid using the negative-binomial marginal likelihood, so
each component's posterior is a finite mixture of Gamma distributions. Calibration is
deterministic grid integration and the predictive checks are ordinary Monte Carlo from
closed-form posteriors; there is no MCMC. Validation is leave-one-mission-out: both scalars are
calibrated on seven missions and the failure times of the eighth are scored by their log
predictive density. The two-phase model of SQ3 fits the same two scalars separately to an early
phase and to routine operations.

## Repository

```
data/                         inputs
  Mission Data anon8.xlsx     the anonymised failure record and reliability models
  component_classes.csv       functional class of every subsystem (appendix analysis)
results/                      created on the first run: result files (JSON) and figures
pipeline.py                   reads the workbook; builds each mission's component table
calibration.py                the model and the grid integration over c and alpha_0
sensitivity.py                sensitivity scenarios, leave-one-out, failure injection
sq1_*.py, sq2_*.py, sq3_*.py, appendix_*.py   one script per analysis, below
```

## Data

`data/Mission Data anon8.xlsx` holds the whole input: eight missions (`SC1`..`SC8`), 642
component types after merging identical units, 35 recorded failures and 320.1 million operating
hours.

| sheet | contents |
|---|---|
| `mission_list` | one row per mission: id, name, launch date, nominal duration (h), nominal end date, reference date, status, survived duration (h) |
| `failures` | one row per failure: mission, subsystem, component, failure date |
| `SC1` .. `SC8` | the mission's reliability block diagram: `Name` (indented with `" - "` per level), `FIT`, `Reliability` (not used), `SPF`, `Redundancy`, `Number of Elements`, `Req. Elements` |

The data are anonymised. Mission identities are replaced by `SC1`..`SC8`, every mission's
dates are shifted by its own offset (durations and failure ages are exact, calendar dates are
not), and payload instruments are named generically. The reliability block diagrams are used
only for unit counts and standby status, and hence exposure; no reliability above component
level is computed. A failure may be dated on or before launch day; such a unit counts as failed
and its cold spare is taken to have flown from launch.

## Reproducing the results

Python 3.9 or later.

```
pip install -r requirements.txt
```

Every script runs without arguments, reproduces the thesis exactly (seed 11) and writes its
result file to `results/`:

| script | result | thesis |
|---|---|---|
| `python sq1_update.py` | `sq1_calibration.json`, `sq1_sensitivity.json` | Ch4: the update, its sensitivity to the prior, leave-one-out, failure injection |
| `python sq1_data_ladder.py` | `sq1_ladder.json` | Ch4: sensitivity against the amount of data (178 subsets of missions) |
| `python sq1_evidence_horizon.py` | `sq1_horizon.json` | Ch4: simulated future evidence |
| `python sq2_validation.py` | `sq2_validation.json` | Ch5: leave-one-mission-out validation; Ch6: the two-phase model on missions left out |
| `python sq3_failure_timing.py` | `sq3_timing.json` | Ch6: trend test, age bands, predictive check, refits after a commissioning boundary |
| `python sq3_routine_sensitivity.py` | `sq3_routine_sensitivity.json` | Ch6: sensitivity of the routine-operations bias |
| `python appendix_robustness.py` | `appendix_robustness.json` | App C: exact tests, exposure errors, failure clustering, mission predictive check |
| `python appendix_classes.py` | `appendix_classes.json` | App C: handbook agreement by class of equipment (exploratory) |

The scripts are independent, except that `appendix_robustness.py` reads `sq2_validation.json`
and `sq3_timing.json`. `sq1_update.py` and `sq3_routine_sensitivity.py` take about half an hour
each, mostly for the leave-one-out jackknife; the others take minutes to a quarter of an hour.
Each script also writes its figures (PNG) to `results/`, and `pipeline.py` writes each mission's
component table to `results/components/`. Nothing in `results/` is tracked: run the scripts to
produce it.

## Headline results

| quantity | value | file |
|---|---|---|
| fleet bias `c`, whole record | 0.964 [0.627, 1.482] (median, 90% interval) | `sq1_calibration.json` |
| prior strength `alpha_0` | 0.206 [0.105, 0.461] | `sq1_calibration.json` |
| failures in the first 90 days | 10 of 35, against 0.93 at a constant rate | `sq3_timing.json` |
| bias in routine operations (after 90 days) | 0.563 [0.366, 0.913] | `sq3_timing.json` |
| update over handbook, per mission left out | +3.29 +/- 2.15 log units | `sq2_validation.json` |
| two-phase model, boundary chosen in each fold, over handbook | +4.67 +/- 1.85 log units per mission | `sq2_validation.json` |

## Citation

L. Lizza, *Bayesian Updating of Reliability Predictions Using Return of Experience Data*,
MSc thesis, Faculty of Aerospace Engineering, Delft University of Technology, 2026.
