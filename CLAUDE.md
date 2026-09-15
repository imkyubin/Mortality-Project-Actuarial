# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An actuarial/data-science portfolio project built on CDC WONDER mortality data. It takes raw CDC WONDER mortality
exports through cleaning, harmonization, life-table construction, and mortality modeling (Lee-Carter family),
eventually surfacing results through a FastAPI layer, a Streamlit dashboard, and an LLM-based plain-English
reporting layer. The authoritative source of truth for project scope, current status, and design decisions is
`config/config.yaml` — read it before making architectural changes; it documents what's done, what's planned, and
*why* (rationale for each decision, including rejected alternatives).

The project is early-stage: `src/` has scaffolded modules (many are docstring-only stubs), the real prototyping
work currently lives in `scripts/`, and known data-quality/engineering gaps are tracked explicitly in
`config/config.yaml` (`quality_bar` and `data.known_data_quality_risk` sections) rather than in issue tracking.

## Commands

There is no build/lint/test tooling configured yet (`requirements.txt` and `tests/` are currently empty
placeholders — see `quality_bar.engineering_hygiene` in `config/config.yaml`). Run scripts directly with Python
from the repo root so `src` resolves as a package:

```
python -m scripts.data_cleaning_exploration
python -m scripts.data_harmonization
python scripts/step1_baseline_model_prototype.py
```

Dependencies observed in use (not yet pinned anywhere): `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scipy`.

**Git access**: Git is not on this machine's native PowerShell PATH (and isn't installed anywhere findable outside of
WSL). Run git commands through WSL Ubuntu instead — e.g. `wsl.exe -- git status`, or a WSL bash shell directly.
Treat WSL as the default terminal for any git operation (status, commit, push, log, diff) in this repo.

## Architecture

**Two-tier dataset design.** CDC WONDER's export schema changed over time, so raw data is kept in two parallel
tracks that only merge at the harmonized stage:
- *Old* grouped-age data (1968-1998, `Age Group` column, e.g. `'20-24 years'`) from `Compressed Mortality` exports.
- *New* single-year-age data (1999-2024, `Single-Year Ages Code` column, integer ages) from `Underlying Cause of
  Death` exports.

Pipeline: `data/raw/*.csv` → `src/data/wonder.py:load_wonder_csv` (strips CDC WONDER's footer/citation lines) →
`src/data/cleaning.py` (select relevant columns per era via `OLD_COLUMNS`/`NEW_COLUMNS`, drop grand-total rows,
normalize sex codes) → written to `data/processed/old_1968_1998.csv` and `data/processed/new_1999_2024.csv`. The
new (single-age) data is then bucketed into the old data's age groups via `src/data/harmonization.py`
(`map_single_age`, `add_harmonized_col`, `aggregate`) and concatenated into `data/processed/harmonized_1968-2024.csv`.
Scripts in `scripts/` (`data_cleaning_exploration.py`, `data_harmonization.py`) currently drive this pipeline
end-to-end and double as the only working examples of how the `src/data/` functions compose.

**Which processed file to use depends on the task**: `new_1999_2024.csv` (single-year ages, highest resolution,
1999-2024) is the primary modeling input; `harmonized_1968-2024.csv` (grouped ages, full 1968-2024 history) trades
age resolution for a longer time series, and mixes ICD-8/9/10 cause-of-death eras — see `config.yaml`
`step_3_holistic_harmonized_model.limitations`. The planned state/cause/race subsection comparison work (API,
dashboard, LLM reporting) uses `new_1999_2024.csv` exclusively; `harmonized_1968-2024.csv` is reserved for
general national historic-trend visualization only. See `config.yaml` `data_collection_strategy.resolved_decision`
for the finalized location/cause/race scope and why categories like Hispanic origin, Suicide, and Drug-induced
deaths were deferred rather than built into the first version.

**Actuarial layer (`src/actuarial/`)** builds life tables from cleaned deaths/population counts. The core
recursion (in both `src/actuarial/life_table.py` and, not-yet-deduplicated, `scripts/step1_baseline_model_prototype.py`)
is: `qx = Deaths/Population` → `px = 1-qx` → recursive cohort survivors `lx` (starting at a `RADIX`, e.g. 100000,
per `Year Code`/`Sex Code` group, walked in increasing age order) → `dx = qx*lx` → `Lx = lx - 0.5*dx` → `Tx`
(reverse-cumulative sum of `Lx` within each year/sex group) → `ex = Tx/lx`. Rows **must** be sorted by
year, sex, then increasing age before calling this (see `src/actuarial/load_filter.py:load_filter`), since the
`lx`/`Tx` recursions rely on row order, not on the age values themselves. `lee_carter.py` now has a working
SVD-based fit, fit diagnostics, and an Actual/Expected (A/E) validation layer (see the dedicated section below);
`poisson_lee_carter.py` now also has a working Newton-Raphson-calibrated Poisson MLE `kt` fit (`fit_poisson_lc_all`,
mirroring `lee_carter.py`'s `age_all`/`year_all`/`fitted_all` shape so it plugs into the same `lc_outputs.py`/
`forecasting.py` functions) — see `workflow.step_2_modern_single_age_model.lee_carter_utility` in `config.yaml`.
`forecasting.py` has drift-based `kt` forecasting with a 95% CI band (`forecast_kt`/`plot_kt_forecast`) and
propagates that band through to an e65/e-at-age forecast fan chart (`project_ln_mx` → `forecast_ex_fan` →
`plot_ex_fan_chart`), run for both the SVD and Poisson `kt` fits in `scripts/step2_all_dataset_model.py`.
Whichever model is treated as primary,
validation must include an out-of-sample backtest — fit on a training
window, forecast `kt` forward, and compare the forecast against actual holdout years — not just in-sample fit
diagnostics; see `quality_bar.out_of_sample_backtest` in `config.yaml`. `whittaker_henderson.py` is complete: a working graduation-diagnostics module (fit every year/sex group at a
given lambda, a dataset-wide lambda elbow search, a residual heatmap, and an automatic worst-fit/worst-bias
ranking-and-plot layer) — see `workflow.step_2_modern_single_age_model.whittaker_henderson_utility` in `config.yaml`
for the current function list.

**Known data-quality trap**: CDC WONDER suppresses death counts under 10 as the literal string `"Suppressed"`.
`src/data/cleaning.py:flag_suppressed` now flags this explicitly (adds a `{column} Suppressed` boolean column)
*before* `pd.to_numeric(errors="coerce")` turns it into an indistinguishable `NaN` — wired into
`scripts/data_cleaning_exploration.py` right after `combine_datasets`, so both processed files carry the flag.
It's a no-op today (national all-cause single-age counts don't hit the under-10 threshold), but it matters once
the state/cause/race pull below lands, since those combinations will suppress routinely. See
`data.known_data_quality_risk` in `config.yaml` for a related unfixed gap in `harmonization.py`'s `aggregate()`
(plain `Deaths` sum, no suppression propagation) that isn't triggered yet but would need fixing before that
function ever sees suppressible data.

**Downstream layers are scaffolded but not implemented**: `src/api/` (FastAPI, meant to filter processed data/model
outputs by cause, age range, sex, location, year for the dashboard and reporting layer), `src/reporting/` (grounded
LLM commentary — must only narrate precomputed statistics plus a small curated context corpus, never invent numbers
or replace the model; all output must carry the disclaimer "AI-generated interpretation - actuarial review required
before client or external use"), and `src/visualization/` (plots + Power BI/Tableau-ready CSV exports). Read the
corresponding `ai_reporting`, `api_layer`, and `dashboard_strategy` sections of `config.yaml` before implementing
these — they specify constraints (e.g., the RAG pipeline is simple year-range/cause-tag corpus matching, not a
vector DB, and `api_layer.credibility_gate` requires the API to flag or suppress any filter combination whose
underlying cell counts are CDC-suppressed or too small to trust, surfaced as an explicit message in the dashboard
rather than a broken or misleading chart) that aren't visible from the empty stub files alone.

**`src/actuarial/eda.py`** (the planned location was `src/eda/`; it ended up colocated with the other actuarial
modules instead, since it needs the same life-table columns) holds the analyst-facing diagnostics, deliberately
separate from `src/visualization/` (dashboard/export-ready output): `plot_crude_mortality_surface` (raw pre-
graduation mortality surface, sequential colormap), `plot_progression_eda` (mortality curve or survival curve by
age, one line per year — handles both dataset shapes), `top_n_ex`/`plot_top_ex` (ranks ages by life-expectancy
trend slope, plots the most-improving and most-declining automatically), and `pct_change_qx`/`plot_pct_heatmap`
(year-over-year percent mortality-improvement heatmap, diverging colormap). Still outstanding from the original
plan: the missingness/suppression heatmap (recommended to build first, not yet done) and the sex mortality gap
plot — see `workflow.step_2_modern_single_age_model.eda` in `config.yaml` for the full plot list and which ones
are dashboard-facing versus internal-only.

**`Phase1_Hands_On/` and `archive/`** hold an earlier Excel-based prototype (workbook + workflow doc) used only as
a validation reference for spot-checking Python outputs against known-correct formulas — not part of the active
pipeline.

## Whittaker-Henderson module (step 2) — current state

`src/actuarial/whittaker_henderson.py` is done; also recorded in `config.yaml`
`workflow.step_2_modern_single_age_model.whittaker_henderson_utility`. Function list, in dependency order:

- `_second_difference_matrix(n)` — roughness-penalty matrix, reused by the fit itself and by the lambda elbow.
- `fit_whittaker_henderson(df, year, sex_code, lambda_=1, pop_weights="n")` — single year/sex fit; the shared
  building block everything else calls. `pop_weights` uses **population share** (`Population/Population.sum()`
  within that slice), not raw population counts — raw counts made `lambda_`'s effective meaning swing wildly
  between `pop_weights="n"` and population-weighted, since the weight matrix's scale dominated the smoothing
  penalty otherwise.
- `fit_wh_all(df, lambda_=1, pop_weights="n")` — fits every `(Year Code, Sex Code)` group and concatenates into
  one long frame; the actual step 2 entry point.
- `lambda_metrics(df, lambda_grid, pop_weights="n")` / `plot_lambda_elbow(lambda_df)` — sweeps `lambda_grid`
  across the *whole* dataset (not one slice) via `fit_wh_all`, builds a `(Lambda, RSS, Smoothness)` table, and
  plots the Smoothness-vs-Fit elbow with each point labeled by its lambda. `RSS` here is also population-share
  weighted, for the same reason as above.
- `wh_heatmap(df, sex_code)` — Year × Age residual heatmap, one call per sex, `TwoSlopeNorm(vcenter=0)` +
  diverging colormap. The decided replacement for a per-slice residual line plot across the full 1999-2024 range.
- `plot_whittaker_fit(whittaker_df)` — single-slice observed-vs-fitted curve; kept for spot-checking one fit's
  actual shape, since nothing else in the module does that job.
- `wh_worst_fit(whittaker_df, n=10, pop_weights="n")` / `wh_worst_cumdev(whittaker_df, n=10, pop_weights="n")` —
  rank every `(Year Code, Sex Code)` group by population-share-weighted RSS or by max absolute cumulative
  deviation, returning the top `n` worst as a small ranking table. `plot_3worst_fit(top_df, whittaker_df,
  top_k=3)` then plots the top `top_k` of whichever ranking table it's given, reusing `plot_whittaker_fit` per
  slice — this replaced the originally-planned single-slice `plot_cumulative_deviations` function (never built)
  with an automatic multi-slice ranking-and-plot layer instead, once it became clear that manually picking one
  slice at a time didn't scale to the full dataset.

`scripts/step2_all_dataset_model.py` wires this (and `eda.py`, below) together end to end and is the working
reference for how these functions compose.

## Lee-Carter / Poisson Lee-Carter — current state and planned visualizations

`src/actuarial/lee_carter.py` (SVD-based fit) is now built out through the full non-forecasting A/E and
e-at-age workflow; also recorded in `config.yaml` `workflow.step_2_modern_single_age_model.lee_carter_utility`.
`src/actuarial/poisson_lee_carter.py` now also has a working Newton-Raphson-calibrated `kt` fit (the
standard Brouhns/Denuit/Vermunt Poisson Lee-Carter MLE, which has no closed-form SVD solution and is
solved iteratively): `death_exposure_matrix` pivots `Deaths`/`Population` into Year × Age matrices,
`initial_guess` seeds `ax`/`bx`/`kt` off the SVD fit's `ln(mx_hat)`, `update_ax`/`update_bx`/`update_kt`
are the per-parameter Newton-Raphson steps (score/Hessian off the Poisson log-likelihood), `normalize_param`
enforces the usual `sum(bx)=1`/`mean(kt)=0` identification constraints each iteration, and `fit_poisson_lc`/
`fit_poisson_lc_all` wrap the iteration loop and per-sex concatenation. `fit_poisson_lc_all` reuses
`lc_outputs.py`'s `reconstruct_lc`/`build_lc_tables`, so its `age_all`/`year_all`/`fitted_all` output has the
exact same shape as `lee_carter.py`'s SVD fit — every downstream function (`ae_metric`, `ex_progression_plot`,
everything in `forecasting.py` below) works unmodified on either fit's output, and
`scripts/step2_all_dataset_model.py` now runs the full pipeline for both.

Function list, in the order they appear in the file (fit pipeline → fit diagnostics → A/E workflow →
e-at-age comparison):

- `lc_matrix(df, sex_code)` — pivots `ln(qx)` into a Year × Age matrix for one sex; returns `matrix, ages,
  years` only (the `Deaths`/`Population` pivot matrices were dropped since nothing consumed them). Shared
  input to `fit_lee_carter` and `lc_variance_explained`.
- `fit_lee_carter(matrix)` / `reconstruct_lc(ax, bx, kt)` / `build_lc_tables(...)` / `fit_lc_all(df)` —
  the core SVD-based `ax`/`bx`/`kt` fit, wrapped per `Sex Code` group into long age/year/fitted tables
  across the whole dataset; `fit_lc_all` is the actual step 2 entry point.
- `plot_lc_parameters(age_all, year_all)` — **done**. Fit sanity-check: `ax` and `bx` vs age, `kt` vs
  year, one figure per parameter per sex (loops over every `Sex Code` in `age_all` itself, so it always
  plots all sexes — there is no `sex_code` filter argument). Deliberately placed right after `fit_lc_all`
  in the module, ahead of the A/E functions below, since those numbers shouldn't be trusted until this has
  been reviewed.
- `lc_variance_explained(matrix)` — **done**. Recenters `matrix` the same way `fit_lee_carter` does, reruns
  the SVD to get the full singular-value array `S` (rather than having `fit_lee_carter` return it, which
  would have touched `fit_lc_all`'s call site), and returns a `(Component, Variance Explained, Cumulative
  Variance)` table via `S**2 / sum(S**2)` — checks how much the single-component Lee-Carter approximation
  is actually capturing.
- `ae_metric(df, fitted_all)` — **done**. Merges `fit_lc_all`'s `fitted_all` onto `df` (renaming
  `age`/`year` to `Single-Year Ages Code`/`Year Code` first) to compute `Expected Deaths = Population *
  fitted_mx`, `AE Ratio = Deaths / Expected Deaths`, `AE Deviation = Deaths - Expected Deaths`.
- `ae_heatmap(ae_df)` — **done**. Year × Age heatmap of `AE Ratio - 1` per sex, reusing `wh_heatmap`'s
  `TwoSlopeNorm(vcenter=0)` + `RdBu_r` pattern — the QA/diagnostic view, not the report-facing number
  (that's `ae_summary`).
- `ae_summary(ae_df)` — **done**, simpler than originally planned: groups by `Year Code`/`Sex Code` only
  (no age-band `bins` parameter yet) and `print`s the result rather than returning a DataFrame. Sums
  `Deaths` and `Expected Deaths` separately before dividing — summing first (rather than averaging
  per-cell ratios) avoids small-exposure ages/years dominating the aggregate.
- `ex_progression_plot(single_age_life_table, ae_df, age=65)` — **done**, combining what was originally
  planned as two separate functions (`e65_progression` to build the comparison frame, `plot_e65_progression`
  to plot it) into one: derives a fitted life table from `ae_df`'s `Expected Deaths` via `add_life_tab_col`
  (`life_table.py`), then plots fitted `ex` against the actual (unsmoothed) `ex` at `age` for every sex in
  the data, one figure per sex. Historical comparison only, distinct from the forecast e65 fan chart below.

`src/actuarial/forecasting.py` — **done** through the kt forecast and e65 fan chart (ported and generalized
from `scripts/step1_baseline_model_prototype.py`'s single-sex `forecast_kt`/`plot_kt_forecast` prototype to
handle both sexes via per-`Sex Code` loops, and to work off either the SVD or Poisson `kt` fit):

- `calculate_kt_changes(year_all)` / `estimate_drift(year_sorted)` — per-sex `delta_kt = diff(kt)`, then
  per-sex drift (`mean`) and `sigma` (`std`) of that difference — the random-walk-with-drift parameters.
- `kt_in_sample_fit(year_sorted, drift_df)` / `plot_kt_in_sample_fit(kt_fit_df)` — extends the drift line
  back across the historical `kt` (sanity-check that the drift assumption isn't a bad fit) before ever
  forecasting forward with it.
- `forecast_kt(year_sorted, drift_df, forecast_years)` — projects `kt` forward per sex by `forecast_years`
  steps of `drift`, plus a 95% CI band (`kt_forecast ± 1.96 * sigma * sqrt(step)`) — the standard random-walk
  forecast interval, widening with the square root of the horizon.
- `plot_kt_forecast(year_sorted, kt_forecast_df)` — historical `kt` plus the forecast line and red dashed
  CI bounds, one figure per sex; the historical segment has no CI columns so the band only renders over the
  forecast horizon, matching the step1 prototype's behavior.
- `project_ln_mx(age_all, kt_forecast_df)` — reconstructs the full forecast mortality surface via the
  Lee-Carter identity `ln(mx) = ax + bx*kt`, for every age × forecast year × sex — run once each for the
  central `kt_forecast` and for the `kt_lower`/`kt_upper` CI bounds (`forecast_mx`, `forecast_mx_kt_low`,
  `forecast_mx_kt_high`). Since `bx`'s sign varies by age, `kt_lower` does not always correspond to the
  lower-mortality scenario at every age — that's resolved downstream rather than assumed here.
- `build_forecast_life_table(ln_mx_projection, mx_column)` — turns one of those three mx columns into a
  full projected life table by reusing `life_table.py`'s `add_life_tab_col` (setting `Population=1`,
  `Deaths=mx` so `qx` recovers `mx` directly, since there's no real forecast-year exposure to divide by).
- `forecast_ex_fan(ln_mx_projection, age=65)` — builds all three life tables (central, `kt_lower`,
  `kt_upper`), pulls `ex` at `age` from each, and takes elementwise `min`/`max` of the two bound scenarios
  to get `ex_lower`/`ex_upper` — rather than assuming the `kt_lower` scenario is the lower-`ex` one, since
  that direction flips wherever `bx` is negative at that age.
- `plot_ex_fan_chart(single_age_life_table, ex_fan_df, age=65)` — the report-facing chart: historical
  actual `ex` at `age`, the forecast central line, and a shaded `ex_lower`/`ex_upper` band, one figure per
  sex. This is the number that actually gets presented to a non-technical audience (distinct from the
  historical `ex_progression_plot` above).

Both are wired end-to-end for the SVD and Poisson `kt` fits in `scripts/step2_all_dataset_model.py`.

Still outstanding:
- **Cumulative A/E deviation over time** — same spirit as `wh_worst_cumdev`, checking whether the
  Poisson-calibrated model systematically over/under-predicts deaths moving away from the fit window.
- **Out-of-sample backtest plot** — forecast vs. actual mx/ex for held-out years, required by
  `quality_bar.out_of_sample_backtest` in `config.yaml`; call out COVID 2020-2021 explicitly as expected
  divergence, not a bug.
- **Deviance-residual diagnostic** (heatmap or QQ-plot) — once fitting is Poisson-based, plain residuals
  are the wrong diagnostic; standardized deviance residuals validate the Poisson assumption itself.
