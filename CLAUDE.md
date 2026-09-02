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
`step_3_holistic_harmonized_model.limitations`.

**Actuarial layer (`src/actuarial/`)** builds life tables from cleaned deaths/population counts. The core
recursion (in both `src/actuarial/life_table.py` and, not-yet-deduplicated, `scripts/step1_baseline_model_prototype.py`)
is: `qx = Deaths/Population` → `px = 1-qx` → recursive cohort survivors `lx` (starting at a `RADIX`, e.g. 100000,
per `Year Code`/`Sex Code` group, walked in increasing age order) → `dx = qx*lx` → `Lx = lx - 0.5*dx` → `Tx`
(reverse-cumulative sum of `Lx` within each year/sex group) → `ex = Tx/lx`. Rows **must** be sorted by
year, sex, then increasing age before calling this (see `src/actuarial/load_filter.py:load_filter`), since the
`lx`/`Tx` recursions rely on row order, not on the age values themselves. `lee_carter.py` and `poisson_lee_carter.py`
are still stub files — **this is the active next task**: extract the reference implementation out of
`scripts/step1_baseline_model_prototype.py` into real, dataset-wide fits (see
`workflow.step_2_modern_single_age_model.models` in `config.yaml`); `forecasting.py` follows after those two are
done. Whichever model is treated as primary, validation must include an out-of-sample backtest — fit on a training
window, forecast `kt` forward, and compare the forecast against actual holdout years — not just in-sample fit
diagnostics; see `quality_bar.out_of_sample_backtest` in `config.yaml`. `whittaker_henderson.py` is complete: a working graduation-diagnostics module (fit every year/sex group at a
given lambda, a dataset-wide lambda elbow search, a residual heatmap, and an automatic worst-fit/worst-bias
ranking-and-plot layer) — see `workflow.step_2_modern_single_age_model.whittaker_henderson_utility` in `config.yaml`
for the current function list.

**Known data-quality trap**: CDC WONDER suppresses death counts under 10 as non-numeric placeholder text. Cleaning
currently uses `pd.to_numeric(errors="coerce")`, which silently turns suppressed cells into `NaN` — this can
propagate through the recursive `lx` calculation and corrupt an entire year/sex group's life table. This is not
yet explicitly detected or handled (`data.known_data_quality_risk` in `config.yaml`); be careful when adding logic
that assumes `Deaths`/`Population` are always clean numerics.

**Downstream layers are scaffolded but not implemented**: `src/api/` (FastAPI, meant to filter processed data/model
outputs by cause, age range, sex, location, year for the dashboard and reporting layer), `src/reporting/` (grounded
LLM commentary — must only narrate precomputed statistics plus a small curated context corpus, never invent numbers
or replace the model; all output must carry the disclaimer "AI-generated interpretation - actuarial review required
before client or external use"), and `src/visualization/` (plots + Power BI/Tableau-ready CSV exports). Read the
corresponding `ai_reporting`, `api_layer`, and `dashboard_strategy` sections of `config.yaml` before implementing
these — they specify constraints (e.g., the RAG pipeline is simple year-range/cause-tag corpus matching, not a
vector DB) that aren't visible from the empty stub files alone.

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

`src/actuarial/lee_carter.py` (SVD-based fit) currently has `lc_matrix` (returns `matrix, ages, years`
only — the `Deaths`/`Population` pivot matrices were dropped from it since nothing consumed them;
`ae_metric` re-derives Actual/Expected by merging back onto the source `df` instead of threading arrays
through), `fit_lee_carter`, `reconstruct_lc`, `build_lc_tables`, `fit_lc_all`, `ae_metric` (done), and a
first-draft `ae_heatmap` (currently broken — see below). `src/actuarial/poisson_lee_carter.py` is still
an empty import-only stub.

A prior session reportedly added a Newton-Raphson-calibrated `kt` (the standard Brouhns/Denuit/Vermunt
Poisson Lee-Carter MLE fit, which has no closed-form SVD solution and is solved iteratively) — none of
that exists in this repo or its git history as of 2026-08-28 (verified via `git diff`/`git stash list`,
both showed nothing matching); it was lost in an unsaved session. Treat it as **not yet done** rather
than assume it exists. (The A/E metric and e65 progression view from that same lost session are being
rebuilt now — see below — independent of whether the Poisson `kt` fit ever gets redone, since both work
fine off the existing SVD-based `kt`.)

### A/E and e65 workflow (non-forecasting), in dependency order

- `ae_metric(df, fitted_all)` — **done**. Merges `fit_lc_all`'s `fitted_all` onto `df` (renaming
  `age`/`year` to `Single-Year Ages Code`/`Year Code` first) to compute `Expected Deaths = Population *
  fitted_mx`, `AE Ratio = Deaths / Expected Deaths`, `AE Deviation = Deaths - Expected Deaths`.
- `ae_heatmap(ae_df)` — **drafted but broken**. Inside the per-`Sex Code` loop it computes
  `df_filtered['AE Scaled']` but then calls `df_model.pivot_table(..., values="AE Scaled", ...)` on the
  *unfiltered* frame, which never gets that column — raises `KeyError: 'AE Scaled'` (confirmed by running
  it against synthetic data on 2026-09-01). Fix: pivot `df_filtered`, not `df_model`; also assign
  `'AE Scaled'` via an explicit `.copy()` of the filtered slice to avoid the chained-assignment warning.
  Once fixed: Year × Age heatmap of `AE Ratio - 1` per sex, reusing `wh_heatmap`'s `TwoSlopeNorm(vcenter=0)`
  + `RdBu_r` pattern — this is the QA/diagnostic view, not the report-facing number (that's `ae_summary`).
- `ae_summary(ae_df, bins=None)` — **not started**. Group by `Year Code`/`Sex Code` (and age band if
  `bins` given), sum `Deaths` and `Expected Deaths` separately before dividing — summing first (rather than
  averaging per-cell ratios) avoids small-exposure ages/years dominating the aggregate.
- `e65_progression(df, fitted_all, sex_code, age=65)` — **not started**. Build a life table off the fitted
  mx surface via `add_life_tab_col` (from `life_table.py`) and compare `ex` at `age` against the actual
  (unsmoothed) `ex` already in `df`, merged into one frame keyed by `Year Code`. Must sort by
  `Year Code`/`Single-Year Ages Code` first — `add_life_tab_col`'s `lx`/`Tx` recursion depends on row
  order, not on age values.
- `plot_e65_progression(e65_df, sex_code, age=65)` — **not started**. `actual_ex` vs `fitted_ex` line plot
  over `Year Code`; historical comparison only, distinct from the forecast e65 fan chart below.
- `plot_lc_parameters(age_all, year_all, sex_code)` — **not started**. Sanity-check plot for the fit
  itself, currently missing entirely: `ax`/`bx` vs age (from `age_all`), `kt` vs year (from `year_all`),
  filtered to one sex. Should exist before trusting A/E or e65 numbers derived from the fit.
- `lc_variance_explained(matrix)` — **not started**. Needs the full SVD singular-value array `S`, which
  `fit_lee_carter` currently discards past `S[0]`. Decide whether `fit_lee_carter` should return `S` too,
  or whether this function reruns the SVD itself on `matrix`, before implementing — either choice touches
  `fit_lc_all`'s call site.

Planned visualization roadmap once the Poisson Lee-Carter fit exists, in rough priority order:
- **kt trajectory + forecast fan chart** — historical calibrated `kt` plus forecast with widening
  confidence bands; `scripts/step1_baseline_model_prototype.py`'s `plot_kt_forecast`/`forecast_kt` is the
  un-ported starting point, destined for `src/actuarial/forecasting.py` (currently empty).
- **e65 (or e0) fan chart** — life expectancy forecast propagated from the kt fan chart; this is the
  number that actually gets presented to a non-technical audience. (Distinct from the historical
  `e65_progression` above.)
- **Cumulative A/E deviation over time** — same spirit as `wh_worst_cumdev`, checking whether the
  Poisson-calibrated model systematically over/under-predicts deaths moving away from the fit window.
- **Out-of-sample backtest plot** — forecast vs. actual mx/ex for held-out years, required by
  `quality_bar.out_of_sample_backtest` in `config.yaml`; call out COVID 2020-2021 explicitly as expected
  divergence, not a bug.
- **Deviance-residual diagnostic** (heatmap or QQ-plot) — once fitting is Poisson-based, plain residuals
  are the wrong diagnostic; standardized deviance residuals validate the Poisson assumption itself.
