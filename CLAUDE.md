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
`lx`/`Tx` recursions rely on row order, not on the age values themselves. `lee_carter.py`, `poisson_lee_carter.py`,
and `forecasting.py` are stub files — the reference implementation for those models currently lives only in
`scripts/step1_baseline_model_prototype.py` and has not been extracted into `src/` yet (tracked as `next_action`
under `workflow.step_1_baseline_model_prototype` in `config.yaml`). `whittaker_henderson.py` is the exception:
it's being actively built out as a reusable graduation utility for step 2 (fitting across every year/sex group in
the full modern dataset, not just step 1's single hand-picked slice) — see
`workflow.step_2_modern_single_age_model.whittaker_henderson_utility` in `config.yaml` for the function build
order and planned diagnostics (residual heatmap, lambda elbow plot, cumulative deviations plot).

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

**`src/eda/` (planned, not yet created)** is deliberately separate from `src/visualization/`: EDA plots (missingness/
suppression heatmap, crude pre-graduation mortality surface, mortality improvement rate, sex mortality gap, etc.)
serve the analyst's own diagnostic understanding and are sequenced before the rest of step 2's modeling work, while
`src/visualization/` is for polished, dashboard/export-ready output. Not every EDA plot belongs on the dashboard —
see `workflow.step_2_modern_single_age_model.eda` in `config.yaml` for the full plot list and which ones are
dashboard-facing versus internal-only.

**`Phase1_Hands_On/` and `archive/`** hold an earlier Excel-based prototype (workbook + workflow doc) used only as
a validation reference for spot-checking Python outputs against known-correct formulas — not part of the active
pipeline.

## Whittaker-Henderson build plan (step 2)

Working notes for finishing `src/actuarial/whittaker_henderson.py`; also recorded in `config.yaml`
`workflow.step_2_modern_single_age_model.whittaker_henderson_utility`.

**Status as of 2026-08-14**: Steps 1-5 of the build order below are functionally done, with one refinement — step
5 ended up split into two functions instead of one: `lambda_metrics(df, lambda_grid, pop_weights="n")` (computes
the `(Lambda, RSS, Smoothness)` table across every year/sex group, one row per lambda) and
`plot_lambda_elbow(lambda_df)` (draws it, each point labeled with its lambda via `plt.annotate`,
`textcoords="offset points"`). Known minor issue left in `plot_lambda_elbow`: the `xlabel`/`ylabel` text is
swapped relative to what's actually plotted (x-data is `Smoothness` but labeled `"RSS"`, and vice versa for y) —
not yet fixed. Log-scale axes (discussed as a readability improvement, since `lambda_grid` should be log-spaced)
also not yet added. Steps 6-7 (`plot_whittaker_residual_heatmap`, `plot_cumulative_deviations`, and deleting
`plot_whittaker_residual`) not yet started.

**Build order and dependencies**
1. Fix the import — line 5 needs to be `from matplotlib.colors import TwoSlopeNorm`, not
   `import matplotlib.colors import TwoSlopeNorm`. Nothing else in the file runs until this is fixed.
2. `_second_difference_matrix(n)` — unchanged. Reused in two places, not one: inside `fit_whittaker_henderson`'s
   `A` matrix, and again inside the elbow plot to score the smoothness of each candidate fit.
3. `fit_whittaker_henderson(df, year, sex_code, lambda_=1, pop_weights="n")` — unchanged. The shared building
   block; three other functions call it repeatedly, so nothing downstream should duplicate its logic.
4. `fit_wh_all(df, lambda_=1, pop_weights="n")` — write this next. Its output shape is what the heatmap and
   cumulative-deviations plot both consume, so lock it in before writing either.
5. `plot_lambda_elbow(df, year, sex_code, lambda_grid, pop_weights="n")` — independent of `fit_wh_all`, depends
   only on `fit_whittaker_henderson` + `_second_difference_matrix`. Worth building before committing to a
   `lambda_` default, since its output is what tells you whether `1` is even a reasonable default.
6. `plot_whittaker_residual_heatmap(whittaker_df, sex_code)` — needs `fit_wh_all`'s output shape decided first
   (step 4).
7. `plot_cumulative_deviations(whittaker_df)` — same single-slice input shape as the old
   `plot_whittaker_residual`. Write this as a direct swap for it, in the same pass as deleting it.
8. `plot_whittaker_fit(whittaker_df)` — leave untouched. Still the only function that shows the actual curve
   shape for one slice; nothing above supersedes it.

**Structure of each new/changed function**
- `fit_wh_all(df, lambda_=1, pop_weights="n")`: get distinct `(Year Code, Sex Code)` pairs present in `df`; loop
  over them calling `fit_whittaker_henderson(df, year, sex_code, lambda_, pop_weights)` per pair, collecting each
  returned slice; `pd.concat` into one long frame, same columns as a single-slice call, now spanning the full
  range; return that frame.
- `plot_lambda_elbow(df, year, sex_code, lambda_grid, pop_weights="n")`: for one representative slice, loop over
  each `lambda_` in `lambda_grid` — fit via `fit_whittaker_henderson`; Fit score = weighted residual sum of
  squares (`weights * whittaker_residual**2`, summed); Smoothness score = roughness penalty of the fitted values
  via `_second_difference_matrix(n) @ ln_qx_fitted_wh`, sum of squares; store `(lambda_, Fit, Smoothness)`. Plot
  Smoothness (x) vs Fit (y), points labeled by `lambda_` — the elbow/knee is the pick.
- `plot_whittaker_residual_heatmap(whittaker_df, sex_code)`: filter `fit_wh_all`'s output to one `sex_code` (one
  call per sex, matching the file's existing one-call-per-facet convention rather than subplots inside the
  function); pivot rows = `Year Code`, columns = `Single-Year Ages Code`, values = `whittaker_residual`;
  `TwoSlopeNorm(vcenter=0, vmin=…, vmax=…)` from the pivoted min/max; `imshow`/`pcolormesh` with a diverging
  colormap and that norm, colorbar labeled as the residual.
- `plot_cumulative_deviations(whittaker_df)`: one slice, already age-sorted (same input shape the old function
  took); `cumulative = whittaker_df["whittaker_residual"].cumsum()`; plot against age with a zero reference line
  (reuse the existing `axhline` pattern). Drift away from zero = systematic bias; hovering near zero = unbiased
  graduation.

**What to remove**
`plot_whittaker_residual` — cut it. It's a raw per-slice residual line plot, and everything it does is now
covered better elsewhere: the heatmap generalizes it across every slice at once, and `plot_cumulative_deviations`
answers the actual question actuaries use this diagnostic for (directional bias) more directly than eyeballing
scatter around zero.
