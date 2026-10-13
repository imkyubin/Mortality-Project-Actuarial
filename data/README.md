# Data

No data is committed. CDC WONDER data stays on your machine (every `*.csv` under `data/` is
gitignored), so after cloning you need to download the exports below and run the pipeline to
rebuild everything else.

```
data/
├── raw/             national CDC WONDER exports, exactly as downloaded
├── processed/       cleaned tables, the state x cause exports, credibility-validation outputs
│   └── model/       precomputed Lee-Carter results read by the dashboard's forecast page
└── outputs/         reserved for report tables and figures (not used yet)
```

## 1. National exports → `data/raw/`

From [CDC WONDER](https://wonder.cdc.gov/), export each query as a CSV with the default
footnotes (the loader, `src/data/wonder.py`, strips them) and save it under the exact file name
below. All four use: Show Totals, Show Zero Values and Show Suppressed on; rates per 100,000.

| File name | CDC WONDER dataset | Years selected | Group by |
|---|---|---|---|
| `Compressed Mortality, 1968-1978.csv` | Compressed Mortality, 1968-1978 | all | Year; Age Group; Sex |
| `Compressed Mortality, 1979-1998.csv` | Compressed Mortality, 1979-1998 | all | Year; Age Group; Sex |
| `Underlying Cause of Death, 1999-2020.csv` | Underlying Cause of Death, 1999-2020 | all | Year; Single-Year Ages; Sex |
| `Underlying Cause of Death, 2018-2024, Single Race.csv` | Underlying Cause of Death, 2018-2024, Single Race | 2021-2024 | Year; Single-Year Ages; Sex |

## 2. State x cause exports → `data/processed/`

The state and cause-of-death comparison uses eleven further exports from the Underlying Cause
of Death datasets (1999-2020, and 2018-2024 Single Race for 2021-2024), grouped by Year,
Single-Year Ages, Sex, State and the ICD-10 113 Cause List. They are saved in `data/processed/`
already reduced to these columns:
`Year Code, Single-Year Ages Code, Sex Code, State, Cause List, Deaths, Population`.
The step that pulled and reduced them is not part of this repository.

| File name | Locations | Causes | Years |
|---|---|---|---|
| `California.csv`, `Florida.csv`, `New York.csv`, `Texas.csv` | one state each | 8 causes | 1999-2020 |
| `State All Causes.csv` | the four states | All causes | 1999-2020 |
| `US 8 Causes.csv`, `US All Causes.csv` | United States | 8 causes / All causes | 1999-2020 |
| `State 8 Causes 2.csv`, `State All Causes 2.csv`, `US 8 Causes 2.csv`, `US All Causes 2.csv` | as above | as above | 2021-2024 |

The 8 causes, by their ICD-10 113 Cause List names: Diseases of heart, Malignant neoplasms,
Cerebrovascular diseases, Chronic lower respiratory diseases, Diabetes mellitus, Alzheimer
disease, Influenza and pneumonia, Chronic liver disease and cirrhosis.

## 3. Everything else is generated

Run from the repo root, in this order (the main [README](../README.md#run-the-pipeline) has the
same commands):

| Script | Reads | Writes to `data/processed/` |
|---|---|---|
| `scripts/national/clean_wonder_exports.py` | the 4 national exports | `old_1968_1998.csv`, `new_1999_2024.csv` |
| `scripts/national/harmonize_age_groups.py` | both files above | `harmonized_1968-2024.csv` (dashboard page 1) |
| `scripts/national/forecast_precompute.py` | `new_1999_2024.csv` | `model/*.csv` (dashboard page 3) |
| `scripts/sublevel/combine_exports.py` | the 11 state x cause exports | `final.csv` |
| `scripts/sublevel/validate_credibility.py` | `final.csv` | `single_age_final.csv`, `grouped_age_final.csv`, `validation_report.csv` (dashboard page 2), audit tables `audit_single_age.csv`, `audit_grouped_age.csv`, `dim_flags.csv` |

## Data use

CDC WONDER data is provided under its data use restrictions, which among other things bar
publishing counts of 9 or fewer (or rates based on them). This project goes further: the
data is never redistributed, and the dashboard shows no `Deaths` or `Population` values at all,
only derived rates, life-table values and ratios from cells that pass the credibility check
(at least 1,082 deaths in every year). The audit tables hold per-cell minimum death counts, some
under 10, and stay offline.

Source: Centers for Disease Control and Prevention, National Center for Health Statistics.
Compressed Mortality File 1968-1998 and Underlying Cause of Death 1999-2024, CDC WONDER Online
Database.
