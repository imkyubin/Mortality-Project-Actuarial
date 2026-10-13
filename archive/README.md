# Archive

Earlier work kept for reference. Nothing here is part of the active pipeline, and nothing in
`src/` or `scripts/` imports from it.

| Path | What it is | Status |
|---|---|---|
| `phase1_excel/Mortality_Workbook.xlsx` | Phase 1: life table, graduation (Gompertz, Makeham, moving average, Whittaker-Henderson) and a first Lee-Carter fit built by hand in Excel on 2015-2020 national data | Validation reference: Python outputs were spot-checked against its formulas |
| `phase1_excel/Mortality Table Project Workflow.docx` | Write-up of the Excel workflow | Reference |
| `phase1_prototype/Mortality_Table.py` | First Python script: survival curves and mortality plots from the Excel table | Historical; its data paths point at its original location and no longer resolve |
| `step1_baseline_model_prototype.py` | Step 1: the Excel workflow recreated in Python (life table, Gompertz, Makeham, MA and Whittaker-Henderson graduation, single-sex Lee-Carter and kt forecast) on 2015-2020 | Superseded by `src/actuarial/`, which holds the same logic generalized to every year and sex. Still runs: `python archive/step1_baseline_model_prototype.py` from the repo root (needs `data/processed/new_1999_2024.csv`) |
| `scratch/check.py` | One-off dtype check on the processed file | Scratch |
| `scratch/mortality_table_exploration.ipynb` | Notebook used only to confirm the Python environment worked | Scratch |

`phase1_excel/data/` holds the Phase 1 CDC WONDER extracts on the author's machine only; it is
gitignored like the rest of the raw data.
