"""Offline precompute for the dashboard's forecast page (page 3).

Run from the repo root: python -m scripts.step3.forecast_precompute

Fits both models (Poisson Lee-Carter primary, SVD for comparison) on both calibration
windows (1999-2024 and pre-COVID 1999-2019), forecasts to LAST_FORECAST_YEAR, runs the
1999-2014 / 2015-2024 backtest, and writes small CSVs to data/processed/model/. The page
only reads these files; nothing is fitted live (the Poisson fit is iterative).

CDC WONDER display rule: every file holds derived values only (parameters, ex, improvement
rates, A/E ratios, variance explained), never Deaths or Population.
Design: dashboard_strategy.primary.forecast_view_design in config.yaml.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.step2.load_filter import load_filter
from src.step2.life_table import add_life_tab_col
from src.step2.lee_carter import fit_lc_all, lc_matrix, lc_variance_explained
from src.step2.poisson_lee_carter import fit_poisson_lc_all
from src.step2.lc_outputs import ae_metric
from src.step2.forecasting import calculate_kt_changes, estimate_drift, forecast_kt, project_ln_mx, forecast_ex_fan
from src.step2.backtest import backtest, mean_abs_pct_error

PROCESSED = Path("data/processed")
OUT = PROCESSED / "model"

MODELS = {"Poisson": fit_poisson_lc_all, "SVD": fit_lc_all}
CALIBRATIONS = {"1999-2024": 2024, "1999-2019": 2019}
FIRST_YEAR = 1999
LAST_FORECAST_YEAR = 2050
BACKTEST_TRAIN_END = 2014
PRE_COVID_END = 2019

AGE = "Single-Year Ages Code"


def observed_improvement(life_table, start, end):
    """Average annual improvement per age from a log-linear trend of ln(qx) over start-end.

    1 - exp(slope): the same quantity Lee-Carter implies with 1 - exp(bx * drift), but read off
    the data directly. A regression slope is used rather than the two endpoint years, so one
    noisy year at a single age doesn't set the whole rate.
    """
    window = life_table[life_table["Year Code"].between(start, end)]
    rows = []
    for (sex_code, age), group in window.groupby(["Sex Code", AGE]):
        slope = np.polyfit(group["Year Code"], np.log(group["qx"]), 1)[0]
        rows.append({"Sex Code": sex_code, "Age": int(age), "Observed Improvement": 1 - np.exp(slope)})

    return pd.DataFrame(rows)


def fit_one(life_table, model, fit_fn, calibration, end_year):
    """Fit one model on one calibration window and forecast to LAST_FORECAST_YEAR."""
    train = life_table[life_table["Year Code"] <= end_year]
    age_all, year_all, fitted_all = fit_fn(train)

    year_sorted = calculate_kt_changes(year_all)
    drift_df = estimate_drift(year_sorted)
    kt_forecast_df = forecast_kt(year_sorted, drift_df, LAST_FORECAST_YEAR - end_year)
    ex_fan = forecast_ex_fan(project_ln_mx(age_all, kt_forecast_df), age=None)

    tags = {"Model": model, "Calibration": calibration}

    params = age_all.rename(columns={"age": "Age"}).merge(drift_df, on="Sex Code")
    params["LC Improvement"] = 1 - np.exp(params["bx"] * params["drift"])
    params = params.merge(observed_improvement(train, FIRST_YEAR, end_year), on=["Sex Code", "Age"])

    kt = pd.concat([
        year_all.assign(Type="Fitted"),
        kt_forecast_df.rename(columns={"kt_forecast": "kt"}).assign(Type="Forecast"),
    ], ignore_index=True).rename(columns={"year": "Year"})

    ae = ae_metric(train, fitted_all)
    ae_overall = ae.groupby("Sex Code").apply(
        lambda g: pd.Series({
            "In-sample AE": g["Deaths"].sum() / g["Expected Deaths"].sum(),
            "Mean Abs Cell Deviation": (g["AE Ratio"] - 1).abs().mean(),
        }), include_groups=False
    ).reset_index()

    summary = drift_df.merge(ae_overall, on="Sex Code")
    summary["Variance Explained"] = summary["Sex Code"].map(
        lambda s: lc_variance_explained(lc_matrix(train, s)[0])["Variance Explained"].iloc[0])

    ae_cells = ae[["Year Code", AGE, "Sex Code", "AE Ratio"]].rename(columns={"Year Code": "Year", AGE: "Age"})
    fan = ex_fan.rename(columns={"Year Code": "Year", AGE: "Age"})

    return {
        "params": params[["Sex Code", "Age", "ax", "bx", "LC Improvement", "Observed Improvement"]].assign(**tags),
        "kt": kt[["Sex Code", "Year", "kt", "kt_lower", "kt_upper", "Type"]].assign(**tags),
        "summary": summary.assign(**tags),
        "ae_in_sample": ae_cells.assign(**tags),
        "forecast_ex": fan[["Sex Code", "Age", "Year", "ex_forecast", "ex_lower", "ex_upper"]].assign(**tags),
    }


def run_precompute():
    OUT.mkdir(parents=True, exist_ok=True)

    single_age_df = load_filter(PROCESSED / "new_1999_2024.csv", age_col=AGE, min_age=20, max_age=84)
    life_table = add_life_tab_col(single_age_df)

    results = {}
    for model, fit_fn in MODELS.items():
        for calibration, end_year in CALIBRATIONS.items():
            print(f"Fitting {model} on {calibration}...")
            for name, frame in fit_one(life_table, model, fit_fn, calibration, end_year).items():
                results.setdefault(name, []).append(frame)

    backtest_ex, backtest_ae, backtest_err = [], [], []
    for model, fit_fn in MODELS.items():
        print(f"Backtesting {model}: train {FIRST_YEAR}-{BACKTEST_TRAIN_END}...")
        ex_compare, ae_year = backtest(life_table, fit_fn, BACKTEST_TRAIN_END)
        last_year = int(ex_compare["Year Code"].max())

        errors = mean_abs_pct_error(ex_compare, BACKTEST_TRAIN_END + 1, PRE_COVID_END).rename(
            columns={"MAPE": "MAPE Pre-COVID"}).merge(
            mean_abs_pct_error(ex_compare, BACKTEST_TRAIN_END + 1, last_year).rename(
                columns={"MAPE": "MAPE Full"}), on=["Sex Code", AGE])

        backtest_ex.append(ex_compare.assign(Model=model))
        backtest_ae.append(ae_year.assign(Model=model))
        backtest_err.append(errors.assign(Model=model))

    actual_ex = life_table[["Year Code", "Sex Code", AGE, "ex"]]

    outputs = {name: pd.concat(frames, ignore_index=True) for name, frames in results.items()}
    outputs["actual_ex"] = actual_ex
    outputs["backtest_ex"] = pd.concat(backtest_ex, ignore_index=True).drop(columns="Pct Error")
    outputs["backtest_ae"] = pd.concat(backtest_ae, ignore_index=True)
    outputs["backtest_error"] = pd.concat(backtest_err, ignore_index=True)

    for name, frame in outputs.items():
        frame = frame.rename(columns={"Year Code": "Year", AGE: "Age"})
        assert not {"Deaths", "Population", "Expected Deaths"} & set(frame.columns), name
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"Wrote {OUT / name}.csv ({len(frame):,} rows)")


if __name__ == "__main__":
    run_precompute()
