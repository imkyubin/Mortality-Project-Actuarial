"""Out-of-sample backtest: fit on a training window, forecast kt forward, compare with the holdout.

quality_bar.out_of_sample_backtest in config.yaml. Works with either fit (fit_lc_all or
fit_poisson_lc_all), since both return the same age_all/year_all/fitted_all shape.
The dashboard trains on 1999-2014 so that 2015-2019 gives five pre-COVID holdout years
(dashboard_strategy.primary.forecast_view_design.backtest_window_note).
"""
import pandas as pd

from src.actuarial.forecasting import (
    calculate_kt_changes,
    estimate_drift,
    forecast_kt,
    project_ln_mx,
    forecast_ex_fan
)

KEYS = ['Year Code', 'Sex Code', 'Single-Year Ages Code']

def split_train_holdout(life_table, train_end):
    train = life_table[life_table['Year Code'] <= train_end]
    holdout = life_table[life_table['Year Code'] > train_end]

    return train, holdout

def backtest(life_table, fit_fn, train_end, **fit_kwargs):
    """Fit on years <= train_end, forecast every holdout year, return (ex_compare, ae_year).

    life_table needs Deaths, Population and the life-table columns (add_life_tab_col), so the
    actual ex in the holdout comes from the observed table, independent of the training fit.
    ex_compare: forecast ex with its 95% band against actual ex, per year x sex x age.
    ae_year: actual / expected deaths per holdout year and sex, summed over ages before dividing.
    """
    train, holdout = split_train_holdout(life_table, train_end)

    age_all, year_all, _ = fit_fn(train, **fit_kwargs)
    year_sorted = calculate_kt_changes(year_all)
    drift_df = estimate_drift(year_sorted)

    horizon = int(holdout['Year Code'].max()) - train_end
    kt_forecast_df = forecast_kt(year_sorted, drift_df, horizon)
    projection = project_ln_mx(age_all, kt_forecast_df)

    ex_compare = forecast_ex_fan(projection, age=None).merge(
        holdout[KEYS + ['ex']].rename(columns={'ex': 'ex_actual'}), on=KEYS
    )
    ex_compare['Pct Error'] = ex_compare['ex_forecast'] / ex_compare['ex_actual'] - 1

    expected = projection.rename(columns={'age': 'Single-Year Ages Code', 'year': 'Year Code'})
    ae_df = holdout.merge(expected[KEYS + ['forecast_mx']], on=KEYS)
    ae_df['Expected Deaths'] = ae_df['Population'] * ae_df['forecast_mx']

    ae_year = ae_df.groupby(['Year Code', 'Sex Code'], as_index=False)[['Deaths', 'Expected Deaths']].sum()
    ae_year['AE Ratio'] = ae_year['Deaths'] / ae_year['Expected Deaths']

    return ex_compare, ae_year[['Year Code', 'Sex Code', 'AE Ratio']]

def mean_abs_pct_error(ex_compare, start, end):
    """Mean absolute % error of forecast ex over holdout years start-end, per sex and age."""
    window = ex_compare[ex_compare['Year Code'].between(start, end)]

    return (
        window.assign(abs_error=window['Pct Error'].abs())
        .groupby(['Sex Code', 'Single-Year Ages Code'], as_index=False)['abs_error'].mean()
        .rename(columns={'abs_error': 'MAPE'})
    )
