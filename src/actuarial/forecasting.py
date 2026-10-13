"""Forecast Lee-Carter mortality and turn it into projected life tables.

kt is projected as a random walk with drift: drift and sigma are the mean and standard
deviation of the yearly change in kt, and h years ahead kt = last kt + h * drift with a 95%
band of +/- 1.96 * sigma * sqrt(h). The band covers only this kt process error, not
uncertainty in ax, bx or the drift itself. Projected rates exp(ax + bx * kt) then go through
the life table to give ex and its band. Works on the output of either Lee-Carter fit.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.actuarial.life_table import add_life_tab_col

def calculate_kt_changes(year_all):
    """Sort year_all by sex and year and add delta_kt, the year-on-year change in kt."""
    year_sorted = year_all.sort_values(['Sex Code', 'year']).copy()
    year_sorted['delta_kt'] = year_sorted.groupby('Sex Code')['kt'].diff()

    return year_sorted

def estimate_drift(year_sorted):
    """Per sex: drift (mean of delta_kt) and sigma (standard deviation of delta_kt)."""
    drift_list = []

    for sex_code, group in year_sorted.groupby('Sex Code'):
        drift = group['delta_kt'].dropna().mean()
        sigma = group['delta_kt'].dropna().std()
        drift_list.append({'Sex Code': sex_code, 'drift': drift, 'sigma': sigma})

    drift_list_final = pd.DataFrame(drift_list)

    return drift_list_final

def kt_in_sample_fit(year_sorted, drift_df):
    """Draw the drift line through fitted kt from its first year, with residuals.

    A check that a straight-line trend is reasonable before forecasting with it.
    """
    fit_list = []

    for sex_code, group in year_sorted.groupby('Sex Code'):
        group = group.sort_values('year').copy()
        drift = drift_df.loc[drift_df['Sex Code'] == sex_code, 'drift'].iloc[0]

        first_kt = group['kt'].iloc[0]
        first_year = group['year'].iloc[0]

        group['kt_fitted'] = first_kt + (group['year'] - first_year) * drift
        group['residual'] = group['kt'] - group['kt_fitted']

        fit_list.append(group)

    fit_list_final = pd.concat(fit_list, ignore_index=True)

    return fit_list_final

def plot_kt_in_sample_fit(kt_fit_df):
    """Plot fitted kt against its drift line, one figure per sex."""
    for sex_code, group in kt_fit_df.groupby('Sex Code'):
        plt.figure(figsize=(8, 6))

        plt.plot(group['year'], group['kt'], label='kt (actual)')
        plt.plot(group['year'], group['kt_fitted'], linestyle='--', label='kt (drift line)')

        plt.title(f"kt In-Sample Drift Fit: {sex_code}")
        plt.xlabel("Year")
        plt.ylabel("kt")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def forecast_kt(year_sorted, drift_df, forecast_years):
    """Project kt `forecast_years` steps past the last fitted year, per sex, with a 95% band.

    Returns year, kt_forecast, kt_lower, kt_upper and Sex Code.
    """
    forecast_list = []

    for sex_code, group in year_sorted.groupby('Sex Code'):
        group = group.sort_values('year')
        drift = drift_df.loc[drift_df['Sex Code'] == sex_code, 'drift'].iloc[0]
        sigma = drift_df.loc[drift_df['Sex Code'] == sex_code, 'sigma'].iloc[0]

        last_kt = group['kt'].iloc[-1]
        last_year = group['year'].iloc[-1]

        steps = np.arange(1, forecast_years + 1)
        standard_error = sigma * np.sqrt(steps)
        kt_forecast = last_kt + steps * drift

        forecast_list.append(pd.DataFrame({
            'year': last_year + steps,
            'kt_forecast': kt_forecast,
            'kt_lower': kt_forecast - 1.96 * standard_error,
            'kt_upper': kt_forecast + 1.96 * standard_error,
            'Sex Code': sex_code
        }))

    forecast_list_final = pd.concat(forecast_list, ignore_index=True)

    return forecast_list_final

def plot_kt_forecast(year_sorted, kt_forecast_df):
    """Plot historical kt followed by the forecast and its 95% band, one figure per sex."""
    columns = ['year', 'kt', 'kt_lower', 'kt_upper']

    for sex_code, hist_group in year_sorted.groupby('Sex Code'):
        forecast_group = kt_forecast_df[kt_forecast_df['Sex Code'] == sex_code]
        forecast_renamed = forecast_group.rename(columns={'kt_forecast': 'kt'})

        combined = pd.concat(
            [hist_group.reindex(columns=columns), forecast_renamed.reindex(columns=columns)],
            ignore_index=True
        )

        plt.figure(figsize=(8, 6))

        plt.plot(combined['year'], combined['kt'], label='kt')
        plt.plot(combined['year'], combined['kt_lower'], color='red', linestyle='--', label='95% CI')
        plt.plot(combined['year'], combined['kt_upper'], color='red', linestyle='--')

        plt.title(f"kt Forecast: {sex_code}")
        plt.xlabel("Year")
        plt.ylabel("kt")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def project_ln_mx(age_all, kt_forecast_df):
    """Forecast rates ax + bx * kt for every age, forecast year and sex, at the central kt and both band edges.

    Adds forecast_mx, forecast_mx_kt_low and forecast_mx_kt_high (with their logs). Where bx < 0
    the low-kt edge gives higher mortality, so forecast_ex_fan sorts the edges out afterwards.
    """
    projection_df = age_all.merge(kt_forecast_df, on='Sex Code', how='left')

    projection_df['forecast_ln_mx'] = (
        projection_df['ax'] + projection_df['bx'] * projection_df['kt_forecast']
    )
    projection_df['forecast_mx'] = np.exp(projection_df['forecast_ln_mx'])

    projection_df['forecast_ln_mx_kt_low'] = (
        projection_df['ax'] + projection_df['bx'] * projection_df['kt_lower']
    )
    projection_df['forecast_mx_kt_low'] = np.exp(projection_df['forecast_ln_mx_kt_low'])

    projection_df['forecast_ln_mx_kt_high'] = (
        projection_df['ax'] + projection_df['bx'] * projection_df['kt_upper']
    )
    projection_df['forecast_mx_kt_high'] = np.exp(projection_df['forecast_ln_mx_kt_high'])

    return projection_df

def build_forecast_life_table(ln_mx_projection, mx_column='forecast_mx'):
    """Life table from one projected rate column (Population = 1, Deaths = rate, so qx equals the rate)."""
    forecast_lt = ln_mx_projection.rename(
        columns={'age': 'Single-Year Ages Code', 'year': 'Year Code'}
    ).copy()

    forecast_lt['Population'] = 1
    forecast_lt['Deaths'] = forecast_lt[mx_column]

    forecast_lt = forecast_lt.sort_values(['Year Code', 'Sex Code', 'Single-Year Ages Code'])

    complete_forecast_lt = add_life_tab_col(forecast_lt, RADIX=100000)

    return complete_forecast_lt

def forecast_ex_fan(ln_mx_projection, age=65):
    # age=None keeps every age (one fan per age), used by the backtest and the dashboard precompute.
    """Projected ex at `age` (every age if None) with its 95% band, per forecast year and sex.

    Builds life tables for the central and both band-edge scenarios and takes the elementwise min
    and max of the two edges as ex_lower and ex_upper.
    """
    central_lt = build_forecast_life_table(ln_mx_projection, 'forecast_mx')
    kt_low_lt = build_forecast_life_table(ln_mx_projection, 'forecast_mx_kt_low')
    kt_high_lt = build_forecast_life_table(ln_mx_projection, 'forecast_mx_kt_high')

    keys = ['Year Code', 'Sex Code', 'Single-Year Ages Code']

    def _ex_at_age(life_table, label):
        rows = life_table if age is None else life_table[life_table['Single-Year Ages Code'] == age]
        return rows[keys + ['ex']].rename(columns={'ex': label})

    ex_fan_df = (
        _ex_at_age(central_lt, 'ex_forecast')
        .merge(_ex_at_age(kt_low_lt, 'ex_kt_low'), on=keys)
        .merge(_ex_at_age(kt_high_lt, 'ex_kt_high'), on=keys)
    )

    ex_fan_df['ex_lower'] = ex_fan_df[['ex_kt_low', 'ex_kt_high']].min(axis=1)
    ex_fan_df['ex_upper'] = ex_fan_df[['ex_kt_low', 'ex_kt_high']].max(axis=1)

    ex_fan_df_final = ex_fan_df.drop(columns=['ex_kt_low', 'ex_kt_high'])

    return ex_fan_df_final

def plot_ex_fan_chart(single_age_life_table, ex_fan_df, age=65):
    """Plot actual ex at `age`, the central forecast and its shaded 95% band, one figure per sex."""
    for sex_code, group in ex_fan_df.groupby('Sex Code'):
        historical = single_age_life_table[
            (single_age_life_table['Sex Code'] == sex_code)
            & (single_age_life_table['Single-Year Ages Code'] == age)
        ]

        group_sorted = group.sort_values('Year Code')

        plt.figure(figsize=(10, 6))

        plt.plot(
            historical['Year Code'],
            historical['ex'],
            label=f"e{age} (actual)"
        )

        plt.plot(
            group_sorted['Year Code'],
            group_sorted['ex_forecast'],
            label=f"e{age} (forecast)",
            linestyle="--"
        )

        plt.fill_between(
            group_sorted['Year Code'],
            group_sorted['ex_lower'],
            group_sorted['ex_upper'],
            color="gray",
            alpha=0.3,
            label="95% CI"
        )

        plt.title(f"e{age} Forecast Fan Chart: {sex_code}")
        plt.xlabel("Year")
        plt.ylabel("Life Expectancy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
