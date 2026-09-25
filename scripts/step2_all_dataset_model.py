import numpy as np

from src.step2.load_filter import load_filter
from src.step2.life_table import add_life_tab_col
from src.step2.eda import (
    plot_crude_mortality_surface,
    plot_progression_eda,
    plot_top_ex,
    plot_pct_heatmap
)
from src.step2.whittaker_henderson import (
    lambda_metrics,
    plot_lambda_elbow,
    fit_wh_all,
    wh_heatmap,
    wh_worst_fit,
    wh_worst_cumdev,
    plot_3worst_fit
)

from src.step2.lee_carter import fit_lc_all

from src.step2.poisson_lee_carter import fit_poisson_lc_all

from src.step2.lc_outputs import (
    plot_lc_parameters,
    ae_metric,
    ae_heatmap,
    ae_summary,
    ex_progression_plot
)

from src.step2.forecasting import (
    calculate_kt_changes,
    estimate_drift,
    kt_in_sample_fit,
    plot_kt_in_sample_fit,
    forecast_kt,
    plot_kt_forecast,
    project_ln_mx,
    forecast_ex_fan,
    plot_ex_fan_chart
)

FORECAST_YEARS = 10

# Load Datasets and Create New Life Table Columns
single_age_df = load_filter(
    "data/processed/new_1999_2024.csv", 
    age_col="Single-Year Ages Code", 
    min_age=20, 
    max_age=84
)

group_age_df = load_filter(
    "data/processed/harmonized_1968-2024.csv", 
    age_col="Age Group", 
    min_age=None, 
    max_age=None
)

single_age_life_table = add_life_tab_col(single_age_df)

grouple_age_life_table = add_life_tab_col(group_age_df)

# EDA (Single Age)
plot_crude_mortality_surface(single_age_life_table, 'T')
plot_crude_mortality_surface(single_age_life_table, 'M')
plot_crude_mortality_surface(single_age_life_table, 'F')

plot_progression_eda(single_age_life_table, 'T', column='ln_qx', start_year=None, gaps=5)
plot_progression_eda(single_age_life_table, 'M', column='ln_qx', start_year=None, gaps=5)
plot_progression_eda(single_age_life_table, 'F', column='ln_qx', start_year=None, gaps=5)

plot_progression_eda(single_age_life_table, 'T', column='Cumulative Survival', start_year=None, gaps=5)
plot_progression_eda(single_age_life_table, 'M', column='Cumulative Survival', start_year=None, gaps=5)
plot_progression_eda(single_age_life_table, 'F', column='Cumulative Survival', start_year=None, gaps=5)

plot_top_ex(single_age_life_table, 'T', plot_n=3)
plot_top_ex(single_age_life_table, 'M', plot_n=3)
plot_top_ex(single_age_life_table, 'F', plot_n=3)

plot_pct_heatmap(single_age_life_table, 'T')
plot_pct_heatmap(single_age_life_table, 'M')
plot_pct_heatmap(single_age_life_table, 'F')

# EDA (Age Group)
plot_crude_mortality_surface(grouple_age_life_table, 'T')
plot_crude_mortality_surface(grouple_age_life_table, 'M')
plot_crude_mortality_surface(grouple_age_life_table, 'F')

plot_progression_eda(grouple_age_life_table, 'T', column='ln_qx', start_year=None, gaps=5)
plot_progression_eda(grouple_age_life_table, 'M', column='ln_qx', start_year=None, gaps=5)
plot_progression_eda(grouple_age_life_table, 'F', column='ln_qx', start_year=None, gaps=5)

plot_progression_eda(grouple_age_life_table, 'T', column='Cumulative Survival', start_year=None, gaps=5)
plot_progression_eda(grouple_age_life_table, 'M', column='Cumulative Survival', start_year=None, gaps=5)
plot_progression_eda(grouple_age_life_table, 'F', column='Cumulative Survival', start_year=None, gaps=5)

plot_top_ex(grouple_age_life_table, 'T', plot_n=3)
plot_top_ex(grouple_age_life_table, 'M', plot_n=3)
plot_top_ex(grouple_age_life_table, 'F', plot_n=3)

plot_pct_heatmap(grouple_age_life_table, 'T')
plot_pct_heatmap(grouple_age_life_table, 'M')
plot_pct_heatmap(grouple_age_life_table, 'F')

# Whittaker Henderson Graduation (Single Age)
lambda_df = lambda_metrics(
    single_age_life_table, 
    lambda_grid=np.logspace(-2, 2, 5)
    )

plot_lambda_elbow(lambda_df)

whittaker_df = fit_wh_all(single_age_life_table)

wh_heatmap(whittaker_df, "T")
wh_heatmap(whittaker_df, "F")
wh_heatmap(whittaker_df, "M")

top_rss_df = wh_worst_fit(whittaker_df)
print(top_rss_df)
plot_3worst_fit(top_rss_df, whittaker_df)

top_cumdev_df = wh_worst_cumdev(whittaker_df)
print(top_cumdev_df)
plot_3worst_fit(top_cumdev_df, whittaker_df)

# Lee Carter dfs and Outputs
age_all, year_all, fitted_all = fit_lc_all(single_age_life_table)

plot_lc_parameters(age_all, year_all)

ae_df = ae_metric(single_age_life_table, fitted_all)

ae_heatmap(ae_df)
ae_summary(ae_df)

ex_progression_plot(single_age_life_table, ae_df, age=65)

# Forecasting (Lee Carter)
year_sorted = calculate_kt_changes(year_all)
drift_df = estimate_drift(year_sorted)

kt_fit_df = kt_in_sample_fit(year_sorted, drift_df)
plot_kt_in_sample_fit(kt_fit_df)

kt_forecast_df = forecast_kt(year_sorted, drift_df, FORECAST_YEARS)
plot_kt_forecast(year_sorted, kt_forecast_df)

ln_mx_projection = project_ln_mx(age_all, kt_forecast_df)

ex_fan_df = forecast_ex_fan(ln_mx_projection, age=65)
plot_ex_fan_chart(single_age_life_table, ex_fan_df, age=65)

# Poisson Lee Carter dfs and Outputs
age_all_plc, year_all_plc, fitted_all_plc = fit_poisson_lc_all(single_age_life_table, max_iter=100, tol=1e-8)

plot_lc_parameters(age_all_plc, year_all_plc)

ae_df_plc = ae_metric(single_age_life_table, fitted_all_plc)

ae_heatmap(ae_df_plc)
ae_summary(ae_df_plc)

ex_progression_plot(single_age_life_table, ae_df_plc, age=65)

# Forecasting (Poisson Lee Carter)
year_sorted_plc = calculate_kt_changes(year_all_plc)
drift_df_plc = estimate_drift(year_sorted_plc)

kt_fit_df_plc = kt_in_sample_fit(year_sorted_plc, drift_df_plc)
plot_kt_in_sample_fit(kt_fit_df_plc)

kt_forecast_df_plc = forecast_kt(year_sorted_plc, drift_df_plc, FORECAST_YEARS)
plot_kt_forecast(year_sorted_plc, kt_forecast_df_plc)

ln_mx_projection_plc = project_ln_mx(age_all_plc, kt_forecast_df_plc)

ex_fan_df_plc = forecast_ex_fan(ln_mx_projection_plc, age=65)
plot_ex_fan_chart(single_age_life_table, ex_fan_df_plc, age=65)
