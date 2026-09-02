import numpy as np

from src.actuarial.load_filter import load_filter
from src.actuarial.life_table import add_life_tab_col
from src.actuarial.eda import (
    plot_crude_mortality_surface,
    plot_progression_eda,
    plot_top_ex,
    plot_pct_heatmap
)
from src.actuarial.whittaker_henderson import (
    lambda_metrics,
    plot_lambda_elbow,
    fit_wh_all,
    wh_heatmap,
    wh_worst_fit,
    wh_worst_cumdev,
    plot_3worst_fit
)

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

