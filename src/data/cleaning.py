"""Standardize raw mortality data into the project schema.

Purpose:
    Convert CDC WONDER columns into consistent names and types such as:
    year, age, sex, deaths, population, mortality_rate.

Actuarial note:
    The cleaned table should keep deaths and exposure/population separate.
    Poisson Lee-Carter needs count data, not only precomputed rates.
"""

import pandas as pd

OLD_COLUMNS = ["Year Code", "Age Group", "Sex Code", "Deaths", "Population"]

NEW_COLUMNS = [
    "Year Code",
    "Single-Year Ages Code",
    "Sex Code",
    "Deaths",
    "Population",
]


def remove_grand_totals(df, age_column):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df[
        cleaned_df[age_column].notna()
        & (cleaned_df[age_column] != "")
    ]

    return cleaned_df


def filter_by_sex(df, sex_column, sex):
    cleaned_df = df.copy()

    if sex =="M":
        return cleaned_df[cleaned_df[sex_column] == "M"]
    
    if sex =="F":
        return cleaned_df[cleaned_df[sex_column] == "F"]
    
    if sex =="T":
        return cleaned_df[cleaned_df[sex_column].isna()
                          | (cleaned_df[sex_column] == "")
        ]

def fill_na_T(df, sex_column):
    cleaned_df = df.copy()

    cleaned_df.loc[
        cleaned_df[sex_column].isna() | (cleaned_df[sex_column] == ""),
        sex_column
    ] = "T"

    return cleaned_df

def select_columns(df, columns):
    return df[columns].copy()

def combine_datasets(dfs):
    return pd.concat(dfs, ignore_index=True)