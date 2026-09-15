import numpy as np
import pandas as pd

from src.actuarial.lc_outputs import reconstruct_lc, build_lc_tables

def lc_matrix(df, sex_code):
    df_model = df[
        df["Sex Code"] == sex_code
    ].copy()

    df_model["ln_mx"] = np.log(df_model["qx"]).to_numpy()

    ln_mx_pivot = df_model.pivot_table(
        index="Single-Year Ages Code",
        columns="Year Code",
        values="ln_mx",
        aggfunc="first"
    )

    matrix = ln_mx_pivot.to_numpy()
    ages = ln_mx_pivot.index.to_numpy()
    years = ln_mx_pivot.columns.to_numpy()

    return matrix, ages, years

def fit_lee_carter(matrix):
    ax = np.mean(matrix, axis=1)
    centered = matrix - ax[:, None]

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    bx_raw = U[:, 0]
    kt_raw = S[0] * Vt[0, :]

    bx = bx_raw / np.sum(bx_raw)
    kt = kt_raw * np.sum(bx_raw)

    return ax, bx, kt

def fit_lc_all(df):
    sex_codes = df["Sex Code"].unique()

    age_list, year_list, fitted_list = [], [], []

    for sex_code in sex_codes:
        matrix, ages, years = lc_matrix(df, sex_code)
        ax, bx, kt = fit_lee_carter(matrix)
        fitted_ln_mx = reconstruct_lc(ax, bx, kt)
        age_df, year_df, fitted_df = build_lc_tables(ax, bx, kt, fitted_ln_mx, ages, years, sex_code)

        age_list.append(age_df)
        year_list.append(year_df)
        fitted_list.append(fitted_df)

    age_all = pd.concat(age_list, ignore_index=True)
    year_all = pd.concat(year_list, ignore_index=True)
    fitted_all = pd.concat(fitted_list, ignore_index=True)

    return age_all, year_all, fitted_all

def lc_variance_explained(matrix):
    ax = np.mean(matrix, axis=1)
    centered = matrix - ax[:, None]

    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    variance_explained = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(variance_explained)

    variance_df = pd.DataFrame({
        "Component": np.arange(1, len(S) + 1),
        "Variance Explained": variance_explained,
        "Cumulative Variance": cumulative_variance
    })

    return variance_df




