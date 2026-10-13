"""Lee-Carter mortality model fitted by singular value decomposition (SVD).

ln m(x, t) = ax + bx * kt: ax is the average log death rate at age x, kt a single mortality
index over time, and bx how strongly each age follows it. ax is the row mean of the
ages x years log-rate matrix; bx and kt come from the first SVD component of what remains,
scaled so sum(bx) = 1 (mean(kt) = 0 follows from the centering). This least-squares fit weights
every age-year cell equally; poisson_lee_carter.py fits the same model to death counts instead.
"""

import numpy as np
import pandas as pd

from src.actuarial.lc_outputs import reconstruct_lc, build_lc_tables

def lc_matrix(df, sex_code):
    """Log death rates for one sex as an ages x years matrix; returns (matrix, ages, years).

    Uses ln(qx), where qx = Deaths / Population for single ages (the central death rate).
    """
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
    """Fit ax, bx, kt to an ages x years log-rate matrix by SVD; returns (ax, bx, kt)."""
    ax = np.mean(matrix, axis=1)
    centered = matrix - ax[:, None]

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    bx_raw = U[:, 0]
    kt_raw = S[0] * Vt[0, :]

    bx = bx_raw / np.sum(bx_raw)
    kt = kt_raw * np.sum(bx_raw)

    return ax, bx, kt

def fit_lc_all(df):
    """Fit Lee-Carter separately for every Sex Code in a single-age life table.

    Returns three long tables: age_all (age, ax, bx, Sex Code), year_all (year, kt, Sex Code) and
    fitted_all (age, year, fitted_ln_mx, fitted_mx, Sex Code). fit_poisson_lc_all returns the same
    shapes, so every downstream function accepts either fit.
    """
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
    """Share of variance in the centered log-rate matrix captured by each SVD component.

    Returns Component, Variance Explained (S**2 / sum(S**2)) and Cumulative Variance. The first
    row tests the single-factor assumption: how much of the age-time pattern one kt explains.
    """
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




