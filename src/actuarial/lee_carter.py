import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

def reconstruct_lc(ax, bx, kt):
    fitted_ln_mx = ax[:, None] + np.outer(bx, kt)

    return fitted_ln_mx

def build_lc_tables(ax, bx, kt, fitted_ln_mx, ages, years, sex_code):
    age_df = pd.DataFrame({
        "age": ages,
        "ax": ax,
        "bx": bx,
        "Sex Code": sex_code
    })

    year_df = pd.DataFrame({
        "year": years,
        "kt": kt,
        "Sex Code": sex_code
    })

    fitted_wide = pd.DataFrame(
        fitted_ln_mx,
        index=ages,
        columns=years,
    )

    fitted_df = (
        fitted_wide
        .rename_axis(index="age", columns="year")
        .stack()
        .reset_index(name="fitted_ln_mx")
    )

    fitted_df["fitted_mx"] = np.exp(fitted_df["fitted_ln_mx"])
    fitted_df["Sex Code"] = sex_code

    return age_df, year_df, fitted_df

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

def ae_metric(df, fitted_all):
    fitted_all_edited = fitted_all.rename(columns={'age': 'Single-Year Ages Code', 'year': 'Year Code'}).copy()
    df_copy = df.copy()

    ae_df = pd.merge(
        df_copy, 
        fitted_all_edited, 
        on=['Single-Year Ages Code', 'Year Code', 'Sex Code'], 
        how='left'
        )

    ae_df['Expected Deaths'] = ae_df['Population'] * ae_df['fitted_mx']

    ae_df['AE Ratio'] = ae_df['Deaths'] / ae_df['Expected Deaths']

    ae_df['AE Deviation'] = ae_df['Deaths'] - ae_df['Expected Deaths']

    return ae_df

def ae_heatmap(ae_df):
    df_model = ae_df.copy()
    sex_codes = df_model["Sex Code"].unique()

    for sex_code in sex_codes:
        df_filtered = df_model[df_model["Sex Code"] == sex_code].copy()
        df_filtered['AE Scaled'] = df_filtered['AE Ratio'] - 1

        ln_mx_pivot = df_filtered.pivot_table(
                index="Single-Year Ages Code",
                columns="Year Code",
                values="AE Scaled",
                aggfunc="first"
            )
        
        max = ln_mx_pivot.max().max()
        min = ln_mx_pivot.min().min()

        norm = TwoSlopeNorm(vcenter=0, vmin=min, vmax=max)

        plt.figure(figsize=(10,6))

        im = plt.imshow(
            ln_mx_pivot.to_numpy(),
            cmap='RdBu_r',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        plt.xticks(
            ticks=range(0, len(ln_mx_pivot.columns), 5),
            labels=ln_mx_pivot.columns[::5].astype(int)
        )
        plt.yticks(
            ticks=range(len(ln_mx_pivot.index)),
            labels=ln_mx_pivot.index.astype(int)
        )
        
        plt.colorbar(im, label='Improvement Scale')
        
        plt.title(f'AE Heatmap ({sex_code})')
        plt.xlabel('Age')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.show()


    

    

    









