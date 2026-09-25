import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.step2.life_table import add_life_tab_col

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

def plot_lc_parameters(age_all, year_all):
    sex_codes = age_all["Sex Code"].unique()

    for sex_code in sex_codes:
        ax = age_all.loc[age_all['Sex Code'] == sex_code, "ax"]
        bx = age_all.loc[age_all['Sex Code'] == sex_code, "bx"]
        age = age_all.loc[age_all['Sex Code'] == sex_code, "age"]
        kt = year_all.loc[year_all['Sex Code'] == sex_code, "kt"]
        year = year_all.loc[year_all['Sex Code'] == sex_code, "year"]

        plt.figure(figsize=(8,6))

        plt.plot(
            age,
            ax
        )

        plt.xticks(np.arange(age.min(), age.max(), 5))

        plt.title(f"{ax.name} Plot: {sex_code}")
        plt.xlabel("Age")
        plt.ylabel(f"{ax.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))

        plt.plot(
            age,
            bx
        )

        plt.xticks(np.arange(age.min(), age.max(), 5))

        plt.title(f"{bx.name} Plot: {sex_code}")
        plt.xlabel("Age")
        plt.ylabel(f"{bx.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))

        plt.plot(
            year,
            kt
        )

        plt.xticks(np.arange(year.min(), year.max(), 5))

        plt.title(f"{kt.name} Plot: {sex_code}")
        plt.xlabel("Year")
        plt.ylabel(f"{kt.name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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

def ae_summary(ae_df):
    ae_df_2 = ae_df.copy()

    summary_list = []

    for (year, sex), df in ae_df_2.groupby(['Year Code', 'Sex Code']):
        AE_ratio_total = df['Deaths'].sum() / df['Expected Deaths'].sum()
        summary_list.append({'Year': year, 'Sex': sex, 'AE Ratio Total': AE_ratio_total})

    ae_summary_df = pd.DataFrame(summary_list)

    print(ae_summary_df)

def ex_progression_plot(single_age_life_table, ae_df, age=65):
    fitted_all_edited = (
        ae_df.drop(columns='Deaths')
        .rename(columns={'Expected Deaths': 'Deaths'})
        .copy()
    )

    fitted_life_table = add_life_tab_col(fitted_all_edited, RADIX=100000)

    for sex_code, df in fitted_life_table.groupby('Sex Code'):
        single_age_sex = single_age_life_table[
        (single_age_life_table['Sex Code']==sex_code)
        & (single_age_life_table['Single-Year Ages Code']==age)
        ]

        fitted_age = df[df['Single-Year Ages Code']==age]

        plt.figure(figsize=(10,6))

        plt.plot(
            fitted_age['Year Code'],
            fitted_age['ex'],
            label=f"Lee Carter Fitted e{age}",
            linestyle="--"
        )

        plt.plot(
            single_age_sex['Year Code'],
            single_age_sex['ex'],
            label=f"e{age}"
        )

        plt.title(f"Lee Carter Life Expectancy Progression\nAge: {age} Sex: {sex_code}")
        plt.xlabel("Year")
        plt.ylabel("Life Expectancy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
