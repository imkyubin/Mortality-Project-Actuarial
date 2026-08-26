import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

def plot_crude_mortality_surface(df, sex_code):
    filtered_df = df[df["Sex Code"]==sex_code].copy()
    filtered_df["ln_qx"] = np.log(filtered_df["qx"])

    if "Single-Year Ages Code" in filtered_df.columns:
        pivot = filtered_df.pivot(
            index='Year Code',
            columns='Single-Year Ages Code',
            values='ln_qx'
        )

        max = pivot.max().max()
        min = pivot.min().min()

        norm = Normalize(vmin=min, vmax=max)

        plt.figure(figsize=(10,6))

        im = plt.imshow(
            pivot.to_numpy(),
            cmap='viridis',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        plt.xticks(
            ticks=range(0, len(pivot.columns), 5),
            labels=pivot.columns[::5].astype(int)
        )
        plt.yticks(
            ticks=range(len(pivot.index)),
            labels=pivot.index.astype(int)
        )
        
        plt.colorbar(im, label='LN Mortality')
        
        plt.title(f'Single Year Crude Mortality Surface ({sex_code})')
        plt.xlabel('Age')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.show()

    elif "Age Group" in filtered_df.columns:
        pivot = filtered_df.pivot(
            index='Year Code',
            columns='Age Group',
            values='ln_qx'
        )

        max = pivot.max().max()
        min = pivot.min().min()

        norm = Normalize(vmin=min, vmax=max)

        plt.figure(figsize=(10,6))

        im = plt.imshow(
            pivot.to_numpy(),
            cmap='viridis',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        plt.xticks(
            ticks=range(len(pivot.columns)),
            labels=pivot.columns
        )
        plt.yticks(
            ticks=range(len(pivot.index)),
            labels=pivot.index.astype(int)
        )
        
        plt.colorbar(im, label='LN Mortality')
        
        plt.title(f'Age Group Crude Mortality Surface ({sex_code})')
        plt.xlabel('Age')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.show()

def plot_progression_eda(df, sex_code, column, start_year=None, gaps=1):
    if start_year is None:
        min_year = df['Year Code'].min()
    else:
        min_year = start_year

    filtered_df = df[
        (df['Sex Code']==sex_code)
        & ((df['Year Code']-min_year) % gaps == 0)
    ].copy()

    filtered_df['ln_qx'] = np.log(filtered_df['qx'])

    max_year = filtered_df['Year Code'].max()

    if column == 'ln_qx':
        metric = 'LN Mortality'
    elif column == 'Cumulative Survival':
        metric = 'Survival'

    age_col = "Single-Year Ages Code" if "Single-Year Ages Code" in filtered_df.columns else "Age Group"

    type = "Single Year" if "Single-Year Ages Code" in filtered_df.columns else "Harmonized"

    plt.figure(figsize=(10,6))

    for year, year_df in filtered_df.groupby(['Year Code']):
        plt.plot(
        year_df[age_col],
        year_df[column],
        label=f"{year}"
        )

    plt.title(f"{type} {metric} Progression\n{sex_code}, {min_year} - {max_year}")
    plt.xlabel("Age")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def top_n_ex(df, sex_code, df_n=5):
    if 'Single-Year Ages Code' in df.columns:
        top_df = df[df['Sex Code'] == sex_code][['Single-Year Ages Code', 'Year Code', 'ex']].copy()

        slope_list = []

        for age, age_df in top_df.groupby(['Single-Year Ages Code']):
            age_df = age_df.sort_values(['Year Code'])
            slope = np.polyfit(age_df['Year Code'], age_df['ex'], 1)[0]
            slope_list.append({'Age': age, 'Sex': sex_code, 'Slope': slope})

        slope_df = pd.DataFrame(slope_list)

        top_n_neg_df = slope_df.nsmallest(df_n, "Slope")
        top_n_pos_df = slope_df.nlargest(df_n, "Slope")

        return top_n_neg_df, top_n_pos_df

    elif 'Age Group' in df.columns:
        top_df = df[df['Sex Code'] == sex_code][['Age Group', 'Year Code', 'ex']].copy()

        slope_list = []

        for age, age_df in top_df.groupby(['Age Group']):
            age_df = age_df.sort_values(['Year Code'])
            slope = np.polyfit(age_df['Year Code'], age_df['ex'], 1)[0]
            slope_list.append({'Age': age, 'Sex': sex_code, 'Slope': slope})

        slope_df = pd.DataFrame(slope_list)

        top_n_neg_df = slope_df.nsmallest(df_n, "Slope")
        top_n_pos_df = slope_df.nlargest(df_n, "Slope")

        return top_n_neg_df, top_n_pos_df

def plot_top_ex(df, sex_code, plot_n=3):
    top_n_neg_df, top_n_pos_df = top_n_ex(df, sex_code, plot_n)

    top_neg = 0
    top_pos = 0

    if 'Single-Year Ages Code' in df.columns:

        for agen, sexn in top_n_neg_df[['Age', 'Sex']].itertuples(index=False, name=None):
            neg_df = df[
                (df['Single-Year Ages Code']==agen)
                & (df['Sex Code']==sexn)
            ]

            neg_df = neg_df.sort_values(['Year Code'])

            top_neg += 1

            plt.figure(figsize=(10,6))

            plt.plot(
                neg_df['Year Code'],
                neg_df['ex']
            )

            plt.title(f"Top {top_neg} Worst Life Expectancy Progression\nAge: {agen} Sex: {sexn}")
            plt.xlabel("Year")
            plt.ylabel("Life Expectancy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        for agep, sexp in top_n_pos_df[['Age', 'Sex']].itertuples(index=False, name=None):
            pos_df = df[
                (df['Single-Year Ages Code']==agep)
                & (df['Sex Code']==sexp)
            ]

            pos_df = pos_df.sort_values(['Year Code'])

            top_pos += 1

            plt.figure(figsize=(10,6))

            plt.plot(
                pos_df['Year Code'],
                pos_df['ex']
            )

            plt.title(f"Top {top_pos} Best Life Expectancy Progression\nAge: {agep} Sex: {sexp}")
            plt.xlabel("Year")
            plt.ylabel("Life Expectancy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    elif 'Age Group' in df.columns:

        for agen, sexn in top_n_neg_df[['Age', 'Sex']].itertuples(index=False, name=None):
            neg_df = df[
                (df['Age Group']==agen)
                & (df['Sex Code']==sexn)
            ]

            neg_df = neg_df.sort_values(['Year Code'])

            top_neg += 1

            plt.figure(figsize=(10,6))

            plt.plot(
                neg_df['Year Code'],
                neg_df['ex']
            )

            plt.title(f"Top {top_neg} Worst Life Expectancy Progression\nAge: {agen} Sex: {sexn}")
            plt.xlabel("Year")
            plt.ylabel("Life Expectancy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        for agep, sexp in top_n_pos_df[['Age', 'Sex']].itertuples(index=False, name=None):
            pos_df = df[
                (df['Age Group']==agep)
                & (df['Sex Code']==sexp)
            ]

            pos_df = pos_df.sort_values(['Year Code'])

            top_pos += 1

            plt.figure(figsize=(10,6))

            plt.plot(
                pos_df['Year Code'],
                pos_df['ex']
            )

            plt.title(f"Top {top_pos} Best Life Expectancy Progression\nAge: {agep} Sex: {sexp}")
            plt.xlabel("Year")
            plt.ylabel("Life Expectancy")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def pct_change_qx(df, sex_code):
    filtered_df = df[df["Sex Code"]==sex_code].copy()

    if 'Single-Year Ages Code' in df.columns:

        results_list = []
        
        for _, age_df in filtered_df.groupby(['Sex Code', 'Single-Year Ages Code']):
            age_df = age_df.sort_values(['Year Code'])
            age_df['Improvement'] = -age_df['qx'].pct_change()
            results_list.append(age_df)

        improvement_df = pd.concat(results_list, ignore_index=True)

        improvement_pivot = improvement_df.pivot(
            index='Year Code',
            columns='Single-Year Ages Code',
            values='Improvement'
        )

        return improvement_pivot

    elif 'Age Group' in df.columns:

        results_list = []
        
        for _, age_df in filtered_df.groupby(['Sex Code', 'Age Group']):
            age_df = age_df.sort_values(['Year Code'])
            age_df['Improvement'] = -age_df['qx'].pct_change()
            results_list.append(age_df)

        improvement_df = pd.concat(results_list, ignore_index=True)

        improvement_pivot = improvement_df.pivot(
            index='Year Code',
            columns='Age Group',
            values='Improvement'
        )

        return improvement_pivot

def plot_pct_heatmap(df, sex_code):
    improvement_pivot = pct_change_qx(df, sex_code)

    if 'Single-Year Ages Code' in df.columns:
        max = improvement_pivot.max().max()
        min = improvement_pivot.min().min()

        norm = TwoSlopeNorm(vcenter=0, vmin=min, vmax=max)

        plt.figure(figsize=(10,6))

        im = plt.imshow(
            improvement_pivot.to_numpy(),
            cmap='RdBu_r',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        plt.xticks(
            ticks=range(0, len(improvement_pivot.columns), 5),
            labels=improvement_pivot.columns[::5].astype(int)
        )
        plt.yticks(
            ticks=range(len(improvement_pivot.index)),
            labels=improvement_pivot.index.astype(int)
        )
        
        plt.colorbar(im, label='% Yearly Mortality Improvement')
        
        plt.title(f'Yearly Percent Mortality Improvement Heatmap ({sex_code})')
        plt.xlabel('Age')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.show()

    elif 'Age Group' in df.columns:
        max = improvement_pivot.max().max()
        min = improvement_pivot.min().min()

        norm = TwoSlopeNorm(vcenter=0, vmin=min, vmax=max)

        plt.figure(figsize=(10,6))

        im = plt.imshow(
            improvement_pivot.to_numpy(),
            cmap='RdBu_r',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        plt.xticks(
            ticks=range(len(improvement_pivot.columns)),
            labels=improvement_pivot.columns
        )
        plt.yticks(
            ticks=range(len(improvement_pivot.index)),
            labels=improvement_pivot.index.astype(int)
        )
        
        plt.colorbar(im, label='% Yearly Mortality Improvement')
        
        plt.title(f'Yearly Percent Mortality Improvement Heatmap ({sex_code})')
        plt.xlabel('Age')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.show()

        
    





            


