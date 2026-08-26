import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse
from scipy.sparse.linalg import spsolve

def _second_difference_matrix(n):
    D = sparse.diags(
        diagonals=[np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
        offsets=[0, 1, 2],
        shape=(n-2, n)
    )

    return D

def fit_whittaker_henderson(df, year, sex_code, lambda_=1, pop_weights="n"):
    new_df = df.copy()

    model_df = new_df[
        (new_df["Year Code"] == year)
        & (new_df["Sex Code"] == sex_code)
    ].copy()

    model_df = model_df.sort_values("Single-Year Ages Code")

    model_df["ln_qx"] = np.log(model_df["qx"])
    
    y = model_df["ln_qx"].to_numpy()
    w = (model_df["Population"]/model_df["Population"].sum()).to_numpy()

    n = len(model_df)

    if pop_weights == "n":
        weights = list(np.ones(n))
    else:
        weights = w
    
    D = _second_difference_matrix(n)

    W = sparse.diags(weights)

    A = W + lambda_ * (D.T @ D)
    b = W @ y

    z = spsolve(A, b)

    model_df["ln_qx_fitted_wh"] = z
    model_df["whittaker_residual"] = (
        model_df["ln_qx"] - model_df["ln_qx_fitted_wh"]
        )

    return model_df

def fit_wh_all(df, lambda_=1, pop_weights="n"):
    df_distinct = df[["Year Code", "Sex Code"]].drop_duplicates()

    results_list = []

    for year, sex_code in df_distinct.itertuples(index=False, name=None):
        each_df = fit_whittaker_henderson(df, year, sex_code, lambda_, pop_weights)
        results_list.append(each_df)

    combined_df = pd.concat(results_list, ignore_index=True)

    return combined_df
    
def lambda_metrics(df, lambda_grid, pop_weights="n"):
    lambda_grid_sorted = sorted(lambda_grid)

    lambda_tuple = []
    
    for lambda_ in lambda_grid_sorted:
        combined_df = fit_wh_all(df, lambda_, pop_weights)

        n = len(combined_df)
        if pop_weights == "n":
            rss = (list(np.ones(n)) * combined_df["whittaker_residual"]**2).sum()
        else:
            rss = ((combined_df["Population"]/combined_df["Population"].sum()) * combined_df["whittaker_residual"]**2).sum()

        smoothness = 0
        for _, group_df in combined_df.groupby(["Year Code", "Sex Code"]):
            n2 = len(group_df)
            z = group_df["ln_qx_fitted_wh"].to_numpy()
            D = _second_difference_matrix(n2)
            k = ((D @ z)**2).sum()
            smoothness += k

        lambda_tuple.append({'Lambda': lambda_, 'RSS':rss, 'Smoothness':smoothness})

    lambda_df = pd.DataFrame(lambda_tuple)

    return lambda_df

def plot_lambda_elbow(lambda_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        lambda_df["Smoothness"],
        lambda_df["RSS"],
        marker='o'
        )

    for lambda_, rss, smoothness in lambda_df.itertuples(index=False, name=None):
        plt.annotate(f"{lambda_}", (smoothness, rss), xytext=(5,5), textcoords="offset points")
    
    plt.title("Lambda Elbow Plot")
    plt.xlabel("Smoothness")
    plt.ylabel("RSS")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def wh_heatmap(df, sex_code):
    group_df = df[df['Sex Code']==sex_code]

    pivot_df = group_df.pivot(
        index='Year Code',
        columns='Single-Year Ages Code',
        values='whittaker_residual'
    )

    min = pivot_df.min().min()
    max = pivot_df.max().max()

    norm = TwoSlopeNorm(vcenter=0, vmin=min, vmax=max)

    plt.figure(figsize=(10,6))

    im = plt.imshow(
        pivot_df.to_numpy(),
        cmap='RdBu_r',
        norm=norm,
        aspect='auto',
        origin='lower'
    )

    plt.xticks(
        ticks=range(0, len(pivot_df.columns), 5),
        labels=pivot_df.columns[::5].astype(int)
    )
    plt.yticks(
        ticks=range(len(pivot_df.index)),
        labels=pivot_df.index.astype(int)
    )

    plt.colorbar(im, label='ln(qx) Residual')

    plt.title(f'Whittaker-Henderson Residual Heatmap ({sex_code})')
    plt.xlabel('Age')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()

def plot_whittaker_fit(whittaker_df, metric=None, position=None, year=None, sex=None):
    if metric == None and position == None and year == None and sex == None:
        plt.figure(figsize=(8,5))

        plt.plot(
            whittaker_df["Single-Year Ages Code"],
            whittaker_df["ln_qx"],
            label="Observed ln(qx)"
        )

        plt.plot(
            whittaker_df["Single-Year Ages Code"],
            whittaker_df["ln_qx_fitted_wh"],
            label="Fitted Whittaker ln(qx)",
            linestyle="--"
        )

        plt.title("Observed ln(qx) vs Whittaker Fitted ln(qx)")
        plt.xlabel("Age")
        plt.ylabel("ln(qx)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8,5))
        
        plt.plot(
            whittaker_df["Single-Year Ages Code"],
            whittaker_df["ln_qx"],
            label="Observed ln(qx)"
        )
        
        plt.plot(
            whittaker_df["Single-Year Ages Code"],
            whittaker_df["ln_qx_fitted_wh"],
            label="Fitted Whittaker ln(qx)",
            linestyle="--"
        )
        
        plt.title(f"{position} Worst {metric}: {int(year)} {sex}")
        plt.xlabel("Age")
        plt.ylabel("ln(qx)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def wh_worst_fit(whittaker_df, n=10, pop_weights="n"):
    rss_list = []

    for (key1, key2), df in whittaker_df.groupby(["Year Code", "Sex Code"]):
        n2 = len(df)
        if pop_weights == "n":
            rss = (list(np.ones(n2)) * df["whittaker_residual"]**2).sum()
        else:
            rss = ((df["Population"]/df["Population"].sum()) * df["whittaker_residual"]**2).sum()

        rss_list.append({'Year Code': key1, 'Sex Code': key2, 'RSS': rss})

    rss_df = pd.DataFrame(rss_list)
    top_rss_df = rss_df.nlargest(n, "RSS")

    return top_rss_df

def wh_worst_cumdev(whittaker_df, n=10, pop_weights="n"):
    cumdev_list = []
    
    for (key1, key2), df in whittaker_df.groupby(["Year Code", "Sex Code"]):
        if pop_weights == "n":
            maxabscumdev = (df["whittaker_residual"]).cumsum().abs().max()
        else:
            maxabscumdev = ((df["Population"]/df["Population"].sum()) * df["whittaker_residual"]).cumsum().abs().max()
    
        cumdev_list.append({'Year Code': key1, 'Sex Code': key2, 'Max Deviation': maxabscumdev})
    
    cumdev_df = pd.DataFrame(cumdev_list)
    top_cumdev_df = cumdev_df.nlargest(n, "Max Deviation")
    
    return top_cumdev_df

def plot_3worst_fit(top_df, whittaker_df, top_k=3):
    top_n_df = top_df.iloc[0:top_k].copy()
    top_n_df["Rank"] = range(1, len(top_n_df) + 1)

    for year, sex, rank in top_n_df[['Year Code', 'Sex Code', 'Rank']].itertuples(index=False, name=None):
        plot_df = whittaker_df[
            (whittaker_df['Year Code'] == year)
            & (whittaker_df['Sex Code'] == sex)
        ]
        if "RSS" in top_n_df.columns:
            plot_whittaker_fit(plot_df, "RSS", rank, year, sex)
        elif "Max Deviation" in top_n_df.columns:
            plot_whittaker_fit(plot_df, "Max CumSum Deviation", rank, year, sex)

