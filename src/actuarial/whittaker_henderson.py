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
    w = model_df["Population"].to_numpy()

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
            rss = (combined_df["Population"] * combined_df["whittaker_residual"]**2).sum()

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
    plt.xlabel("RSS")
    plt.ylabel("Smoothness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()