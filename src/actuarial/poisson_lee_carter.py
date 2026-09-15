import numpy as np
import pandas as pd

from src.actuarial.lee_carter import fit_lee_carter
from src.actuarial.lc_outputs import reconstruct_lc, build_lc_tables 

def death_exposure_matrix(df, sex_code):
    df_model = df[
        df["Sex Code"] == sex_code
    ].copy()

    death_pivot = df_model.pivot_table(
        index="Single-Year Ages Code",
        columns="Year Code",
        values="Deaths",
        aggfunc="first"
    )

    pop_pivot = df_model.pivot_table(
        index="Single-Year Ages Code",
        columns="Year Code",
        values="Population",
        aggfunc="first"
    )

    D = death_pivot.to_numpy()
    E = pop_pivot.to_numpy()
    ages = death_pivot.index.to_numpy()
    years = death_pivot.columns.to_numpy()

    return D, E, ages, years

def initial_guess(D, E):
    mx_hat = D / E
    ax0, bx0, kt0 = fit_lee_carter(np.log(mx_hat))

    return ax0, bx0, kt0

def death_est(ax, bx, kt, E):
    dhat = E * np.exp(ax[:, None] + np.outer(bx, kt))
    return dhat

def poisson_loglik(dhat, D):
    loglik = np.sum(D * np.log(dhat) - dhat)
    
    return loglik

def normalize_param(ax, bx, kt):
    c = np.sum(bx)
    bx = bx / c
    kt = kt * c

    d = np.mean(kt)
    kt = kt - d
    ax = ax + bx * d

    return ax, bx, kt    

def update_ax(ax, bx, kt, D, E):
    dhat = death_est(ax, bx, kt, E)
    score = (D - dhat).sum(axis=1)
    hessian = -dhat.sum(axis=1)

    ax_new = ax - score / hessian

    return ax_new

def update_bx(ax, bx, kt, D, E):
    dhat = death_est(ax, bx, kt, E)

    score = ((D - dhat) * kt).sum(axis=1)
    hessian = -(dhat * kt**2).sum(axis=1)

    bx_new = bx - score / hessian

    return bx_new

def update_kt(ax, bx, kt, D, E):
    dhat = death_est(ax, bx, kt, E)

    score = (bx[:, None] * (D - dhat)).sum(axis=0)
    hessian = -((bx**2)[:, None] * dhat).sum(axis=0)

    kt_new = kt - score / hessian

    return kt_new

def fit_poisson_lc(D, E, ax0, bx0, kt0, max_iter=100, tol=1e-8):
    ax, bx, kt = ax0, bx0, kt0
    dhat = death_est(ax, bx, kt, E)
    loglik_old = poisson_loglik(dhat, D)

    for i in range(max_iter):
        ax = update_ax(ax, bx, kt, D, E)
        bx = update_bx(ax, bx, kt, D, E)
        kt = update_kt(ax, bx, kt, D, E)
        ax, bx, kt = normalize_param(ax, bx, kt)

        dhat = death_est(ax, bx, kt, E)
        loglik_new = poisson_loglik(dhat, D)

        if abs(loglik_new - loglik_old) / abs(loglik_old) < tol:
            break

        loglik_old = loglik_new

    return ax, bx, kt, loglik_new, i+1

def fit_poisson_lc_all(df, max_iter=100, tol=1e-8):
    sex_codes = df["Sex Code"].unique()
    age_list, year_list, fitted_list = [], [], []

    for sex_code in sex_codes:
        D, E, ages, years = death_exposure_matrix(df, sex_code)
        ax0, bx0, kt0 = initial_guess(D, E)
        ax, bx, kt, _, _ = fit_poisson_lc(D, E, ax0, bx0, kt0, max_iter, tol)
        fitted_ln_mx = reconstruct_lc(ax, bx, kt)
        age_df, year_df, fitted_df = build_lc_tables(ax, bx, kt, fitted_ln_mx, ages, years, sex_code)
        age_list.append(age_df)

        year_list.append(year_df)
        fitted_list.append(fitted_df)
        
    age_all = pd.concat(age_list, ignore_index=True)
    year_all = pd.concat(year_list, ignore_index=True)
    fitted_all = pd.concat(fitted_list, ignore_index=True)
        
    return age_all, year_all, fitted_all



