"""Poisson Lee-Carter (Brouhns, Denuit and Vermunt, 2002), the project's primary mortality model.

Deaths D(x, t) ~ Poisson(E(x, t) * exp(ax + bx * kt)), with exposure E = Population. The
parameters maximize the Poisson log-likelihood, so each age-year cell counts in proportion to
its deaths and the fit reproduces total deaths at every age. There is no closed form: starting
from the SVD solution, ax, bx and kt are updated in turn with Newton-Raphson steps (score over
Hessian of the log-likelihood) and re-normalized to sum(bx) = 1, mean(kt) = 0 each round,
until the relative change in log-likelihood falls below `tol`.
"""

import numpy as np
import pandas as pd

from src.actuarial.lee_carter import fit_lee_carter
from src.actuarial.lc_outputs import reconstruct_lc, build_lc_tables 

def death_exposure_matrix(df, sex_code):
    """Deaths and Population for one sex as ages x years matrices; returns (D, E, ages, years)."""
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
    """Starting values for the Newton-Raphson fit: the SVD Lee-Carter fit to ln(D / E)."""
    mx_hat = D / E
    ax0, bx0, kt0 = fit_lee_carter(np.log(mx_hat))

    return ax0, bx0, kt0

def death_est(ax, bx, kt, E):
    """Expected deaths E * exp(ax + bx * kt) as an ages x years matrix."""
    dhat = E * np.exp(ax[:, None] + np.outer(bx, kt))
    return dhat

def poisson_loglik(dhat, D):
    """Poisson log-likelihood up to a constant: sum(D * ln(Dhat) - Dhat)."""
    loglik = np.sum(D * np.log(dhat) - dhat)
    
    return loglik

def normalize_param(ax, bx, kt):
    """Impose sum(bx) = 1 and mean(kt) = 0 without changing the fitted rates (ax absorbs the kt shift)."""
    c = np.sum(bx)
    bx = bx / c
    kt = kt * c

    d = np.mean(kt)
    kt = kt - d
    ax = ax + bx * d

    return ax, bx, kt    

def update_ax(ax, bx, kt, D, E):
    """One Newton-Raphson step for every ax (each age's score and Hessian summed over years)."""
    dhat = death_est(ax, bx, kt, E)
    score = (D - dhat).sum(axis=1)
    hessian = -dhat.sum(axis=1)

    ax_new = ax - score / hessian

    return ax_new

def update_bx(ax, bx, kt, D, E):
    """One Newton-Raphson step for every bx, holding ax and kt fixed."""
    dhat = death_est(ax, bx, kt, E)

    score = ((D - dhat) * kt).sum(axis=1)
    hessian = -(dhat * kt**2).sum(axis=1)

    bx_new = bx - score / hessian

    return bx_new

def update_kt(ax, bx, kt, D, E):
    """One Newton-Raphson step for every kt (each year's score and Hessian summed over ages)."""
    dhat = death_est(ax, bx, kt, E)

    score = (bx[:, None] * (D - dhat)).sum(axis=0)
    hessian = -((bx**2)[:, None] * dhat).sum(axis=0)

    kt_new = kt - score / hessian

    return kt_new

def fit_poisson_lc(D, E, ax0, bx0, kt0, max_iter=100, tol=1e-8):
    """Iterate the ax, bx, kt updates until the log-likelihood's relative change is below `tol`.

    Stops after `max_iter` rounds at most. Returns (ax, bx, kt, final log-likelihood, rounds used).
    """
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
    """Fit Poisson Lee-Carter for every Sex Code in `df`; same three output tables as fit_lc_all."""
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



