import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

def ref_
