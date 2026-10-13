"""Life-table columns from deaths and population.

Per (Year Code, Sex Code): qx -> px = 1 - qx -> lx (starting at RADIX) -> dx = qx * lx ->
Lx = lx - 0.5 * dx -> Tx (reverse cumulative sum of Lx) -> ex = Tx / lx.
Single-year ages use qx = Deaths / Population. Grouped ages (n = 5 years for 20-24, else 10)
convert the central rate m to qx = n*m / (1 + 0.5*n*m) and use Lx = n * (lx - 0.5 * dx).
The table stops at the last age in the data (84 here), so ex is expected years lived up to
85, not full life expectancy.
"""

import numpy as np

def add_life_tab_col(df, RADIX=100000):
    """Return a copy of `df` with qx, px, lx, dx, Lx, Tx, ex and Cumulative Survival (lx / RADIX).

    `df` must already be sorted by Year Code, Sex Code, then increasing age (load_filter does
    this), because lx and Tx are built from row order, not from the age values.
    """
    validation_df = df.copy()
    if "Single-Year Ages Code" in validation_df.columns:
        validation_df["qx"] = validation_df["Deaths"]/validation_df["Population"]
    elif "Age Group" in validation_df.columns:
        validation_df["n"] = np.where(
            validation_df["Age Group"] == '20-24 years', 5, 10
            )  
        validation_df["qx"] = (validation_df["n"] * validation_df["Deaths"]/validation_df["Population"]) / (1 + 0.5 * validation_df["n"] * validation_df["Deaths"]/validation_df["Population"])

    validation_df["px"] = 1 - validation_df["qx"]

    lx_list = []

    for _, group in validation_df.groupby(
        ["Year Code", "Sex Code"]
    ):
        
        current_lx = RADIX

        for _, row in group.iterrows():
            lx_list.append(current_lx)
            current_lx = current_lx*row["px"]
        
    validation_df["lx"] = lx_list

    validation_df["dx"] = validation_df["qx"]*validation_df["lx"]

    if "Single-Year Ages Code" in validation_df.columns:
            validation_df["Lx"] = validation_df["lx"] - 0.5*validation_df["dx"]
    elif "Age Group" in validation_df.columns:
            validation_df["Lx"] = validation_df["n"] * (validation_df["lx"] - 0.5*validation_df["dx"])

    Tx_list = []

    for _, group in validation_df.groupby(
        ["Year Code", "Sex Code"]
    ):
      
        Lx_reversed = group["Lx"].iloc[::-1]
        Tx_reversed = Lx_reversed.cumsum()
        Tx = Tx_reversed.iloc[::-1]
        Tx_list.extend(Tx)
    
    validation_df["Tx"] = Tx_list

    validation_df["ex"] = validation_df["Tx"]/validation_df["lx"]
    
    validation_df["Cumulative Survival"] = validation_df["lx"]/RADIX

    return validation_df