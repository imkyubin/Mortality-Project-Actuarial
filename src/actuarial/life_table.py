import numpy as np

def add_life_tab_col(df, RADIX):
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