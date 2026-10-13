import pandas as pd

ny_df = pd.read_csv("data/processed/New York.csv")

ca_df = pd.read_csv("data/processed/California.csv")

tx_df = pd.read_csv("data/processed/Texas.csv")

fl_df = pd.read_csv("data/processed/Florida.csv")

st82_df = pd.read_csv("data/processed/State 8 Causes 2.csv")

stall1_df = pd.read_csv("data/processed/State All Causes.csv")

stall2_df = pd.read_csv("data/processed/State All Causes 2.csv")

us81_df = pd.read_csv("data/processed/US 8 Causes.csv")

us82_df = pd.read_csv("data/processed/US 8 Causes 2.csv")

usall1_df = pd.read_csv("data/processed/US All Causes.csv")

usall2_df = pd.read_csv("data/processed/US All Causes 2.csv")

dfs = [ny_df, ca_df, tx_df, fl_df, st82_df, stall1_df, stall2_df, us81_df, us82_df, usall1_df, usall2_df]

mf_df = pd.concat(dfs, ignore_index=True)

total_df = mf_df.groupby(
    ["Year Code", "Single-Year Ages Code", "State", "Cause List"], as_index=False
    )[["Deaths", "Population"]].sum()

total_df["Sex Code"] = "T"

combined_df = pd.concat([mf_df, total_df], ignore_index=True)

combined_df = combined_df.sort_values(by=["State", "Cause List", "Sex Code", "Year Code", "Single-Year Ages Code"])

combined_df.to_csv("data/processed/final.csv", index=False)

