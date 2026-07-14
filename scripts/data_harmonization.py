from src.data.cleaning import combine_datasets

from src.data.harmonization import (
    show_age_groups,
    add_harmonized_col,
    aggregate
)

import pandas as pd

old_wonder = pd.read_csv("data/processed/old_1968_1998.csv")
new_wonder = pd.read_csv("data/processed/new_1999_2024.csv")

unique = show_age_groups(old_wonder, "Age Group")
print(unique)

adj_new = aggregate(add_harmonized_col(new_wonder, "Single-Year Ages Code"))

harmonized_df = combine_datasets([old_wonder, adj_new])

harmonized_df.to_csv("data/processed/harmonized_1968-2024.csv", index=False)