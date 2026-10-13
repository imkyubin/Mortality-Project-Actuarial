"""Build the 1968-2024 national table on one set of age groups (the data behind dashboard page 1).

Run from the repo root, after clean_wonder_exports:
    python -m scripts.national.harmonize_age_groups

Buckets the 1999-2024 single-year ages into the 1968-1998 age groups, sums them, and appends
them to the older data, writing data/processed/harmonized_1968-2024.csv. Prints the age groups
found in the older data as a check.
"""

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