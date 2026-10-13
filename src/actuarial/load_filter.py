"""Load a processed CDC WONDER CSV into the shape the life-table code expects.

Coerces Year Code, Deaths and Population to numbers, keeps the requested ages (a numeric
single-year range, or a list of age-group labels, by default `age_value`: the 7 harmonized
groups 20-24 ... 75-84), and sorts by year, sex, then age. That sort matters: add_life_tab_col
builds lx and Tx from row order.
"""

import pandas as pd

age_value = ['20-24 years', '25-34 years', '35-44 years', '45-54 years', '55-64 years',
 '65-74 years', '75-84 years']

def load_filter(
        path, 
        age_col="Single-Year Ages Code", 
        min_age=None, 
        max_age=None,
        age_values=age_value
    ):
    """Read `path` and return a numeric, age-filtered table sorted by year, sex and age.

    age_col: "Single-Year Ages Code" (single ages) or "Age Group" (grouped ages).
    min_age, max_age: inclusive single-age range to keep; when given, they take precedence.
    age_values: age-group labels to keep, in order. The column becomes an ordered Categorical so
        sorting follows age rather than text.
    """
    df = pd.read_csv(path)
    validation_df = df.copy()

    validation_df["Year Code"] = pd.to_numeric(validation_df["Year Code"]).astype(int)
    validation_df["Deaths"] = pd.to_numeric(
        validation_df["Deaths"],
        errors='coerce',
        )
    validation_df["Population"] = pd.to_numeric(
        validation_df["Population"],
        errors='coerce',
        )

    if min_age is not None or max_age is not None:
        validation_df[age_col] = pd.to_numeric(
            validation_df[age_col],
            errors='coerce', 
            )
        validation_df = validation_df[
            (validation_df[age_col] >= min_age)
            & (validation_df[age_col] <= max_age)
            ]
    elif age_values is age_value:    
        validation_df = validation_df[
            validation_df[age_col].isin(age_value)
        ]
        validation_df[age_col] = pd.Categorical(
            validation_df[age_col], categories=age_value, ordered=True
        )
    elif age_values is not age_value:    
            validation_df = validation_df[
                validation_df[age_col].isin(age_values)
            ]
            validation_df[age_col] = pd.Categorical(
                validation_df[age_col], categories=age_values, ordered=True
            )

    validation_df = validation_df.sort_values(
        ["Year Code", "Sex Code", age_col]
    )
    
    return validation_df