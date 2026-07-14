import pandas as pd

def show_age_groups(df, age_column):
    """
    Show the unique age groups in the dataframe.
    """
    return df[age_column].dropna().unique()

"""[   '< 1 year',   '1-4 years',   '5-9 years', '10-14 years', '15-19 years',
 '20-24 years', '25-34 years', '35-44 years', '45-54 years', '55-64 years',
 '65-74 years', '75-84 years',   '85+ years',  'Not Stated']"""

def map_single_age(age_code):
    """
    Map "Single-year Ages Code" to age group for.
    """

    if age_code == "NS":
        return "Not Stated"
    
    age = int(age_code)

    """
    elif used because search should end once right condition is matched
    """
    if age == 0:
        return "< 1 year"
    elif 1 <= age <= 4:
        return "1-4 years"
    elif 5 <= age <= 9:
        return "5-9 years"
    elif 10 <= age <= 14:
        return "10-14 years"
    elif 15 <= age <= 19:
        return "15-19 years"
    elif 20 <= age <= 24:
        return "20-24 years"
    elif 25 <= age <= 34:
        return "25-34 years"
    elif 35 <= age <= 44:
        return "35-44 years"
    elif 45 <= age <= 54:
        return "45-54 years"
    elif 55 <= age <= 64:
        return "55-64 years"
    elif 65 <= age <= 74:
        return "65-74 years"
    elif 75 <= age <= 84:
        return "75-84 years"
    elif 85 <= age:
        return "85+ years"
    
    else:
        return None

def man_add_harmonized_col(df, age_code_column):
    """
    create a manual new matching age group column for the single age dataset
    """
    harmonized_df = df.copy()

    groups = []

    for age_code in harmonized_df[age_code_column]:
        groups.append(map_single_age(age_code))

    harmonized_df["Age Group"] = groups
    
    return harmonized_df

def add_harmonized_col(df, age_code_column):
    """
    create a new matching age group column for the single age dataset
    """
    harmonized_df = df.copy()

    harmonized_df["Age Group"] = harmonized_df[age_code_column].apply(map_single_age)
    """
    .apply does what ever function inside () to every value of df column
    """

    return harmonized_df

def sum_or_cat_rule(x):
    """
    x is any series
    build my own aggregation rule since not applicable is in the Population category
    """
    numeric = pd.to_numeric(x, errors='coerce')

    if numeric.notna().any():
        return numeric.sum()

    return x.mode().iloc[0]
    """
    mode() returns a series not single value
    """

def aggregate(df):
    """
    sum numeric or mode category grouped by age, year, sex
    """
    aggregate_df = df.copy()

    aggregate_df = (aggregate_df.groupby(["Year Code", "Age Group", "Sex Code"], 
                                        as_index=False)
    .agg({
        "Deaths": 'sum', 
        "Population": sum_or_cat_rule
    }))

    return aggregate_df