from src.actuarial.load_filter import load_filter

single_age_df = load_filter(
        "data/processed/new_1999_2024.csv", 
        age_col="Single-Year Ages Code", 
        min_age=20, 
        max_age=84
    )

group_age_df = load_filter(
        "data/processed/harmonized_1968-2024.csv", 
        age_col="Age Group", 
        min_age=None, 
        max_age=None
    )