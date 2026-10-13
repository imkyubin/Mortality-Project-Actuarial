"""Clean the raw national CDC WONDER exports into the two processed tables.

Run from the repo root: python -m scripts.national.clean_wonder_exports

Reads the four exports in data/raw/: Compressed Mortality 1968-1978 and 1979-1998 (age groups)
and Underlying Cause of Death 1999-2020 and 2021-2024 (single-year ages; the second file is the
"2018-2024, Single Race" dataset exported for 2021-2024). Keeps year, age, sex, deaths and
population, drops subtotal rows, labels the blank-sex total rows "T", flags CDC "Suppressed"
cells, prints shape checks, and writes data/processed/old_1968_1998.csv and new_1999_2024.csv.
"""

from src.data.cleaning import (
    combine_datasets,
    filter_by_sex,
    flag_suppressed,
    NEW_COLUMNS,
    OLD_COLUMNS,
    remove_grand_totals,
    select_columns,
    fill_na_T
)

from src.data.wonder import load_wonder_csv

if __name__ == "__main__":
    wonder_1968_1978 = load_wonder_csv(
        "data/raw/Compressed Mortality, 1968-1978.csv"
    )
    wonder_1979_1998 = load_wonder_csv(
        "data/raw/Compressed Mortality, 1979-1998.csv"
    )
    wonder_1999_2020 = load_wonder_csv(
        "data/raw/Underlying Cause of Death, 1999-2020.csv"
    )
    wonder_2021_2024 = load_wonder_csv(
        "data/raw/Underlying Cause of Death, 2018-2024, Single Race.csv"
    )

    wonder_1968_1978 = select_columns(wonder_1968_1978, OLD_COLUMNS)
    wonder_1979_1998 = select_columns(wonder_1979_1998, OLD_COLUMNS)
    wonder_1999_2020 = select_columns(wonder_1999_2020, NEW_COLUMNS)
    wonder_2021_2024 = select_columns(wonder_2021_2024, NEW_COLUMNS)

    wonder_1968_1978 = remove_grand_totals(wonder_1968_1978, "Age Group")
    wonder_1979_1998 = remove_grand_totals(wonder_1979_1998, "Age Group")
    wonder_1999_2020 = remove_grand_totals(
        wonder_1999_2020,
        "Single-Year Ages Code",
    )
    wonder_2021_2024 = remove_grand_totals(
        wonder_2021_2024,
        "Single-Year Ages Code",
    )

    wonder_1968_1978 = fill_na_T(wonder_1968_1978, "Sex Code")
    wonder_1979_1998 = fill_na_T(wonder_1979_1998, "Sex Code")
    wonder_1999_2020 = fill_na_T(wonder_1999_2020, "Sex Code")
    wonder_2021_2024 = fill_na_T(wonder_2021_2024, "Sex Code")

    old_wonder = combine_datasets(
        [wonder_1968_1978, wonder_1979_1998]
    )
    new_wonder = combine_datasets(
        [wonder_1999_2020, wonder_2021_2024]
    )

    old_wonder = flag_suppressed(old_wonder)
    new_wonder = flag_suppressed(new_wonder)

    print("old_wonder:", old_wonder.shape)
    print("old_male:", filter_by_sex(old_wonder, "Sex Code", "M").shape)
    print("old_female:", filter_by_sex(old_wonder, "Sex Code", "F").shape)
    print("old_total:", filter_by_sex(old_wonder, "Sex Code", "T").shape)

    print("new_wonder:", new_wonder.shape)
    print("new_male:", filter_by_sex(new_wonder, "Sex Code", "M").shape)
    print("new_female:", filter_by_sex(new_wonder, "Sex Code", "F").shape)
    print("new_total:", filter_by_sex(new_wonder, "Sex Code", "T").shape)

    print("old_wonder unique age groups:", old_wonder["Age Group"].unique())
    print(
        "new_wonder unique single-year ages:",
        new_wonder["Single-Year Ages Code"].unique(),
    )

    old_wonder.to_csv("data/processed/old_1968_1998.csv", index=False)
    new_wonder.to_csv("data/processed/new_1999_2024.csv", index=False)
