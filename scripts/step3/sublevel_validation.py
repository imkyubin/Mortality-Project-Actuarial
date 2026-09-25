import warnings

import numpy as np
import pandas as pd

from src.data.harmonization import (
    add_harmonized_col
)
from src.step2.lee_carter import lc_variance_explained

"""
Sublevel decision procedure, per (Sex, State, Cause) combination — see
config.yaml data_collection_strategy.resolved_decision.age_floor_by_cause.sublevel_decision_procedure:
age floor -> single-year credibility -> grouped credibility -> young-end truncation
-> dimension check -> at least 3 groups -> lc_variance_explained vs national benchmark
"""

CREDIBILITY_THRESHOLD = 1082
MIN_GROUPS = 3
MAX_AGE = 84
"""
first-component variance explained may sit at most this far below the national all-cause
benchmark for the same sex — placeholder, config.yaml says to fix it at implementation time
"""
VARIANCE_TOLERANCE = 0.05

KEYS = ["Sex Code", "State", "Cause List"]

expected_group_size = {
    "20-24 years": 5,
    "25-34 years": 10,
    "35-44 years": 10,
    "45-54 years": 10,
    "55-64 years": 10,
    "65-74 years": 10,
    "75-84 years": 10
}

expected_group_order = {key: i for i, key in enumerate(expected_group_size)}

"""
"25-34 years" -> (25, 34)
"""
group_bounds = {key: (int(key.split("-")[0]), int(key.split("-")[1].split()[0])) for key in expected_group_size}

bin_edges = {low for low, _ in group_bounds.values()}

age_floors = {
    "All": 20,
    "#Chronic liver disease and cirrhosis (K70,K73-K74)": 35,
    "#Diseases of heart (I00-I09,I11,I13,I20-I51)": 45,
    "#Malignant neoplasms (C00-C97)": 45,
    "#Cerebrovascular diseases (I60-I69)": 55,
    "#Chronic lower respiratory diseases (J40-J47)": 55,
    "#Diabetes mellitus (E10-E14)": 55,
    "#Influenza and pneumonia (J09-J18)": 55,
    "#Alzheimer disease (G30)": 70
}

def apply_age_floors(df):
    """
    keep only ages from each cause's floor up to 84
    """
    missing = set(df["Cause List"].unique()) - set(age_floors)
    if missing:
        raise ValueError(f"no age floor set for: {sorted(missing)}")

    for cause, floor in age_floors.items():
        if floor not in bin_edges:
            warnings.warn(
                f"{cause} floor {floor} is not a grouped-bin edge {sorted(bin_edges)}: its youngest "
                f"group will be incomplete, so a grouped fallback can never keep it "
                f"(see config.yaml age_floor_by_cause.bin_alignment_note)"
            )

    floor_df = df[
        (df["Single-Year Ages Code"] >= df["Cause List"].map(age_floors))
        & (df["Single-Year Ages Code"] <= MAX_AGE)
    ].copy()

    return floor_df

def full_age_grid(sub_df, floor, years):
    """
    one row per (age, year) from the floor to 84; a row missing from the data, or with NaN deaths
    (e.g. "Suppressed" coerced by to_numeric), gets 0 deaths and Present=False, so it fails
    credibility and makes its age group incomplete instead of disappearing
    """
    grid = pd.MultiIndex.from_product(
        [range(floor, MAX_AGE + 1), years], names=["Single-Year Ages Code", "Year Code"]
    )
    indexed = sub_df.set_index(["Single-Year Ages Code", "Year Code"])[["Deaths", "Population"]]

    grid_df = indexed.reindex(grid).reset_index()
    grid_df["Present"] = grid_df["Deaths"].notna()
    grid_df["Deaths"] = grid_df["Deaths"].fillna(0)

    return grid_df

def into_group(sub_df, floor, years):
    """
    aggregate single ages into harmonized groups; a group missing any single age in a year
    gets 0 deaths so it fails credibility rather than being silently dropped
    """
    grid_df = add_harmonized_col(full_age_grid(sub_df, floor, years), "Single-Year Ages Code")

    group_df = grid_df.groupby(["Age Group", "Year Code"], as_index=False).agg(
        Deaths=("Deaths", "sum"),
        Population=("Population", "sum"),
        row_count=("Present", "sum")
    )
    group_df["Complete Group"] = group_df["row_count"] == group_df["Age Group"].map(expected_group_size)
    group_df.loc[~group_df["Complete Group"], "Deaths"] = 0
    group_df = group_df.drop(columns=["row_count"])

    return group_df

def concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def count_validation(df, years):
    """
    single-year credibility first; any failing cell sends the whole combination to grouping.
    also returns audit tables 1 (pass/fail per single age) and 2 (pass/fail per age group)
    """
    single_age_list = []
    grouped_age_list = []
    single_audit_list = []
    group_audit_list = []

    for (sex, state, cause), sub_df in df.groupby(KEYS):
        floor = age_floors[cause]

        single_audit = (full_age_grid(sub_df, floor, years)
                        .groupby("Single-Year Ages Code", as_index=False)
                        .agg(**{"Min Deaths": ("Deaths", "min")}))
        single_audit["Pass"] = single_audit["Min Deaths"] >= CREDIBILITY_THRESHOLD
        single_audit[KEYS] = [sex, state, cause]
        single_audit_list.append(single_audit)

        if single_audit["Pass"].all():
            single_age_list.append(sub_df)
        else:
            sub_grouped_df = into_group(sub_df, floor, years)
            sub_grouped_df[KEYS] = [sex, state, cause]
            grouped_age_list.append(sub_grouped_df)

            group_audit = sub_grouped_df.groupby("Age Group", as_index=False).agg(
                **{"Min Deaths": ("Deaths", "min"), "Complete": ("Complete Group", "all")})
            group_audit["Pass"] = group_audit["Min Deaths"] >= CREDIBILITY_THRESHOLD
            group_audit[KEYS] = [sex, state, cause]
            group_audit_list.append(group_audit)

    single_audit_df = concat(single_audit_list)
    group_audit_df = concat(group_audit_list)
    if len(group_audit_df):
        group_audit_df = (group_audit_df
                          .assign(Order=group_audit_df["Age Group"].map(expected_group_order))
                          .sort_values(KEYS + ["Order"])
                          .drop(columns="Order"))

    return concat(single_age_list), concat(grouped_age_list), single_audit_df, group_audit_df

def years_validation(df):
    """
    grouped combinations only: drop the oldest failing group and every younger group;
    if 75-84 fails nothing survives. Single-year combinations never need this, since they
    only exist if every cell already passed.
    """
    if df.empty:
        return df

    val_df = df.copy()
    keys = list(expected_group_size)

    for (sex, state, cause), sub2_df in val_df.groupby(KEYS):
        last_fail_index = -1
        for i, key in enumerate(keys):
            if key in sub2_df["Age Group"].values and sub2_df.loc[sub2_df['Age Group'] == key, 'Deaths'].min() < CREDIBILITY_THRESHOLD:
                last_fail_index = i

        val_df = val_df[~((val_df['Sex Code'] == sex)
            & (val_df['State'] == state)
            & (val_df['Cause List'] == cause)
            & (val_df['Age Group'].isin(keys[:last_fail_index + 1])))]

    return val_df

def dim_validation(df, n_years):
    """
    backstop: every year must be present and cover the same unbroken age range;
    any combination that doesn't is dropped and flagged
    """
    flag_columns = KEYS + ["Year Code", "Reason"]
    if df.empty:
        return df, pd.DataFrame(columns=flag_columns)

    dim_df = df.copy()
    flag_list = []

    if 'Single-Year Ages Code' in dim_df.columns:
        age_col = 'Single-Year Ages Code'
    else:
        age_col = 'Order'
        dim_df["Order"] = dim_df["Age Group"].map(expected_group_order)

    for (sex, state, cause), sub3_df in dim_df.groupby(KEYS):
        row_count = sub3_df.groupby("Year Code")[age_col].agg(First="min", Last="max")
        lower_bound = row_count["First"].max()
        upper_bound = row_count["Last"].min()

        in_combo = ((dim_df['Sex Code'] == sex)
                    & (dim_df['State'] == state)
                    & (dim_df['Cause List'] == cause))
        dim_df = dim_df[~(in_combo & ((dim_df[age_col] < lower_bound) | (dim_df[age_col] > upper_bound)))]

        in_combo = ((dim_df['Sex Code'] == sex)
                    & (dim_df['State'] == state)
                    & (dim_df['Cause List'] == cause))
        combo = dim_df[in_combo]

        has_gap = False
        if combo["Year Code"].nunique() != n_years:
            has_gap = True
            flag_list.append({'Sex Code': sex, 'State': state, 'Cause List': cause,
                              'Year Code': None, 'Reason': 'missing years'})
        for year, sub5_df in combo.groupby("Year Code"):
            if len(sub5_df) != upper_bound - lower_bound + 1:
                has_gap = True
                flag_list.append({'Sex Code': sex, 'State': state, 'Cause List': cause,
                                  'Year Code': year, 'Reason': 'missing ages'})

        if has_gap:
            dim_df = dim_df[~in_combo]

    if age_col == 'Order':
        dim_df = dim_df[['Year Code', 'Age Group', 'Sex Code', 'State', 'Cause List', 'Deaths', 'Population']]

    return dim_df, pd.DataFrame(flag_list, columns=flag_columns)

def min_groups_validation(df):
    """
    a truncated grouped combination needs at least MIN_GROUPS groups for a meaningful fit
    """
    if df.empty:
        return df, []

    n_groups = df.groupby(KEYS)["Age Group"].nunique()
    too_few = list(n_groups[n_groups < MIN_GROUPS].index)

    return drop_combos(df, too_few), too_few

def first_component(df, age_col):
    """
    share of variance the first Lee-Carter component captures for one combination
    (lc_matrix is single-age and needs qx, so the ln(mx) matrix is built here for both resolutions)
    """
    ln_mx_df = df.assign(ln_mx=np.log(df["Deaths"] / df["Population"]))
    if age_col == "Age Group":
        ln_mx_df["Order"] = ln_mx_df["Age Group"].map(expected_group_order)
        age_col = "Order"
    matrix = ln_mx_df.pivot_table(index=age_col, columns="Year Code", values="ln_mx").to_numpy()

    return lc_variance_explained(matrix)["Variance Explained"].iloc[0]

def national_benchmark(df):
    """
    first-component variance explained of the national all-cause single-year 20-84 fit, per sex
    """
    national = df[(df["State"] == "United States") & (df["Cause List"] == "All")]

    return {sex: first_component(sub_df, "Single-Year Ages Code")
            for sex, sub_df in national.groupby("Sex Code")}

def variance_validation(single_df, grouped_df, benchmark):
    """
    Lee-Carter single-factor check per combination against the same-sex national benchmark
    """
    variance_list = []

    for resolution_df, age_col in [(single_df, "Single-Year Ages Code"), (grouped_df, "Age Group")]:
        if resolution_df.empty:
            continue
        for (sex, state, cause), sub_df in resolution_df.groupby(KEYS):
            value = first_component(sub_df, age_col)
            variance_list.append({'Sex Code': sex, 'State': state, 'Cause List': cause,
                                  'Variance Explained': value,
                                  'Benchmark': benchmark[sex],
                                  'Variance Pass': value >= benchmark[sex] - VARIANCE_TOLERANCE})

    return pd.DataFrame(variance_list, columns=KEYS + ["Variance Explained", "Benchmark", "Variance Pass"])

def drop_combos(df, combos):
    if df.empty or not combos:
        return df
    return df[~df.set_index(KEYS).index.isin(combos)]

def combos_of(df):
    return set() if df.empty else set(df.groupby(KEYS).groups)

def build_report(all_combos, single_df, grouped_df, drop_reasons, variance_df):
    """
    final rollup per combination (audit table 3): resolution that survived, age range kept,
    variance explained, and why anything was dropped
    """
    single_range = (single_df.groupby(KEYS)["Single-Year Ages Code"].agg(["min", "max"])
                    if len(single_df) else pd.DataFrame())
    groups = grouped_df.groupby(KEYS)["Age Group"].unique() if len(grouped_df) else pd.Series(dtype=object)

    report_list = []
    for combo in all_combos:
        sex, state, cause = combo
        row = {'Sex Code': sex, 'State': state, 'Cause List': cause,
               'Age Floor': age_floors[cause],
               'Resolution': 'dropped', 'Min Age Kept': None, 'Max Age Kept': None,
               'Groups Kept': None, 'Drop Reason': drop_reasons.get(combo)}

        if combo in single_range.index:
            row.update({'Resolution': 'single',
                        'Min Age Kept': single_range.loc[combo, "min"],
                        'Max Age Kept': single_range.loc[combo, "max"]})
        elif combo in groups.index:
            kept = sorted(groups.loc[combo], key=expected_group_order.get)
            row.update({'Resolution': 'grouped',
                        'Min Age Kept': group_bounds[kept[0]][0],
                        'Max Age Kept': group_bounds[kept[-1]][1],
                        'Groups Kept': len(kept)})

        report_list.append(row)

    report_columns = KEYS + ['Age Floor', 'Resolution', 'Min Age Kept', 'Max Age Kept', 'Groups Kept', 'Drop Reason']
    report_df = pd.DataFrame(report_list, columns=report_columns).merge(variance_df, on=KEYS, how="left")

    return report_df.sort_values(["State", "Cause List", "Sex Code"])

def run_validation(final_df, benchmark=None):
    """
    full sublevel decision procedure; returns the two model-ready tables, the final report,
    the two audit tables, and the dimension flags
    """
    years = sorted(final_df["Year Code"].unique())

    if benchmark is None:
        benchmark = national_benchmark(final_df)

    floored_df = apply_age_floors(final_df)
    all_combos = sorted(combos_of(floored_df))

    single_age_df, grouped_age_df, single_audit_df, group_audit_df = count_validation(floored_df, years)

    drop_reasons = {}

    truncated_df = years_validation(grouped_age_df)
    for combo in combos_of(grouped_age_df) - combos_of(truncated_df):
        drop_reasons[combo] = "75-84 failed: no credible age range"

    final_single_age_df, single_flags = dim_validation(single_age_df, len(years))
    final_grouped_age_df, grouped_flags = dim_validation(truncated_df, len(years))
    for combo in ((combos_of(single_age_df) - combos_of(final_single_age_df))
                  | (combos_of(truncated_df) - combos_of(final_grouped_age_df))):
        drop_reasons[combo] = "dimension gap (missing ages or years)"

    final_grouped_age_df, too_few = min_groups_validation(final_grouped_age_df)
    for combo in too_few:
        drop_reasons[combo] = f"fewer than {MIN_GROUPS} age groups after truncation"

    variance_df = variance_validation(final_single_age_df, final_grouped_age_df, benchmark)
    failed = [tuple(row) for row in variance_df.loc[~variance_df["Variance Pass"], KEYS].to_numpy()]
    for combo in failed:
        drop_reasons[combo] = "variance explained below national benchmark"
    final_single_age_df = drop_combos(final_single_age_df, failed)
    final_grouped_age_df = drop_combos(final_grouped_age_df, failed)

    report_df = build_report(all_combos, final_single_age_df, final_grouped_age_df, drop_reasons, variance_df)

    if len(final_single_age_df):
        final_single_age_df = final_single_age_df.sort_values(by=["State", "Cause List", "Sex Code", "Year Code", "Single-Year Ages Code"])

    if len(final_grouped_age_df):
        final_grouped_age_df = (final_grouped_age_df
                                .assign(Order=final_grouped_age_df["Age Group"].map(expected_group_order))
                                .sort_values(by=["State", "Cause List", "Sex Code", "Year Code", "Order"])
                                .drop(columns="Order"))

    dim_flags_df = pd.concat([single_flags, grouped_flags], ignore_index=True)

    return final_single_age_df, final_grouped_age_df, report_df, single_audit_df, group_audit_df, dim_flags_df

if __name__ == "__main__":
    final_df = pd.read_csv("data/processed/final.csv")

    (final_single_age_df, final_grouped_age_df, report_df,
     single_audit_df, group_audit_df, dim_flags_df) = run_validation(final_df)

    final_single_age_df.to_csv("data/processed/single_age_final.csv", index=False)

    final_grouped_age_df.to_csv("data/processed/grouped_age_final.csv", index=False)

    report_df.to_csv("data/processed/validation_report.csv", index=False)

    single_audit_df.to_csv("data/processed/audit_single_age.csv", index=False)

    group_audit_df.to_csv("data/processed/audit_grouped_age.csv", index=False)

    dim_flags_df.to_csv("data/processed/dim_flags.csv", index=False)
