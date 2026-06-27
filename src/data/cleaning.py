"""Standardize raw mortality data into the project schema.

Purpose:
    Convert CDC WONDER columns into consistent names and types such as:
    year, age, sex, deaths, population, mortality_rate.

Actuarial note:
    The cleaned table should keep deaths and exposure/population separate.
    Poisson Lee-Carter needs count data, not only precomputed rates.
"""
