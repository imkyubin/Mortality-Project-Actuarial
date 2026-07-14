import pandas as pd

from src.data.wonder import load_wonder_csv

new_wonder = load_wonder_csv("data/processed/new_1999_2024.csv")
print(new_wonder["Population"].dtype)
