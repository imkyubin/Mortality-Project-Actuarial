import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "mortality_2015_2020.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"


def load_table():
    return pd.read_csv(DATA_PATH)


def add_survival(df):
    df = df.copy()

    df["pxf"] = 1 - df["qxf"]
    df["pxm"] = 1 - df["qxm"]

    df["female_survival"] = df["pxf"].cumprod()
    df["male_survival"] = df["pxm"].cumprod()

    return df


def plot_mortality(df):
    plt.figure(figsize=(9, 6))

    plt.plot(df["Age"], df["qxf"], label="Female")
    plt.plot(df["Age"], df["qxm"], label="Male")

    plt.xlabel("Age")
    plt.ylabel("qx")
    plt.title("Mortality Rates by Age")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "mortality_rates.png",
        bbox_inches="tight"
    )

    plt.show()


def plot_log_mortality(df):
    plt.figure(figsize=(9, 6))

    plt.plot(df["Age"], np.log(df["qxf"]), label="Female")
    plt.plot(df["Age"], np.log(df["qxm"]), label="Male")

    plt.xlabel("Age")
    plt.ylabel("log(qx)")
    plt.title("Log Mortality Rates")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "log_mortality_rates.png",
        bbox_inches="tight"
    )

    plt.show()


def plot_survival(df):
    plt.figure(figsize=(9, 6))

    plt.plot(df["Age"], df["female_survival"], label="Female")
    plt.plot(df["Age"], df["male_survival"], label="Male")

    plt.xlabel("Age")
    plt.ylabel("Survival Probability")

    plt.title("Survival Curves from Age 20")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "survival_curves.png",
        bbox_inches="tight"
    )

    plt.show()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_table()
    df = add_survival(df)

    print(df.head())

    df.to_csv(
        OUTPUT_DIR / "mortality_with_survival.csv",
        index=False
    )

    plot_mortality(df)
    plot_log_mortality(df)
    plot_survival(df)


if __name__ == "__main__":
    main()