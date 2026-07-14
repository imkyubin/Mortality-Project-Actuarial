import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve

INPUT_FILE = "data/processed/new_1999_2024.csv"

START_YEAR = 2015
END_YEAR = 2020
MIN_AGE = 20
MAX_AGE = 84

RADIX = 100000

def load_data(path):
    return pd.read_csv(path)

def filter_for_excel(df):
    validation_df = df.copy()
    validation_df["Year Code"] = pd.to_numeric(validation_df["Year Code"])
    validation_df["Single-Year Ages Code"] = pd.to_numeric(
        validation_df["Single-Year Ages Code"],
        errors='coerce',
        )
    validation_df["Deaths"] = pd.to_numeric(
        validation_df["Deaths"],
        errors='coerce',
        )
    validation_df["Population"] = pd.to_numeric(
        validation_df["Population"],
        errors='coerce',
        )
    
    validation_df = validation_df[
        (validation_df["Year Code"] >= START_YEAR)
        & (validation_df["Year Code"] <= END_YEAR)
        & (validation_df["Single-Year Ages Code"] >= MIN_AGE)
        & (validation_df["Single-Year Ages Code"] <= MAX_AGE)
    ]

    validation_df = validation_df.sort_values(
        ["Year Code", "Sex Code", "Single-Year Ages Code"]
    )

    return validation_df

def add_life_tab_col(df):
    validation_df = df.copy()
    validation_df["qx"] = validation_df["Deaths"]/validation_df["Population"]
    validation_df["px"] = 1 - validation_df["qx"]

    lx_list = []

    for _, group in validation_df.groupby(
        ["Year Code", "Sex Code"]
    ):
        
        current_lx = RADIX
       
        for _, row in group.iterrows():
            lx_list.append(current_lx)
            current_lx = current_lx*row["px"]
        
    validation_df["lx"] = lx_list

    validation_df["dx"] = validation_df["qx"]*validation_df["lx"]
    
    validation_df["Lx"] = validation_df["lx"] - 0.5*validation_df["dx"]

    Tx_list = []

    for _, group in validation_df.groupby(
        ["Year Code", "Sex Code"]
    ):
      
        Lx_reversed = group["Lx"].iloc[::-1]
        Tx_reversed = Lx_reversed.cumsum()
        Tx = Tx_reversed.iloc[::-1]
        Tx_list.extend(Tx)
    
    validation_df["Tx"] = Tx_list

    validation_df["ex"] = validation_df["Tx"]/validation_df["lx"]
    
    validation_df["Cumulative Survival"] = validation_df["lx"]/RADIX

    return validation_df
 
def build_avg_tab(df):
    validation_df = df.copy()

    validation_df = validation_df[
        (validation_df["Sex Code"].isin(["M", "F"]))
    ].copy()

    validation_df["ln_qx"] = np.log(validation_df["qx"])
    
    avg_df = validation_df.groupby(
        ["Sex Code", "Single-Year Ages Code"],
        as_index=False
    ).agg(
        avg_qx=("qx", "mean"),
        avg_ln_qx=("ln_qx", "mean"),
        avg_sur=("Cumulative Survival", "mean"),
        avg_ex=("ex", "mean"),
        avg_dx=("dx", "mean")
    )

    return avg_df

def plot_avg_tab(avg_df, metric_column, y_label, title):
    plt.figure(figsize=(8,5))

    for sex_code in ["F", "M"]:
        sex_df = avg_df[avg_df["Sex Code"] == sex_code]

        plt.plot(
            sex_df["Single-Year Ages Code"],
            sex_df[metric_column],
            label=sex_code
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_avg_plots(avg_df):
    plot_avg_tab(
        avg_df,
        "avg_qx",
        "Average qx",
        "Average qx by Age and Sex"
    )

    plot_avg_tab(
        avg_df,
        "avg_ln_qx",
        "Average ln(qx)",
        "Average ln(qx) by Age and Sex"
    )

    plot_avg_tab(
        avg_df,
        "avg_sur",
        "Average Survival",
        "Average Survival by Age and Sex"
    )

    plot_avg_tab(
        avg_df,
        "avg_ex",
        "Average Life Expectancy",
        "Average Life Expectancy by Age and Sex"
    )

    plot_avg_tab(
        avg_df,
        "avg_dx",
        "Average Life Table Deaths",
        "Average Life Table Deaths by Age and Sex"
    )

def fit_gompertz(df, year, sex_code):
    model_df = df.copy()

    model_df = model_df[
        (model_df["Year Code"] == year)
        & (model_df["Sex Code"] == sex_code)
    ].copy()

    model_df["ln_qx"] = np.log(model_df["qx"])

    X = model_df[["Single-Year Ages Code"]]
    X = sm.add_constant(X)

    y = model_df["ln_qx"]

    model = sm.OLS(y, X).fit()

    model_df["fitted_ln_qx"] = model.predict(X)
    model_df["residual"] = model_df["ln_qx"] - model_df["fitted_ln_qx"]

    return model, model_df

def plot_gompertz_fit(gompertz_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        gompertz_df["Single-Year Ages Code"],
        gompertz_df["ln_qx"],
        label="Observed ln(qx)"
    )

    plt.plot(
        gompertz_df["Single-Year Ages Code"],
        gompertz_df["fitted_ln_qx"],
        label="Fitted ln(qx)",
        linestyle="--"
    )

    plt.title("Observed ln(qx) vs Gompertz Fitted ln(qx) Model")
    plt.xlabel("Age")
    plt.ylabel("ln(qx)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gompertz_residual(gompertz_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        gompertz_df["Single-Year Ages Code"],
        gompertz_df["residual"],
        label="Residual ln(qx)"
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=1
    )
    
    plt.title("ln(qx) Residual Gompertz Model")
    plt.xlabel("Age")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_ma_graduation(df, year, sex_code):
    ma_df = df.copy()

    ma_df = ma_df[
        (ma_df["Year Code"] == year)
        & (ma_df["Sex Code"] == sex_code)
    ].copy()

    ma_df = ma_df.sort_values("Single-Year Ages Code")

    ma_df["ln_qx"] = np.log(ma_df["qx"])

    ma_df["qx_ma_3"] = ma_df["qx"].rolling(
        window=3,
        center=True,
        min_periods=1
    ).mean()

    ma_df["qx_ma_5"] = ma_df["qx"].rolling(
        window=5,
        center=True,
        min_periods=3
    ).mean()

    ma_df["ln_qx_ma_3"] = ma_df["ln_qx"].rolling(
        window=3,
        center=True,
        min_periods=1
    ).mean()

    ma_df["ln_qx_ma_5"] = ma_df["ln_qx"].rolling(
        window=5,
        center=True,
        min_periods=3
    ).mean()

    return ma_df

def plot_ma(ma_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["qx"],
        label="Observed qx"
    )

    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["qx_ma_3"],
        label="MA3 Smoothed qx",
        linestyle="--"
    )
    
    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["qx_ma_5"],
        label="MA5 Smoothed qx",
        linestyle="--"
    )

    plt.title("Observed qx vs MA Smoothed qx Model")
    plt.xlabel("Age")
    plt.ylabel("qx")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ln_ma(ma_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["ln_qx"],
        label="Observed ln(qx)"
    )

    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["ln_qx_ma_3"],
        label="MA3 Smoothed ln(qx)",
        linestyle="--"
    )
    
    plt.plot(
        ma_df["Single-Year Ages Code"],
        ma_df["ln_qx_ma_5"],
        label="MA5 Smoothed ln(qx)",
        linestyle="--"
    )

    plt.title("Observed ln(qx) vs MA Smoothed ln(qx) Model")
    plt.xlabel("Age")
    plt.ylabel("ln(qx)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def makeham_fn(age, A, B, c):
    return A + B * (c ** age)
    # correct format for curve_fit
    # age is an array or series

def fit_makeham(df, year, sex_code):
    model_df = df.copy()

    model_df = model_df[
        (model_df["Year Code"] == year)
        & (model_df["Sex Code"] == sex_code)
    ].copy()

    x = model_df["Single-Year Ages Code"]
    y = model_df["qx"]

    params, _ = curve_fit(
        makeham_fn,
        x,
        y,
        p0=[0.0001, 0.000001, 1.08],
        bounds=(
            [0, 0, 1],
            [1, 1, 2]
        ),
        maxfev=10000
    )
    # Order: function, input_values, output_values, starting parameter, bounds, ...

    A, B, c = params

    model_df["makeham_fitted_qx"] = makeham_fn(x, A, B, c)
    model_df["makeham_residual"] = model_df["qx"] - model_df["makeham_fitted_qx"]
                                                     
    return params, model_df

def plot_makeham_fit(makeham_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        makeham_df["Single-Year Ages Code"],
        makeham_df["qx"],
        label="Observed qx"
    )

    plt.plot(
        makeham_df["Single-Year Ages Code"],
        makeham_df["makeham_fitted_qx"],
        label="Fitted makeham qx",
        linestyle="--"
    )

    plt.title("Observed qx vs Makeham Fitted qx Model")
    plt.xlabel("Age")
    plt.ylabel("qx")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_makeham_residual(makeham_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        makeham_df["Single-Year Ages Code"],
        makeham_df["makeham_residual"],
        label="Residual qx"
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=1
    )
    
    plt.title("qx Residual Makeham Model")
    plt.xlabel("Age")
    plt.ylabel("Relative Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def fit_whittaker_henderson(df, year, sex_code, lambda_=1, pop_weights="n"):
    new_df = df.copy()

    model_df = new_df[
        (new_df["Year Code"] == year)
        & (new_df["Sex Code"] == sex_code)
    ].copy()

    model_df = model_df.sort_values("Single-Year Ages Code")

    model_df["ln_qx"] = np.log(model_df["qx"])
    
    y = model_df["ln_qx"].to_numpy()
    w = model_df["Population"].to_numpy()

    n = len(model_df)

    if pop_weights == "n":
        weights = list(np.ones(n))
    else:
        weights = w
    
    D = sparse.diags(
        diagonals=[np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
        offsets=[0, 1, 2],
        shape=(n-2, n)
    )

    W = sparse.diags(weights)

    A = W + lambda_ * (D.T @ D)
    b = W @ y

    z = spsolve(A, b)

    model_df["ln_qx_fitted_wh"] = z
    model_df["whittaker_residual"] = (
        model_df["ln_qx"] - model_df["ln_qx_fitted_wh"]
        )

    return model_df

def plot_whittaker_fit(whittaker_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        whittaker_df["Single-Year Ages Code"],
        whittaker_df["ln_qx"],
        label="Observed ln(qx)"
    )

    plt.plot(
        whittaker_df["Single-Year Ages Code"],
        whittaker_df["ln_qx_fitted_wh"],
        label="Fitted Whittaker ln(qx)",
        linestyle="--"
    )

    plt.title("Observed ln(qx) vs Whittaker Fitted ln(qx) Model")
    plt.xlabel("Age")
    plt.ylabel("ln(qx)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_whittaker_residual(whittaker_df):
    plt.figure(figsize=(8,5))

    plt.plot(
        whittaker_df["Single-Year Ages Code"],
        whittaker_df["whittaker_residual"],
        label="Whittaker Residual ln(qx)"
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=1
    )
    
    plt.title("ln(qx) Residual Whittaker Model")
    plt.xlabel("Age")
    plt.ylabel("ln(qx) Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visual_table(df):
    model_df = df.copy()

    model_df["ln_qx"] = np.log(model_df["qx"])

    model_df = model_df.sort_values(
        ["Sex Code", "Single-Year Ages Code", "Year Code"]
    )

    model_df["qx_yoy_pct"] = (
        model_df.groupby(["Sex Code", "Single-Year Ages Code"])["qx"]
        .pct_change()
        * 100
    )

    return model_df

def plot_heatmap(df, sex_code, column):
    model_df = df.copy()

    filtered_df = model_df[
        model_df["Sex Code"] == sex_code
    ]

    heatmap_df = filtered_df.pivot(
        index="Single-Year Ages Code",
        columns="Year Code",
        values=column
    )

    if column == "ln_qx":
        column_label = "ln(qx)"
    elif column == "qx_yoy_pct":
        column_label = "qx % Change Year over Year"
    else:
        column_label = column

    if sex_code == "T":
        sex_label = "Total"
    elif sex_code == "M":
        sex_label = "Male"
    elif sex_code == "F":
        sex_label = "Female"
    else:
        sex_label = sex_code

    if column == "qx_yoy_pct":
        cmap = "coolwarm"
    else:
        cmap = "viridis"

    plt.figure(figsize=(8, 6))

    plt.imshow(
        heatmap_df,
        aspect = "auto",
        origin = "lower",
        cmap = cmap
    )

    plt.colorbar(label=column_label)
    plt.title(f"{column_label} Heatmap by Age and Year - {sex_label}")
    plt.xlabel("Year")
    plt.ylabel("Age")

    plt.xticks(
        ticks = range(len(heatmap_df.columns)),
        labels = heatmap_df.columns
    )

    plt.yticks(
        ticks = range(len(heatmap_df.index)),
        labels = heatmap_df.index
    )

    plt.tight_layout()
    plt.show()
    
def main():
    mortality_df = load_data(INPUT_FILE)
    validation_df = filter_for_excel(mortality_df)
    life_df = add_life_tab_col(validation_df)
    
    avg_tab = build_avg_tab(life_df)
    create_avg_plots(avg_tab)

    gompertz, gompertz_df = fit_gompertz(life_df, 2019, "F")
    print(gompertz.summary())

    plot_gompertz_fit(gompertz_df)
    plot_gompertz_residual(gompertz_df)

    ma_df = build_ma_graduation(life_df, 2019, "F")
    plot_ln_ma(ma_df)
    plot_ma(ma_df)

    makeham_params, makeham_df = fit_makeham(life_df, 2019, "F")
    print(makeham_params)
    plot_makeham_fit(makeham_df)
    plot_makeham_residual(makeham_df)

    whittaker_df = fit_whittaker_henderson(life_df, 2019, "F")
    plot_whittaker_fit(whittaker_df)
    plot_whittaker_residual(whittaker_df)

    heatmap_table = visual_table(life_df)
    plot_heatmap(heatmap_table, "T", "ln_qx")
    plot_heatmap(heatmap_table, "F", "ln_qx")
    plot_heatmap(heatmap_table, "M", "ln_qx")
    plot_heatmap(heatmap_table, "T", "qx_yoy_pct")
    plot_heatmap(heatmap_table, "F", "qx_yoy_pct")
    plot_heatmap(heatmap_table, "M", "qx_yoy_pct")

if __name__ == "__main__":
    main()
