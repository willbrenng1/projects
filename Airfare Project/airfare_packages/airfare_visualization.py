import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations


def visualize_data_distribution(df):

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols
        and "date" not in col
        and "id" not in col
        and "time" not in col
        and "code" not in col
    ]
    date_cols = [col for col in df.columns if col and "date" in col]
    time_cols = [col for col in df.columns if col and "time" in col]

    numeric_n = len(numeric_cols)

    numeric_rows = int(np.ceil(numeric_n / 1))  # 3 plots per row
    numeric_fig = plt.figure(figsize=(15, numeric_rows * 3))
    numeric_gs = gridspec.GridSpec(numeric_rows, 1, wspace=0.4, hspace=0.6)

    numeric_fig.suptitle("Numeric Data Type Distributions", fontsize=16, y=1.02)

    for i, col in enumerate(numeric_cols):
        ax = numeric_fig.add_subplot(numeric_gs[i])
        ax.hist(df[col].dropna(), bins=20, color="green", edgecolor="black")
        ax.set_title(col)
        ax.set_ylabel("Frequency")

    plt.show()

    cat_n = len(categorical_cols)
    cat_rows = int(np.ceil(cat_n / 1))
    cat_fig = plt.figure(figsize=(15, cat_rows * 3))
    cat_gs = gridspec.GridSpec(cat_rows, 1, wspace=0.4, hspace=0.6)

    cat_fig.suptitle("Categorical Data Type Distributions", fontsize=16, y=1.02)

    for i, col in enumerate(categorical_cols):
        grouped_df = df.groupby(col, as_index=False).size()
        ax = cat_fig.add_subplot(cat_gs[i])
        ax.bar(
            grouped_df[col].dropna(),
            grouped_df["size"],
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(col)
        ax.set_ylabel("Count")

    plt.show()

    date_n = len(date_cols)
    date_rows = int(np.ceil(date_n / 1))
    date_fig = plt.figure(figsize=(15, date_rows * 3))
    date_gs = gridspec.GridSpec(date_rows, 1, wspace=0.4, hspace=0.9)

    date_fig.suptitle("Date Data Distributions", fontsize=16, y=1.02)

    for i, col in enumerate(date_cols):
        df[col] = df[col].astype(str)

        date_grouped_df = df.groupby(col, as_index=False).size()

        ax = date_fig.add_subplot(date_gs[i])
        ax.bar(
            date_grouped_df[col].dropna(),
            date_grouped_df["size"],
            color="Purple",
            edgecolor="black",
        )
        ax.set_title(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=90)

    plt.show()

    time_n = len(time_cols)
    time_rows = int(np.ceil(time_n / 1))
    time_fig = plt.figure(figsize=(15, time_rows * 3))
    time_gs = gridspec.GridSpec(time_rows, 1, wspace=0.4, hspace=0.6)

    time_fig.suptitle("Time Data Distributions", fontsize=16, y=1.02)

    for i, col in enumerate(time_cols):
        df[col] = df[col].astype(str)
        time_series = pd.to_datetime(
            df[col], format="%H:%M:%S", errors="coerce"
        ).dt.time
        hours = [t.hour for t in time_series.dropna()]

        ax = time_fig.add_subplot(time_gs[i])
        ax.hist(
            hours,
            bins=range(25),
            color="Yellow",
            edgecolor="black",
            align="left",
            rwidth=0.8,
        )

        ax.set_title(f"{col}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, 24))
        ax.set_xlim(-0.5, 23.5)
    plt.show()


def visualize_cramers_v(df):

    def cramers_v(x, y):

        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k - 1, r - 1))

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols
        and "date" not in col
        and "id" not in col
        and "time" not in col
        and "code" not in col
    ]
    cat_col_combos = list(combinations(categorical_cols, 2))

    cramers_v_res = pd.DataFrame(columns=categorical_cols, index=categorical_cols)

    for e in cat_col_combos:
        val = cramers_v(df[e[0]], df[e[1]])
        cramers_v_res.loc[e[0], e[1]] = val
        cramers_v_res.loc[e[1], e[0]] = val

    cramers_v_res = cramers_v_res.astype(float).fillna(1)

    sns.heatmap(cramers_v_res, annot=True, cmap="YlGnBu")
    plt.title("Categorical Variables - Cramer's V")
    return cramers_v_res


def visualize_numeric_vars_correlation(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)

    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGnBu")
    plt.title("Numeric Variables - Correlation")

    return df[numeric_cols].corr()


def visualize_numeric_categorical_relationship(df):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols
        and "date" not in col
        and "id" not in col
        and "time" not in col
        and "code" not in col
    ]
    eta_sq_res = pd.DataFrame(columns=categorical_cols, index=numeric_cols)

    for cat in categorical_cols:
        for nums in numeric_cols:

            formula = f'{nums} ~ C(Q("{cat}"))'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            total_ss = anova_table["sum_sq"].sum()
            anova_table["eta_sq"] = anova_table["sum_sq"] / total_ss
            factor_name = f'C(Q("{cat}"))'
            eta_sq_res.loc[rf"{nums}", rf"{cat}"] = anova_table.loc[
                factor_name, "eta_sq"
            ]

    eta_sq_res_numeric = eta_sq_res.astype(float)
    sns.heatmap(eta_sq_res_numeric, annot=True, cmap="YlGnBu")
    plt.title("Numeric v Categorical Variables - Eta Squared")
    return eta_sq_res


def visualization_predictor_vif(df):

    vif_df = df.drop(columns=["id", "price"])
    vif_data = pd.DataFrame()
    vif_data["feature"] = vif_df.columns

    vif_data["VIF"] = [
        variance_inflation_factor(vif_df.values, i) for i in range(len(vif_df.columns))
    ]

    vif_data = vif_data.sort_values("VIF")

    numeric_fig = plt.figure(figsize=(40, 15))
    numeric_gs = gridspec.GridSpec(2, 1, wspace=0.8, hspace=2)

    numeric_fig.suptitle("VIF Scores", fontsize=27, y=1.15)

    for i in range(2):
        if i == 0:

            ax = numeric_fig.add_subplot(numeric_gs[i])
            ax.bar(
                vif_data.head(10)["feature"],
                vif_data.head(10)["VIF"],
                edgecolor="black",
            )
            ax.set_title("Lowest 10 VIF Scores", fontsize=25)
        else:
            ax = numeric_fig.add_subplot(numeric_gs[i])
            ax.bar(
                vif_data.tail(10)["feature"],
                vif_data.tail(10)["VIF"],
                edgecolor="black",
            )
            ax.set_title("Highest 10 VIF Scores", fontsize=25)

        ax.set_ylabel("VIF", fontsize=25)
        ax.tick_params(axis="x", labelrotation=45, labelsize=25)
        ax.tick_params(axis="y", labelsize=25)

    plt.show()

    return vif_data


def visualize_model_comparison_metrics(dicts, top_n_models):
    try:
        metric_dict = {
            k: v["model_metrics"]["precision_stats"] for k, v in dicts.items()
        }
    except KeyError:
        metric_dict = dicts
    metric_df = pd.DataFrame.from_dict(metric_dict, orient="index").reset_index()
    metric_df = metric_df.rename(columns={"index": "model"})
    metric_cols = ["adj_r2", "aic", "bic", "rmse", "mse", "mae"]

    for metric in metric_cols:
        plt.figure(figsize=(8, 4))
        plt.bar(
            metric_df["model"], metric_df[metric], color="steelblue", edgecolor="black"
        )
        plt.ylim([metric_df[metric].min() * 0.95, metric_df[metric].max() * 1.05])
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.show()

    ranking_df = metric_df.copy()
    ranking_df["adj_r2_rank"] = ranking_df["adj_r2"].rank(ascending=False)
    ranking_df["aic_rank"] = ranking_df["aic"].rank()
    ranking_df["bic_rank"] = ranking_df["bic"].rank()
    ranking_df["rmse_rank"] = ranking_df["rmse"].rank()
    ranking_df["mse_rank"] = ranking_df["mse"].rank()
    ranking_df["mae_rank"] = ranking_df["mae"].rank()

    ranking_df["avg_rank"] = (
        ranking_df[
            ["adj_r2_rank", "aic_rank", "bic_rank", "rmse_rank", "mse_rank", "mae_rank"]
        ]
        .mean(axis=1)
        .round(2)
    )

    return ranking_df.sort_values("avg_rank").head(top_n_models)


def plot_linear_regression_diagnostics(residual, y_predictions):

    plt.figure(figsize=(8, 5))
    sns.residplot(
        x=np.asarray(y_predictions),
        y=np.asarray(residual),
        lowess=True,
        line_kws={"color": "red"},
    )
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.axhline(0, linestyle="--", color="grey")
    plt.show()

    stats.probplot(residual, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()

    sns.histplot(residual, kde=True)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.show()

    plt.scatter(y_predictions, np.sqrt(np.abs(residual)))
    plt.xlabel("Fitted values")
    plt.ylabel("√|Residuals|")
    plt.title("Scale-Location")
    plt.axhline(0, linestyle="--", color="grey")
    plt.show()
