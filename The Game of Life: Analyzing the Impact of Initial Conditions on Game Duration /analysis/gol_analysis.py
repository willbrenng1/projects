def cohens_f(f_stat, df_groups, df_obs):
    import numpy as np

    eta_squared = (f_stat * df_groups) / (f_stat * df_groups + df_obs)
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
    return cohens_f


def anova_analysis(df):
    import scipy.stats as stats
    import pandas as pd
    import seaborn as sns

    anova_results = []
    summary_df = df.groupby(
        ["matrix_size", "initial_percent_alive", "run"], as_index=False
    )["gen"].max()
    for size in summary_df["matrix_size"].unique():
        size_df = summary_df[summary_df["matrix_size"] == size]
        percent_alive_grouped_df = size_df.groupby(
            ["initial_percent_alive", "run"], as_index=False
        )["gen"].max()
        grouped_lists = percent_alive_grouped_df.groupby("initial_percent_alive")[
            "gen"
        ].apply(list)

        f_stat, p_value = stats.f_oneway(*grouped_lists)
        grouped_lists
        degree_of_freedom_groups = len(grouped_lists) - 1
        degree_of_freedom_observations = size_df.shape[0] - len(grouped_lists)
        con_f = cohens_f(
            f_stat, degree_of_freedom_groups, degree_of_freedom_observations
        )
        anova_results.append(
            {
                "matrix_size": size,
                "f_stat": f_stat,
                "p_value": p_value,
                "cohen_f": con_f,
            }
        )
    result_df = pd.DataFrame(anova_results)
    return result_df


def calc_matrix_size_and_initial_percent_alive_group_stats(df):
    from scipy.stats import t
    import numpy as np

    sample_grouped = df.groupby(
        ["matrix_size", "initial_percent_alive", "run"], as_index=False
    )["gen"].max()
    sample_grouped_stats = sample_grouped.groupby(
        ["matrix_size", "initial_percent_alive"], as_index=False
    ).agg(gen_mean=("gen", "mean"), gen_std=("gen", "std"), n=("gen", "count"))
    sample_grouped_stats["gen_standard_error"] = sample_grouped_stats[
        "gen_std"
    ] / np.sqrt(sample_grouped_stats["n"])
    sample_grouped_stats["ci_95"] = (
        t.ppf(0.975, sample_grouped_stats["n"] - 1)
        * sample_grouped_stats["gen_standard_error"]
    )
    sample_grouped_stats["matrix_size"] = sample_grouped_stats["matrix_size"].astype(
        str
    )
    return sample_grouped_stats.sort_values(["ci_95"], ascending=False)


def required_n_per_group(grouped_stats, min_sample_size, hw=10, c=0.95):
    from scipy.stats import norm, t
    import numpy as np

    sample_n = grouped_stats["n"].unique()[0]
    alpha = 1 - c
    if sample_n >= 30:
        value = norm.ppf(1 - alpha / 2)
    else:
        degree_of_freedom = sample_n - 1
        value = t.ppf(1 - alpha / 2, degree_of_freedom)

    df = grouped_stats.copy()
    df["required_n"] = ((value * df["gen_std"]) / hw) ** 2
    df["required_n"] = np.ceil(df["required_n"]).clip(lower=min_sample_size).astype(int)

    required_n_dict = {
        (int(row["matrix_size"]), row["initial_percent_alive"]): row["required_n"]
        for i, row in df.iterrows()
    }

    return required_n_dict


def sample_v_final_stat_comparison(sample, final):
    merged_group_stats = sample.merge(
        final, on=["matrix_size", "initial_percent_alive"], how="left"
    )
    merged_group_stats.columns = [
        "matrix_size",
        "initial_percent_alive",
        "sample_gen_mean",
        "sample_gen_std",
        "sample_gen_n",
        "sample_gen_standard_error",
        "sample_gen_ci_95",
        "final_gen_mean",
        "final_gen_std",
        "final_gen_n",
        "final_gen_standard_error",
        "final_gen_ci_95",
    ]
    merged_group_stats["mean_delta"] = merged_group_stats.apply(
        lambda x: x["sample_gen_mean"] - x["final_gen_mean"], axis=1
    )
    merged_group_stats["std_delta"] = merged_group_stats.apply(
        lambda x: x["sample_gen_std"] - x["final_gen_std"], axis=1
    )
    merged_group_stats["standard_error_delta"] = merged_group_stats.apply(
        lambda x: x["sample_gen_standard_error"] - x["final_gen_standard_error"], axis=1
    )
    merged_group_stats["ci_delta"] = merged_group_stats.apply(
        lambda x: x["sample_gen_ci_95"] - x["final_gen_ci_95"], axis=1
    )
    return merged_group_stats
