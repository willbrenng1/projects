def show_end_state_circle_chart(df):
    import pandas as pd
    import matplotlib.pyplot as plt

    end_state_count = pd.DataFrame(
        df.drop_duplicates(subset="id")["term_reason"].value_counts()
    ).reset_index(drop=False)
    colors = ["#AEC7E8", "#FFBB78", "#98DF8A"]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        end_state_count["count"],
        labels=end_state_count["term_reason"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops={"fontsize": 12},
        wedgeprops=dict(width=0.4, edgecolor="w"),
    )

    plt.setp(autotexts, size=12, weight="bold", color="black")
    plt.title("Game of Life Termination Outcomes", fontsize=14, weight="bold", pad=20)
    ax.axis("equal")

    plt.tight_layout()
    plt.show()


def show_random_sample_runs(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    runs_with_multiple_gens = df.groupby("run").filter(lambda x: x["gen"].nunique() > 1)
    sampled_runs = runs_with_multiple_gens["id"].sample(6, random_state=None)
    sample_df = df[df["id"].isin(sampled_runs)]
    sample_df = sample_df.copy()
    sample_df[" "] = sample_df.apply(
        lambda row: f"Size: {row['matrix_size']} | Initial % Alive: {row['initial_percent_alive']:.2f} | Term Reason: {row["term_reason"]}",
        axis=1,
    )

    g = sns.relplot(
        data=sample_df,
        x="gen",
        y="alive_cells",
        kind="line",
        col=" ",
        col_wrap=3,
        markers="O",
    )
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    g.set_axis_labels("Generation", "Alive Cells")
    g.fig.suptitle("Live Cell Count Over Generations", y=1.02)
    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_title(ax.get_title().replace("=", "").strip())

    plt.show()


def show_heat_map(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    heatmap_data = (
        df.groupby(["matrix_size", "initial_percent_alive", "run"], as_index=False)[
            "gen"
        ]
        .max()
        .pivot_table(
            index="matrix_size",
            columns="initial_percent_alive",
            values="gen",
            aggfunc="mean",
        )
    )
    plt.figure(figsize=(20, 20))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=0.5)

    plt.title(
        "Average Generation Until Termination by Matrix Size and Initial Percent Alive"
    )
    plt.xlabel("Initial Percent Alive")
    plt.ylabel("Matrix Size")
    plt.tight_layout()
    plt.show()


def show_initial_percent_alive_ci_avg(df):

    import seaborn as sns
    import matplotlib.pyplot as plt
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
    sample_grouped_initial_percent_alive = sample_grouped_stats.groupby(
        ["initial_percent_alive"], as_index=False
    ).agg(avg_ci=("ci_95", "mean"))
    sns.barplot(
        data=sample_grouped_initial_percent_alive,
        x="initial_percent_alive",
        y="avg_ci",
        hue="initial_percent_alive",
    )

    plt.title("Average 95% Confidence Interval by Initial Percent Alive")
    plt.xlabel("Initial % Alive")
    plt.ylabel("Average 95% CI Width")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_ci_delta(df, top_bottom, n):

    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import t
    import numpy as np

    dfs = df.copy()
    dfs["id"] = (
        dfs[["matrix_size", "initial_percent_alive"]].astype(str).agg("-".join, axis=1)
    )
    if top_bottom == "top":
        plot_df = dfs.sort_values("ci_delta", ascending=False).head(n)
    else:
        plot_df = dfs.sort_values("ci_delta", ascending=False).tail(n)

    sns.barplot(
        data=plot_df,
        x="id",
        y="ci_delta",
        errorbar=None,
        hue="ci_delta",
    )
    plt.legend().remove()
    plt.title("Changes in CI Width Between Sample and Final Simulation")
    plt.xlabel("Matrix Size and Initial % Alive")
    plt.ylabel("Change in 95% CI Width")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def difference_in_average_ci_percent_alive_groups(df, df2):

    import seaborn as sns
    import matplotlib.pyplot as plt
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
    sample_grouped_initial_percent_alive = sample_grouped_stats.groupby(
        ["initial_percent_alive"], as_index=False
    ).agg(avg_ci=("ci_95", "mean"))

    ##
    final_grouped = df2.groupby(
        ["matrix_size", "initial_percent_alive", "run"], as_index=False
    )["gen"].max()
    final_grouped_stats = final_grouped.groupby(
        ["matrix_size", "initial_percent_alive"], as_index=False
    ).agg(gen_mean=("gen", "mean"), gen_std=("gen", "std"), n=("gen", "count"))
    final_grouped_stats["gen_standard_error"] = final_grouped_stats[
        "gen_std"
    ] / np.sqrt(final_grouped_stats["n"])
    final_grouped_stats["ci_95"] = (
        t.ppf(0.975, final_grouped_stats["n"] - 1)
        * final_grouped_stats["gen_standard_error"]
    )
    final_grouped_stats["matrix_size"] = final_grouped_stats["matrix_size"].astype(str)
    final_grouped_initial_percent_alive = final_grouped_stats.groupby(
        ["initial_percent_alive"], as_index=False
    ).agg(avg_ci=("ci_95", "mean"))

    merged_df = final_grouped_initial_percent_alive.merge(
        sample_grouped_initial_percent_alive, on="initial_percent_alive", how="left"
    )
    merged_df = merged_df.rename(
        {"avg_ci_x": "final_ci", "avg_ci_y": "sample_ci"}, axis=1
    )

    merged_df["difference_between_cis"] = merged_df["final_ci"] - merged_df["sample_ci"]
    sns.barplot(
        data=merged_df,
        x="initial_percent_alive",
        y="difference_between_cis",
    )

    plt.title(
        "Changes in CI Width of Initial Percent Alive Groups Between Sample and Final Simulation"
    )
    plt.xlabel("Initial % Alive")
    plt.ylabel("Change in Average 95% Confidence Interval")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return merged_df
