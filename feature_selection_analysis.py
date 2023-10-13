# %%
import json
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


if __name__ == "__main__":
    """
    This file analyzes the results from Optuna's feature selection tuning.
    """
    # %%
    sampler = optuna.samplers.TPESampler()
    study_name = (
        f"XGBoost consumption prediction feature selection {sampler.__class__.__name__}"
    )
    study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")
    filtered_trials = lambda: [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values
    ]

    # %%
    # Plot Pareto front of the two objective values
    fig = optuna.visualization.plot_pareto_front(
        study,
        targets=lambda t: (t.values[0], t.values[1]),
        target_names=["RMSE", "MAPE"],
    )
    fig.show()

    # %%
    # Check effect of including each feature
    all_params = set(
        param for trial in filtered_trials() for param in trial.params.keys()
    )
    feature_params = [param for param in all_params if param.startswith("use_feature_")]
    df = pd.DataFrame(
        [
            [
                trial.values[0],
                trial.values[1],
            ]
            + [trial.params.get(param) for param in feature_params]
            for trial in filtered_trials()
            if trial.values
        ],
        columns=["RMSE", "MAPE", *feature_params],
    )
    feature_inclusion_stats = pd.DataFrame(
        columns=[
            "Mean (Include)",
            "Mean (Exclude)",
            "p-value",
            "t stat",
            "n (Include)",
            "n (Exclude)",
            "Feature",
            "Metric",
        ],
    )
    feature_inclusion_stats.set_index(["Feature", "Metric"], inplace=True)
    for param in feature_params:
        for metric in ["RMSE", "MAPE"]:
            grouped = df.groupby(param)[metric]
            means = grouped.agg("mean")
            counts = grouped.agg("count")
            t_stat, p_value = stats.ttest_ind(
                grouped.get_group(0),
                grouped.get_group(1),
                equal_var=True,
                alternative="greater",
            )
            feature = param.replace("use_feature_", "")
            feature_inclusion_stats.loc[(feature, metric), :] = [
                float(means.loc[1]),
                float(means.loc[0]),
                float(p_value),
                float(t_stat),
                float(counts.loc[1]),
                float(counts.loc[0]),
            ]
    feature_inclusion_stats.sort_values(by="p-value")

    # %%
    print(
        "Statistically significant features (p < 0.05):",
        json.dumps(
            list(
                set(
                    feature_inclusion_stats[
                        feature_inclusion_stats["p-value"] < 0.05
                    ].index.get_level_values("Feature")
                )
            ),
            indent=4,
        ),
    )

    # %%
    print(
        "Features that reduced error:",
        json.dumps(
            list(
                set(
                    feature_inclusion_stats[
                        feature_inclusion_stats["Mean (Include)"]
                        < feature_inclusion_stats["Mean (Exclude)"]
                    ].index.get_level_values("Feature")
                )
            ),
            indent=4,
        ),
    )

    # %%
    # Show hyperparameter importance for RMSE
    fig = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="RMSE"
    )
    fig.show()

    # %%
    # Show hyperparameter importance for MAPE
    fig = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[1], target_name="MAPE"
    )
    fig.show()

# %%
