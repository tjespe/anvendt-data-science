# %%
import json
import numpy as np
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


if __name__ == "__main__":
    """
    This file analyzes the results from Optuna's feature selection tuning.
    """
    # %%
    # Samplers that work in this case include:
    # - NSGAIISampler
    # - TPESampler
    # Change the sampler both here and in `xgb.py` to run different studies
    sampler = None
    study_name = (
        f"XGBoost consumption prediction feature selection {sampler.__class__.__name__}"
    )
    storage = "sqlite:///optuna.db"
    study = optuna.load_study(study_name=study_name, storage=storage)
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
    params = [param for param in all_params if param.startswith("use_feature_")]
    features = [param.replace("use_feature_", "") for param in params]
    df = pd.DataFrame(
        [
            [
                trial.values[0],
                trial.values[1],
            ]
            + [trial.params.get(param) for param in params]
            for trial in filtered_trials()
            if trial.values
        ],
        columns=["RMSE", "MAPE", *features],
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
    for feature in features:
        for metric in ["RMSE", "MAPE"]:
            grouped = df.groupby(feature)[metric]
            means = grouped.agg("mean")
            counts = grouped.agg("count")
            if len(grouped.groups) > 1:
                t_stat, p_value = stats.ttest_ind(
                    grouped.get_group(0),
                    grouped.get_group(1),
                    equal_var=True,
                    alternative="greater",
                )
            else:
                continue
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
        "Number of unique combinations tested:",
        len(set(tuple(vals) for vals in df.values[:, 2:])),
    )

    # %%
    # List significant features
    for level, thresh in [("Significant", 0.05), ("Strongly significant", 0.01)]:
        signifcant_MAPE_p_value = set(
            feature_inclusion_stats[
                (feature_inclusion_stats["p-value"] < thresh)
                & (feature_inclusion_stats.index.get_level_values("Metric") == "MAPE")
            ].index.get_level_values("Feature")
        )
        signifcant_RMSE_p_value = set(
            feature_inclusion_stats[
                (feature_inclusion_stats["p-value"] < thresh)
                & (feature_inclusion_stats.index.get_level_values("Metric") == "RMSE")
            ].index.get_level_values("Feature")
        )
        significant = signifcant_MAPE_p_value & signifcant_RMSE_p_value
        print(
            f"{level} features (both MAPE p value and RMSE p value < {thresh}):",
            json.dumps(
                list(significant),
                indent=4,
            ),
        )

    # %%
    # List significant features (using OR)
    for level, thresh in [("Significant", 0.05), ("Strongly significant", 0.01)]:
        signifcant_MAPE_p_value = set(
            feature_inclusion_stats[
                (feature_inclusion_stats["p-value"] < thresh)
                | (feature_inclusion_stats.index.get_level_values("Metric") == "MAPE")
            ].index.get_level_values("Feature")
        )
        signifcant_RMSE_p_value = set(
            feature_inclusion_stats[
                (feature_inclusion_stats["p-value"] < thresh)
                | (feature_inclusion_stats.index.get_level_values("Metric") == "RMSE")
            ].index.get_level_values("Feature")
        )
        significant = signifcant_MAPE_p_value & signifcant_RMSE_p_value
        print(
            f"{level} features (either MAPE p value or RMSE p value < {thresh}):",
            json.dumps(
                list(significant),
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
