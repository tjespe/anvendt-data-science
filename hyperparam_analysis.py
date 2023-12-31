# %%
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


if __name__ == "__main__":
    """
    This file analyzes the results from Optuna's hyperparameter tuning.
    """
    # %%
    sampler = None  # optuna.samplers.NSGAIIISampler()
    include_location = False
    study_name = f"XGBoost consumption prediction {'w/ location ' if include_location else ''}{sampler.__class__.__name__}"
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
    # Show RMSE for different combos of learning_rate and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[0],
        params=["learning_rate", "max_depth"],
        target_name="RMSE",
    )
    fig.show()

    # %%
    # Show MAPE for different combos of learning_rate and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[1],
        params=["learning_rate", "max_depth"],
        target_name="MAPE",
    )
    fig.show()

    # %%
    # Show RMSE for different combos of n_estimators and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[0],
        params=["n_estimators", "max_depth"],
        target_name="RMSE",
    )
    fig.show()

    # %%
    # Show RMSE for different combos of n_estimators and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[0],
        params=["subsample", "colsample_bytree"],
        target_name="RMSE",
    )
    fig.show()

    # %%
    # Show RMSE for different combos of n_estimators and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[0],
        params=["gamma", "reg_lambda"],
        target_name="RMSE",
    )
    fig.show()

    # %%
    # Show how RMSE changes for different rolling_normalization_window_days
    relevant_trials = sorted(
        [
            trial
            for trial in filtered_trials()
            if "rolling_normalization_window_days" in trial.params
        ],
        key=lambda trial: trial.params["rolling_normalization_window_days"],
    )
    x = [trial.params["rolling_normalization_window_days"] for trial in relevant_trials]
    y = [trial.values[0] for trial in relevant_trials]
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xlabel("Rolling normalization window (days)")
    plt.ylabel("RMSE")
    plt.show()

    # %%
    # Show how RMSE and MAPE changes for the different parameters
    for hyperparam in [
        # "rolling_normalization_window_days",
        # "num_splits",
        # "n_estimators",
        # "max_depth",
        # "learning_rate",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_lambda",
    ]:
        relevant_trials = sorted(
            [
                trial
                for trial in filtered_trials()
                if trial.values and hyperparam in trial.params
            ],
            key=lambda trial: trial.params[hyperparam],
        )
        fig, ax1 = plt.subplots()
        x = pd.Series([trial.params[hyperparam] for trial in relevant_trials])
        y1 = pd.Series([trial.values[0] for trial in relevant_trials])
        # ax1.plot(x, y1, label="RMSE")
        ax1.scatter(x, y1)
        ax1.set_xscale("log")
        plt.xlabel(hyperparam)
        ax1.set_ylabel("RMSE")
        ax2 = ax1.twinx()
        y2 = pd.Series([trial.values[1] for trial in relevant_trials])
        # ax2.plot(x, y2, c="red", label="MAPE")
        ax2.scatter(x, y2, c="red")
        ax2.set_ylabel("MAPE")
        plt.show()

        # Calculating Pearson correlation for hyperparam and RMSE
        correlation_rmse, p_value_rmse = stats.pearsonr(x, y1)
        print(
            f"Pearson correlation coefficient between {hyperparam} and RMSE: {correlation_rmse}"
        )
        print(f"P-value: {p_value_rmse}")

        # Calculating Pearson correlation for hyperparam and MAPE
        correlation_mape, p_value_mape = stats.pearsonr(x, y2)
        print(
            f"Pearson correlation coefficient between {hyperparam} and MAPE: {correlation_mape}"
        )
        print(f"P-value: {p_value_mape}")

    # %%
    # Check effect of using rolling_normalization_window_days instead of all historical data
    df = pd.DataFrame(
        [
            (
                trial.params.get("use_target_normalization", True),
                trial.params.get("use_rolling_normalization", None),
                trial.values[0],
                trial.values[1],
            )
            for trial in filtered_trials()
            if trial.values
        ],
        columns=["use_normalization", "use_rolling_normalization", "RMSE", "MAPE"],
    )
    for metric in ["RMSE", "MAPE"]:
        print(f"Mean {metric} when using rolling window normalization vs. expanding")
        print(
            df[df["use_normalization"] == True]
            .groupby("use_rolling_normalization")[metric]
            .agg("mean")
        )
        grouped = df[df["use_normalization"] == True].groupby(
            "use_rolling_normalization"
        )[metric]
        rmse_no_rolling = grouped.get_group(False)
        rmse_rolling = grouped.get_group(True)
        t_stat_rmse, p_value_rmse = stats.ttest_ind(
            rmse_no_rolling, rmse_rolling, equal_var=False, alternative="greater"
        )
        print("p-value that rolling is best:", p_value_rmse)

    print()
    for metric in ["RMSE", "MAPE"]:
        print(f"Mean {metric} when using normalization vs. no normalization")
        grouped = df.groupby("use_normalization")[metric]
        print(grouped.agg("mean"))
        rmse_no_norming = grouped.get_group(False)
        rmse_norming = grouped.get_group(True)
        t_stat_rmse, p_value_rmse = stats.ttest_ind(
            rmse_no_norming, rmse_norming, equal_var=False, alternative="greater"
        )
        print("p-value that norming is best:", p_value_rmse)

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
