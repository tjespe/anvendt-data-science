# %%
import optuna
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    This file analyzes the results from Optuna's hyperparameter tuning.
    """
    # %%
    study_name = "XGBoost consumption prediction"
    study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")

    # %%
    # Plot Pareto front of the two objective values
    fig = optuna.visualization.plot_pareto_front(
        study,
        targets=lambda t: (t.values[0], t.values[1]),
        target_names=["RMSE", "MAPE"],
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
    # Show MAPE for different combos of n_estimators and max_depth
    fig = optuna.visualization.plot_contour(
        study,
        target=lambda t: t.values[1],
        params=["n_estimators", "max_depth"],
        target_name="MAPE",
    )
    fig.show()

    # %%
    # Show how RMSE changes for different rolling_normalization_window_days
    relevant_trials = sorted(
        [
            trial
            for trial in study.trials
            if "rolling_normalization_window_days" in trial.params and trial.values
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
    # Check effect of using rolling_normalization_window_days instead of all historical data
    df = pd.DataFrame([trial.params["use_norma"]])

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
