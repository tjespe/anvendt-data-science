# %%
import datetime
import pandas as pd
import numpy as np
from denormalize import denormalize_predictions
from preprocessing import preprocess_consumption_data, read_consumption_data
from train_test_split import (
    split_into_cv_folds_and_test_fold,
)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
import json

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# These are features that we are sure improves the model based on
# previous analysis. Thus, Optuna is not given the choice on
# whether or not to include them.
# All these have a p-value < 0.01 for reducing both MAPE and RMSE,
# and have been evaluated with a common sense check.
# **NB**: When this list is changed, any feature selection studies
# should be deleted so that Optuna forgets what it has learned.
manually_chosen_features = {
    "Normalized consumption 1 week ago",
    "Temperature",
    "Average temperature 4 to 6 hours ago",
    "Average temperature 7 to 12 hours ago",
    "Temperature one week ago",
    "Average temperature last week",
    "Average temperature two weeks ago",
    "Normalized mean consumption at same hour last 4 days",
    "Normalized mean consumption at same hour last 7 days",
}


def train_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame, **params):
    """
    Trains an XGBoost model.
    """
    model = xgb.XGBRegressor(
        verbosity=1, objective="reg:squarederror", booster="gbtree", **params
    )
    model.fit(X_train, y_train)
    return model


def predict_xgb(model: xgb.XGBRegressor, X_test: pd.DataFrame):
    """
    Predicts using an XGBoost model.
    """
    return model.predict(X_test)


if __name__ == "__main__":
    """
    This code is run when executing the file directly.
    """
    # %%
    raw_df = read_consumption_data()
    # %%
    # Drop helsingfors
    raw_df = raw_df[raw_df["Location"] != "helsingfors"]

    # %%
    do_feature_selection = True

    def objective(trial):
        """
        Use optuna to select hyperparameters
        """
        if do_feature_selection:
            use_normalization = True
            use_rolling = True
            rolling_normalization_window_days = 35
            num_splits = 10
            n_estimators = 500
            max_depth = 5
            learning_rate = 0.033
            subsample = 0.15
            colsample_bytree = 0.50
            gamma = 0.07
            reg_lambda = 0.3
        else:
            use_normalization = True  # trial.suggest_categorical("use_target_normalization", [True, False])
            use_rolling = False
            if use_normalization:
                use_rolling = trial.suggest_categorical(
                    "use_rolling_normalization", [True, False]
                )
            rolling_normalization_window_days = None
            if use_rolling:
                rolling_normalization_window_days = trial.suggest_int(
                    "rolling_normalization_window_days", 5, 70
                )
            # Based on results, it seemed like num_splits is not an important hyperparam,
            # so we set it to a fixes number instead
            num_splits = 10
            n_estimators = trial.suggest_int("n_estimators", 1, 500, log=True)
            max_depth = trial.suggest_int("max_depth", 1, 10, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
            subsample = trial.suggest_float("subsample", 0.1, 1)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1)
            gamma = trial.suggest_float("gamma", 0, 1)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)

        # %%
        print("Preprocessing data...")
        processed_df = preprocess_consumption_data(
            raw_df, rolling_normalization_window_days
        )
        # %%
        # Ensure first validation day is always the same (differences in
        # rolling_normalization_window_days parameter lead to different nan values,
        # so the start date becomes different and the tests become uncomparable
        fixed_start_date = datetime.date(2022, 4, 29)
        processed_df = processed_df[processed_df["Time"].dt.date >= fixed_start_date]
        print(processed_df["Time"].min())
        assert (min_date := processed_df["Time"].dt.date.min()) == fixed_start_date, (
            f"Expected first date in processed data to be {fixed_start_date}, but"
            f" was {min_date}"
        )
        # %%
        print("Splitting")
        folds = split_into_cv_folds_and_test_fold(
            processed_df,
            n_splits=num_splits,
            target_variable="Normalized consumption"
            if use_normalization
            else "Consumption",
        )
        # %%
        # Skip first fold because it has too little training data
        folds = folds[1:]
        # %%
        # Select features
        features = list(folds[0][0][0].columns)
        features_to_use = []
        if do_feature_selection:
            features_to_use += manually_chosen_features
            for feature in features:
                if feature in manually_chosen_features:
                    continue
                if trial.suggest_int(f"use_feature_{feature}", 0, 1):
                    features_to_use.append(feature)
        else:
            features_to_use = list(manually_chosen_features)
        print(json.dumps(trial.params, indent=4))
        if not features_to_use:
            return float("inf"), float("inf")
        # %%
        cv_folds = folds[:-1]
        results_dfs = []
        for i, (training, validation) in enumerate(cv_folds):
            print(f"CV fold {i}")
            X_train, y_train = training
            X_val, y_val = validation
            X_train = X_train[features_to_use]
            X_train = pd.get_dummies(X_train)
            model = train_xgb(
                X_train,
                y_train,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_lambda=reg_lambda,
            )
            X_val = X_val[features_to_use]
            X_val = pd.get_dummies(X_val)
            X_val = X_val.reindex(columns=X_train.columns, fill_value=0)[
                X_train.columns
            ]
            y_val_pred = predict_xgb(model, X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            print(f"RMSE (normalized): {rmse}")

            results_df = pd.DataFrame(
                {
                    "actual": y_val,
                    "prediction": y_val_pred,
                },
                index=y_val.index,
            )
            if use_normalization:
                denormalized_results_df = denormalize_predictions(
                    results_df, raw_df, rolling_normalization_window_days
                )
            else:
                denormalized_results_df = results_df
            denormalized_results_df["fold"] = i
            results_dfs.append(denormalized_results_df)

        # %%
        # Merge result dataframes
        results_df = pd.concat(results_dfs)

        # %%
        # Calculate RMSE for denormalized data
        rmse = np.sqrt(
            mean_squared_error(results_df["actual"], results_df["prediction"])
        )
        print(f"RMSE (denormalized): {rmse}")

        # Calculate mean absolute percentage error for denormalized data
        results_df["PE"] = 100 * (
            (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
        )
        results_df["APE"] = np.abs(results_df["PE"])
        mape = results_df["APE"].mean()
        print(f"MAPE: {results_df['APE'].mean()}%")
        print(
            "Location stats\n",
            results_df.groupby(
                results_df.index.get_level_values("Location"), observed=True
            )[["APE", "PE"]].mean(),
        )
        fold_stats = (
            results_df.reset_index().groupby("fold")["Time"].agg(["min", "max"])
        )
        fold_stats.columns = ["From", "To"]
        fold_stats["MAPE"] = results_df.groupby("fold")["APE"].mean()
        fold_stats["MPE"] = results_df.groupby("fold")["PE"].mean()
        fold_stats.From = fold_stats.From.dt.strftime("%d. %b")
        fold_stats.To = fold_stats.To.dt.strftime("%d. %b")
        print(fold_stats)

        # %%
        # Look at performance in Oslo
        # for date in results_df.reset_index()["Time"].dt.date.unique():
        #     values_on_date = results_df.reset_index().loc[
        #         pd.Series(results_df.index.get_level_values("Time")).dt.date == date
        #     ]
        #     values_on_date[values_on_date["Location"] == "oslo"].set_index("Time")[
        #         ["actual", "prediction"]
        #     ].plot()

        # %%
        return rmse, mape

    # %%
    sampler = None  # optuna.samplers.NSGAIIISampler()
    study_name = f"XGBoost consumption prediction {'feature selection ' if do_feature_selection else ''}{sampler.__class__.__name__}"
    print(study_name)
    # %%
    try:
        study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")
    except KeyError as e:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=study_name,
            sampler=sampler,
        )

    study.optimize(objective, n_trials=100)


# %%
