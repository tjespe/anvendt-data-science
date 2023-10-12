# %%
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
warnings.filterwarnings("ignore", category=FutureWarning)


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
):
    """
    Trains an XGBoost model.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,  # 100 is default
        max_depth=max_depth,  # 3 is default
        learning_rate=learning_rate,  # 0.1 is default
        verbosity=1,
        objective="reg:squarederror",
        booster="gbtree",
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
    raw_df = raw_df[raw_df["location"] != "helsingfors"]

    # %%
    def objective(trial):
        """
        Use optuna to select hyperparameters
        """
        use_rolling = trial.suggest_categorical(
            "use_rolling_normalization", [True, False]
        )
        rolling_normalization_window_days = None
        if use_rolling:
            rolling_normalization_window_days = trial.suggest_int(
                "rolling_normalization_window_days", 5, 100
            )
        # Based on results, it seemed like num_splits is not an important hyperparam,
        # so we set it to a fixes number instead
        num_splits = 10
        n_estimators = trial.suggest_int("n_estimators", 1, 10_000, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 40, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-9, 1, log=True)
        print(json.dumps(trial.params, indent=4))

        # %%
        print("Preprocessing data...")
        processed_df = preprocess_consumption_data(
            raw_df, rolling_normalization_window_days
        )
        # %%
        print("Splitting")
        folds = split_into_cv_folds_and_test_fold(processed_df, n_splits=num_splits)
        cv_folds = folds[:-1]
        results_dfs = []
        for i, (training, validation) in enumerate(cv_folds):
            print(f"CV fold {i}")
            X_train, y_train = training
            X_val, y_val = validation
            X_train = pd.get_dummies(X_train)
            model = train_xgb(
                X_train,
                y_train,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
            )
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
            denormalized_results_df = denormalize_predictions(
                results_df, raw_df, rolling_normalization_window_days
            )
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
        print(f"RMSE: {rmse}")

        # Calculate mean absolute percentage error for denormalized data
        results_df["PE"] = 100 * (
            (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
        )
        results_df["APE"] = np.abs(results_df["PE"])
        mape = results_df["APE"].mean()
        print(f"MAPE: {results_df['APE'].mean()}%")
        print(
            "MAPE per",
            results_df.groupby(
                results_df.index.get_level_values("location"), observed=True
            )["APE"].mean(),
        )
        print("MAPE per fold", results_df.groupby("fold")["APE"].mean())
        print("\nMPE per fold", results_df.groupby("fold")["PE"].mean())

        # %%
        # Print info about the fold
        print(
            "Fold time info:\n",
            results_df.reset_index().groupby("fold")["time"].agg(["min", "max"]),
        )

        # %%
        # Look at performance in Oslo
        # for date in results_df.reset_index()["time"].dt.date.unique():
        #     values_on_date = results_df.reset_index().loc[
        #         pd.Series(results_df.index.get_level_values("time")).dt.date == date
        #     ]
        #     values_on_date[values_on_date["location"] == "oslo"].set_index("time")[
        #         ["actual", "prediction"]
        #     ].plot()

        # %%
        return rmse, mape

    study_name = "XGBoost consumption prediction"
    try:
        study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")
    except KeyError as e:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=study_name,
        )

    study.optimize(objective, n_trials=100)


# %%
