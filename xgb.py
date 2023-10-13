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
    raw_df = raw_df[raw_df["location"] != "helsingfors"]

    # %%
    def objective(trial):
        """
        Use optuna to select hyperparameters
        """
        use_normalization = trial.suggest_categorical(
            "use_target_normalization", [True, False]
        )
        use_rolling = False
        if use_normalization:
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
        n_estimators = trial.suggest_int("n_estimators", 1, 2000, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 20, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-9, 1, log=True)
        # subsample = trial.suggest_float("subsample", 0.1, 1)
        # colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1)
        # gamma = trial.suggest_float("gamma", 0, 1)
        # reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        print(json.dumps(trial.params, indent=4))

        # %%
        print("Preprocessing data...")
        processed_df = preprocess_consumption_data(
            raw_df, rolling_normalization_window_days
        )
        # %%
        print("Splitting")
        folds = split_into_cv_folds_and_test_fold(
            processed_df,
            n_splits=num_splits,
            target_variable="consumption_normalized"
            if use_normalization
            else "consumption",
        )
        # %%
        # Select features
        features = list(folds[0][0][0].columns)
        features_to_use = []
        for feature in features:
            if trial.suggest_categorical(f"use_{feature}", [True, False]):
                features_to_use.append(feature)
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
                # subsample=subsample,
                # colsample_bytree=colsample_bytree,
                # gamma=gamma,
                # reg_lambda=reg_lambda,
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

    # %%
    study_name = "XGBoost consumption prediction"
    try:
        study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna.db")
    except KeyError as e:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            storage="sqlite:///optuna.db",
            study_name=study_name,
        )

    def one_fold_validation():
        """
        Get validation scores for the current set of chosen parameters
        (useful for simple testing)
        """

        use_normalization = True
        rolling_normalization_window_days = None
        num_splits = 10
        n_estimators = 50
        max_depth = 8
        learning_rate = 0.03
        processed_df = preprocess_consumption_data(
            raw_df, rolling_normalization_window_days
        )
        folds = split_into_cv_folds_and_test_fold(
            processed_df,
            n_splits=num_splits,
            target_variable="consumption_normalized"
            if use_normalization
            else "consumption",
        )
        training, validation = folds[-2]
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
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)[X_train.columns]
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

        # Calculate RMSE for denormalized data
        rmse = np.sqrt(
            mean_squared_error(
                denormalized_results_df["actual"], denormalized_results_df["prediction"]
            )
        )
        print(f"RMSE: {rmse}")

        # Calculate mean absolute percentage error for denormalized data
        denormalized_results_df["PE"] = 100 * (
            (denormalized_results_df["actual"] - denormalized_results_df["prediction"])
            / denormalized_results_df["actual"]
        )
        denormalized_results_df["APE"] = np.abs(denormalized_results_df["PE"])
        mape = denormalized_results_df["APE"].mean()
        print(f"MAPE: {mape}%")

    # one_fold_validation()
    study.optimize(objective, n_trials=100)


# %%
