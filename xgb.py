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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit


def preprocess_X_train_for_xgb(X_train: pd.DataFrame):
    """
    Preprocessing that is specific to XGBoost.
    """
    label_encoders = {}
    for column in [
        "hour",
        # "month",
        # "season",
        "weekday",
        "location",
    ]:
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])
        label_encoders[column] = le
    X_train["weekend"] = X_train["weekend"].astype(int)
    return X_train, label_encoders


def preprocess_X_test_for_xgb(X_test: pd.DataFrame, label_encoders: dict):
    """
    Preprocessing that is specific to XGBoost.
    """
    for column in [
        "hour",
        # "month",
        # "season",
        "weekday",
        "location",
    ]:
        X_test[column] = label_encoders[column].transform(X_test[column])
    X_test["weekend"] = X_test["weekend"].astype(int)
    return X_test


def train_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Trains an XGBoost model.
    """
    model = xgb.XGBRegressor(
        # n_estimators=100, # 100 is default
        max_depth=8,  # 3 is default
        # learning_rate=0.1, # 0.1 is default
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
    print("Preprocessing data...")
    processed_df = preprocess_consumption_data(raw_df)
    # %%
    print("Splitting")
    folds = split_into_cv_folds_and_test_fold(processed_df)
    cv_folds = folds[1:-1]
    results_dfs = []
    for i, (training, validation) in enumerate(cv_folds):
        print(f"CV fold {i}")
        X_train, y_train = training
        X_val, y_val = validation
        X_train, label_encoders = preprocess_X_train_for_xgb(X_train)
        X_val = preprocess_X_test_for_xgb(X_val, label_encoders)
        model = train_xgb(X_train, y_train)
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
        denormalized_results_df = denormalize_predictions(results_df, raw_df)
        results_dfs.append(denormalized_results_df)

    # %%
    # Merge result dataframes
    results_df = pd.concat(results_dfs)

    # %%
    # Calculate RMSE for denormalized data
    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["prediction"]))
    print(f"RMSE: {rmse}")

    # Calculate mean absolute percentage error for denormalized data
    mape = np.mean(
        100
        * np.abs(
            (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
        )
    )
    print(f"MAPE: {mape}%")

# %%
