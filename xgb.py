# %%
import pandas as pd
import numpy as np
from denormalize import denormalize_predictions
from preprocessing import preprocess_consumption_data, read_consumption_data
from train_test_split import split_into_training_validation_and_test
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


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
    print("Preprocessing data...")
    processed_df = preprocess_consumption_data(raw_df)
    # %%
    print("Splitting")
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _X_test,  # We should not use this data yet
        _y_test,  # We should not use this data yet
    ) = split_into_training_validation_and_test(processed_df)

    X_train, label_encoders = preprocess_X_train_for_xgb(X_train)
    X_validation = preprocess_X_test_for_xgb(X_validation, label_encoders)

    model = train_xgb(X_train, y_train)
    y_validation_pred = predict_xgb(model, X_validation)
    rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
    print(f"RMSE (normalized): {rmse}")

    # %%
    results_df = pd.DataFrame(
        {
            "actual": y_validation,
            "prediction": y_validation_pred,
        },
        index=y_validation.index,
    )
    sd_per_location = raw_df.groupby("location")["consumption"].std()
    mean_per_location = raw_df.groupby("location")["consumption"].mean()
    denormalized_results_df = denormalize_predictions(
        results_df, sd_per_location, mean_per_location
    )

    # %%
    # Calculate RMSE for denormalized data
    rmse = np.sqrt(
        mean_squared_error(
            denormalized_results_df["actual"], denormalized_results_df["prediction"]
        )
    )
    print(f"RMSE: {rmse}")

    # Calculate mean absolute percentage error for denormalized data
    mape = np.mean(
        np.abs(
            (denormalized_results_df["actual"] - denormalized_results_df["prediction"])
            / denormalized_results_df["actual"]
        )
    )
    print(f"MAPE: {mape}")

# %%
