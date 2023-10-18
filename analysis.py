# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost
from denormalize import denormalize_predictions

from preprocessing import preprocess_consumption_data, read_consumption_data
from train_test_split import split_into_cv_folds_and_test_fold
from xgb import train_xgb
from xgb import predict_xgb


# %%
use_normalization = True  # Tested to be good, p-value < 0.01
use_rolling = True  # Tested to be bad, p-value < 0.01
rolling_normalization_window_days = 35  # Tested to be bad, p-value < 0.01
num_splits = 10  # Tested to be inconsequential, set to 10 for practical purposes
n_estimators = 500
max_depth = 5
learning_rate = 0.033
subsample = 0.15
colsample_bytree = 0.50
gamma = 0.07
reg_lambda = 0.3

# %%
features_to_use = [
    "consumption_1w_ago_normalized",
    "temperature",
    "temperature_4_to_6h_ago",
    "temperature_7_to_12h_ago",
    "temperature_1w_ago",
    "temperature_prev_week",
    "temperature_prev_prev_week",
    "mean_consumption_at_hour_4d_normalized",
    "mean_consumption_at_hour_7d_normalized",
]


# %%
raw_df = read_consumption_data()

# %%
# Remove Helsingfors
raw_df = raw_df[raw_df["location"] != "helsingfors"]

# %%
print("Preprocessing data...")
processed_df = preprocess_consumption_data(raw_df, rolling_normalization_window_days)
# %%
print("Splitting")
folds = split_into_cv_folds_and_test_fold(
    processed_df,
    n_splits=num_splits,
    target_variable="consumption_normalized" if use_normalization else "consumption",
)
# %%
# Skip first fold because it has too little training data
folds = folds[1:]
# %%
# Do cross validation
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
    denormalized_results_df["fold"] = i
    results_dfs.append(denormalized_results_df)

# %%
# Merge result dataframes
results_df = pd.concat(results_dfs)

# %%
# Calculate RMSE for denormalized data
rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["prediction"]))
print(f"RMSE: {rmse}")

# Calculate mean absolute percentage error for denormalized data
results_df["PE"] = 100 * (
    (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
)
results_df["APE"] = np.abs(results_df["PE"])

# %%
mape = results_df["APE"].mean()
print(f"MAPE: {mape}%")
print(
    "Location stats\n",
    results_df.groupby(results_df.index.get_level_values("location"), observed=True)[
        ["APE", "PE"]
    ].mean(),
)
fold_stats = results_df.reset_index().groupby("fold")["time"].agg(["min", "max"])
fold_stats.columns = ["From", "To"]
fold_stats["MAPE"] = results_df.groupby("fold")["APE"].mean()
fold_stats["MPE"] = results_df.groupby("fold")["PE"].mean()
fold_stats.From = fold_stats.From.dt.strftime("%d. %b")
fold_stats.To = fold_stats.To.dt.strftime("%d. %b")
print(fold_stats)

# %%
results_df = results_df.reset_index()
results_df["date"] = results_df["time"].dt.date
results_df["week"] = results_df["time"].dt.strftime("%W")
dates = results_df["date"].unique()
weeks = results_df["week"].unique()
locations = results_df["location"].unique()
results_df = results_df.set_index(["time", "location"])

# %%
# Loop through each week and create a separate line graph
for week in weeks:
    for location in locations:
        # Filter data for the current week
        subset = results_df[
            (results_df["week"] == week)
            & (results_df.index.get_level_values("location") == location)
        ]

        # Skip if subset is empty
        if subset.empty:
            continue

        # Create a new figure and plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(
            subset.index.get_level_values("time"),
            subset["actual"],
            label="Actual",
        )
        plt.plot(
            subset.index.get_level_values("time"),
            subset["prediction"],
            label="Predicted",
        )

        # Set format for the x-axis
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %H:%M"))

        # Customize the plot
        plt.title(
            f"Week {week} - Actual vs. Predicted Consumption in {location[0].upper()+location[1:]}"
        )
        plt.xlabel("Hour of Day")
        plt.ylabel("Consumption (avg. MW in hour)")
        plt.legend()

        # Start y-axis at 0
        plt.ylim(bottom=0)

        # Set top of y-axis to 1.5 times the maximum value
        plt.ylim(top=1.5 * max(subset["actual"].max(), subset["prediction"].max()))

        # Display the plot or save it as an image
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# %%
# Plot feature importance
xgboost.plot_importance(model)

# %%
