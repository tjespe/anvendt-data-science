# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost
from denormalize import denormalize_predictions
import seaborn as sns

from preprocessing import preprocess_consumption_data, read_consumption_data
from train_test_split import split_into_cv_folds_and_test_fold
from xgb import train_xgb
from xgb import predict_xgb


# %%
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
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_lambda=reg_lambda,
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
# Add baseline prediction (equal to the actual value for the same hour and same location the previous week)
baseline_df = processed_df[
    ["consumption", "location", "hour", "time", "weekday"]
].copy()
baseline_df["baseline"] = (
    baseline_df.sort_values(by="time")
    .groupby(["location", "hour", "weekday"])["consumption"]
    .shift(1)
)
baseline_df = baseline_df.set_index(["time", "location"])
results_df = results_df.join(baseline_df["baseline"])

# %%
# Calculate RMSE for denormalized data
rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["prediction"]))
print(f"RMSE: {rmse}")
rmse_baseline = np.sqrt(
    mean_squared_error(results_df["actual"], results_df["baseline"])
)
print(f"RMSE (baseline): {rmse_baseline}")

# Calculate mean absolute percentage error for denormalized data
results_df["PE"] = 100 * (
    (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
)
results_df["APE"] = np.abs(results_df["PE"])
results_df["PE_baseline"] = 100 * (
    (results_df["actual"] - results_df["baseline"]) / results_df["actual"]
)
results_df["APE_baseline"] = np.abs(results_df["PE_baseline"])

# %%
mape = results_df["APE"].mean()
print(f"MAPE: {mape}%")
print(f"MAPE (baseline): {results_df['APE_baseline'].mean()}%")
print(
    "Location stats\n",
    results_df.groupby(results_df.index.get_level_values("location"), observed=True)[
        ["APE", "PE", "APE_baseline", "PE_baseline"]
    ].mean(),
)
fold_stats = results_df.reset_index().groupby("fold")["time"].agg(["min", "max"])
fold_stats.columns = ["From", "To"]
fold_stats["MAPE"] = results_df.groupby("fold")["APE"].mean()
fold_stats["MPE"] = results_df.groupby("fold")["PE"].mean()
fold_stats["MAPE_baseline"] = results_df.groupby("fold")["APE_baseline"].mean()
fold_stats["MPE_baseline"] = results_df.groupby("fold")["PE_baseline"].mean()
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
# Test on the test fold
test_fold = folds[-1]
X_train, y_train = test_fold[0]
X_test, y_test = test_fold[1]
X_train = X_train[features_to_use]
X_train = pd.get_dummies(X_train)

# Exclude the last 5 days of training data, to account for the 5 day data lag in real
# world predictions
first_date_test = X_test.index.get_level_values("time").min()
last_allowed_date_train = first_date_test - pd.Timedelta(days=5)
X_train = X_train[X_train.index.get_level_values("time") <= last_allowed_date_train]
y_train = y_train[y_train.index.get_level_values("time") <= last_allowed_date_train]

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
X_test = X_test[features_to_use]
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)[X_train.columns]
y_test_pred = predict_xgb(model, X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"RMSE (normalized): {rmse}")
test_results_df = pd.DataFrame(
    {
        "actual": y_test,
        "prediction": y_test_pred,
    },
    index=y_test.index,
)
if use_normalization:
    results_df = denormalize_predictions(
        test_results_df, raw_df, rolling_normalization_window_days
    )
else:
    results_df = test_results_df

# %%
# Test a model trained on raw_df
raw_folds = split_into_cv_folds_and_test_fold(
    raw_df,
    n_splits=num_splits,
    target_variable="consumption",
)
raw_test_fold = raw_folds[-1]
raw_X_train, raw_y_train = raw_test_fold[0]
raw_X_train = pd.get_dummies(raw_X_train)
raw_X_test, raw_y_test = raw_test_fold[1]
raw_X_train = raw_X_train[: -5 * 24]
raw_y_train = raw_y_train[: -5 * 24]
raw_model = train_xgb(
    raw_X_train,
    raw_y_train,
)
raw_X_test = pd.get_dummies(raw_X_test)
raw_y_test_pred = predict_xgb(raw_model, raw_X_test)
rmse = np.sqrt(mean_squared_error(raw_y_test, raw_y_test_pred))
print(f"RMSE (raw): {rmse}")
test_results_df = pd.DataFrame(
    {
        "actual": raw_y_test,
        "raw_prediction": raw_y_test_pred,
    },
    index=raw_y_test.index,
)

# %%
# Add baseline prediction (equal to the actual value for the same hour and same location the previous week)
results_df = results_df.join(baseline_df["baseline"])

# %%
# Add raw prediction
results_df = results_df.join(test_results_df["raw_prediction"])

# %%
# Calculate RMSE for denormalized data
rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["prediction"]))
print(f"RMSE: {rmse}")
rmse_baseline = np.sqrt(
    mean_squared_error(results_df["actual"], results_df["baseline"])
)
print(f"RMSE (baseline): {rmse_baseline}")
rmse_raw = np.sqrt(
    mean_squared_error(results_df["actual"], results_df["raw_prediction"])
)
print(f"RMSE (raw): {rmse_raw}")

# Calculate mean absolute percentage error for denormalized data
results_df["PE"] = 100 * (
    (results_df["actual"] - results_df["prediction"]) / results_df["actual"]
)
results_df["APE"] = np.abs(results_df["PE"])
results_df["PE_baseline"] = 100 * (
    (results_df["actual"] - results_df["baseline"]) / results_df["actual"]
)
results_df["APE_baseline"] = np.abs(results_df["PE_baseline"])
results_df["PE_raw"] = 100 * (
    (results_df["actual"] - results_df["raw_prediction"]) / results_df["actual"]
)
results_df["APE_raw"] = np.abs(results_df["PE_raw"])

# %%
mape = results_df["APE"].mean()
print(f"MAPE: {mape}%")
print(f"MAPE (baseline): {results_df['APE_baseline'].mean()}%")
print(f"MAPE (raw): {results_df['APE_raw'].mean()}%")
print(
    "Location stats\n",
    results_df.groupby(results_df.index.get_level_values("location"), observed=True)[
        ["APE", "PE", "APE_baseline", "PE_baseline", "APE_raw", "PE_raw"]
    ].mean(),
)

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
        plt.savefig(f"analysis/Test data predictions/week_{week}_{location}.png")
        plt.show()

# %%
# Create one line graph for entire period in each location
for location in locations:
    # Filter data for the current week
    subset = results_df[results_df.index.get_level_values("location") == location]

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

    # Customize the plot
    plt.title(
        f"Actual vs. Predicted Consumption in {location[0].upper()+location[1:]} (Test Data)"
    )
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
    plt.savefig(f"analysis/Test data predictions/{location}.png")
    plt.show()

# %%
# Plot feature importance
xgboost.plot_importance(model)

# %%
# Plot correlation between features
all_data = pd.concat([X_train, X_test])
all_target = pd.concat([y_train, y_test])
all_data["consumption"] = all_target
corr = all_data.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.savefig("analysis/correlation.png")
plt.show()

# %%
