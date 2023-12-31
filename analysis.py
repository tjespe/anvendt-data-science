# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost
from denormalize import denormalize_predictions
import seaborn as sns
import matplotlib.colors as mcolors

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
    "Normalized consumption 1 week ago",
    "Temperature",
    "Average temperature 4 to 6 hours ago",
    "Average temperature 7 to 12 hours ago",
    "Temperature one week ago",
    "Average temperature last week",
    "Average temperature two weeks ago",
    "Normalized mean consumption at same hour last 4 days",
    "Normalized mean consumption at same hour last 7 days",
]

# %%
# Define color palette for plots
colors = [
    "#31454A",
    "#4F8A86",
    "#D2AE8D",
    "#C67A6C",
    "#3D5A63",
]

# %%
raw_df = read_consumption_data()

# %%
# Remove Helsingfors
raw_df = raw_df[raw_df["Location"] != "helsingfors"]

# %%
print("Preprocessing data...")
processed_df = preprocess_consumption_data(raw_df, rolling_normalization_window_days)
# %%
print("Splitting")
folds = split_into_cv_folds_and_test_fold(
    processed_df,
    n_splits=num_splits,
    target_variable="Normalized consumption" if use_normalization else "Consumption",
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
    ["Consumption", "Location", "Hour", "Time", "Weekday"]
].copy()
baseline_df["baseline"] = (
    baseline_df.sort_values(by="Time")
    .groupby(["Location", "Hour", "Weekday"])["Consumption"]
    .shift(1)
)
baseline_df = baseline_df.set_index(["Time", "Location"])
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
    (results_df["prediction"] - results_df["actual"]) / results_df["actual"]
)
results_df["APE"] = np.abs(results_df["PE"])
results_df["PE_baseline"] = 100 * (
    (results_df["baseline"] - results_df["actual"]) / results_df["actual"]
)
results_df["APE_baseline"] = np.abs(results_df["PE_baseline"])

# %%
mape = results_df["APE"].mean()
print(f"MAPE: {mape}%")
mape_baseline = results_df["APE_baseline"].mean()
print(f"MAPE (baseline): {mape_baseline}%")
print(
    "Location stats\n",
    results_df.groupby(results_df.index.get_level_values("Location"), observed=True)[
        ["APE", "PE", "APE_baseline", "PE_baseline"]
    ].mean(),
)
fold_stats = results_df.reset_index().groupby("fold")["Time"].agg(["min", "max"])
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
results_df["date"] = results_df["Time"].dt.date
results_df["week"] = results_df["Time"].dt.strftime("%W")
dates = results_df["date"].unique()
weeks = results_df["week"].unique()
locations = results_df["Location"].unique()
results_df = results_df.set_index(["Time", "Location"])

# %%
# Loop through each week and create a separate line graph
for week in weeks:
    for location in locations:
        # Filter data for the current week
        subset = results_df[
            (results_df["week"] == week)
            & (results_df.index.get_level_values("Location") == location)
        ]

        # Skip if subset is empty
        if subset.empty:
            continue

        # Create a new figure and plot the data
        plt.figure(figsize=(12, 6))
        # Apply color palette
        plt.gca().set_prop_cycle(color=colors)
        plt.plot(
            subset.index.get_level_values("Time"),
            subset["actual"],
            label="Actual",
        )
        plt.plot(
            subset.index.get_level_values("Time"),
            subset["prediction"],
            label="Predicted",
        )

        # Use whitegrid style
        sns.set_style("whitegrid")

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
# Plot feature importance with correct colors
sns.set_style("whitegrid")
xgboost.plot_importance(model, color=colors[0])

# %%
# Test on the test fold
test_fold = folds[-1]
X_train, y_train = test_fold[0]
X_test, y_test = test_fold[1]
X_train = X_train[features_to_use]
X_train = pd.get_dummies(X_train)

# Exclude the last 5 days of training data, to account for the 5 day data lag in real
# world predictions
first_date_test = X_test.index.get_level_values("Time").min()
last_allowed_date_train = first_date_test - pd.Timedelta(days=5)
X_train = X_train[X_train.index.get_level_values("Time") <= last_allowed_date_train]
y_train = y_train[y_train.index.get_level_values("Time") <= last_allowed_date_train]

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
    target_variable="Consumption",
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
raw_results_df = pd.DataFrame(
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
results_df = results_df.join(raw_results_df["raw_prediction"])

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
    (results_df["prediction"] - results_df["actual"]) / results_df["actual"]
)
results_df["APE"] = np.abs(results_df["PE"])
results_df["PE_baseline"] = 100 * (
    (results_df["baseline"] - results_df["actual"]) / results_df["actual"]
)
results_df["APE_baseline"] = np.abs(results_df["PE_baseline"])
results_df["PE_raw"] = 100 * (
    (results_df["raw_prediction"] - results_df["actual"]) / results_df["actual"]
)
results_df["APE_raw"] = np.abs(results_df["PE_raw"])

# %%
mape = results_df["APE"].mean()
print(f"MAPE: {mape}%")
mape_baseline = results_df["APE_baseline"].mean()
print(f"MAPE (baseline): {mape_baseline}%")
mape_raw = results_df["APE_raw"].mean()
print(f"MAPE (raw): {mape_raw}%")
print(
    "Location stats\n",
    results_df.groupby(results_df.index.get_level_values("Location"), observed=True)[
        ["APE", "PE", "APE_baseline", "PE_baseline", "APE_raw", "PE_raw"]
    ].mean(),
)

# %%
# Adding differeces from actual
results_df["diff_pred"] = results_df["actual"] - results_df["prediction"]
results_df["diff_baseline"] = results_df["actual"] - results_df["baseline"]
results_df["diff_raw"] = results_df["actual"] - results_df["raw_prediction"]


results_df.head()


# %%
# Calculating standard deviations for each of the differences

std_pred = results_df["diff_pred"].std()
std_baseline = results_df["diff_baseline"].std()
std_raw = results_df["diff_raw"].std()

print(f"Standard deviation for prediction: {std_pred}")
print(f"Standard deviation for baseline: {std_baseline}")
print(f"Standard deviation for raw: {std_raw}")

# %%
# calculating the number of instances where the absolute difference is above 0.1 for each of the models
number_of_instances_pred = results_df[results_df["diff_pred"].abs() > 0.1].shape[0]
number_of_instances_baseline = results_df[
    results_df["diff_baseline"].abs() > 0.1
].shape[0]
number_of_instances_raw = results_df[results_df["diff_raw"].abs() > 0.1].shape[0]
# printing the results
print(
    f"Number of instances where the absolute difference is above 0.1 for prediction: {number_of_instances_pred}"
)
print(
    f"Number of instances where the absolute difference is above 0.1 for baseline: {number_of_instances_baseline}"
)
print(
    f"Number of instances where the absolute difference is above 0.1 for raw: {number_of_instances_raw}"
)
# calculate the percentage of instances for each of the models
percentage_pred = number_of_instances_pred / results_df.shape[0]
percentage_baseline = number_of_instances_baseline / results_df.shape[0]
percentage_raw = number_of_instances_raw / results_df.shape[0]
# printing the results
print(
    f"Percentage of instances where the absolute difference is above 0.1 for prediction: {percentage_pred}"
)
print(
    f"Percentage of instances where the absolute difference is above 0.1 for baseline: {percentage_baseline}"
)
print(
    f"Percentage of instances where the absolute difference is above 0.1 for raw: {percentage_raw}"
)

# calculating the number of instances where the absolute difference is above 0.5 for each of the models
number_of_instances_pred = results_df[results_df["diff_pred"].abs() > 0.5].shape[0]
number_of_instances_baseline = results_df[
    results_df["diff_baseline"].abs() > 0.5
].shape[0]
number_of_instances_raw = results_df[results_df["diff_raw"].abs() > 0.5].shape[0]
# printing the results
print(
    f"Number of instances where the absolute difference is above 0.5 for prediction: {number_of_instances_pred}"
)
print(
    f"Number of instances where the absolute difference is above 0.5 for baseline: {number_of_instances_baseline}"
)
print(
    f"Number of instances where the absolute difference is above 0.5 for raw: {number_of_instances_raw}"
)
# calculate the percentage of instances for each of the models
percentage_pred = number_of_instances_pred / results_df.shape[0]
percentage_baseline = number_of_instances_baseline / results_df.shape[0]
percentage_raw = number_of_instances_raw / results_df.shape[0]
# printing the results
print(
    f"Percentage of instances where the absolute difference is above 0.5 for prediction: {percentage_pred}"
)
print(
    f"Percentage of instances where the absolute difference is above 0.5 for baseline: {percentage_baseline}"
)
print(
    f"Percentage of instances where the absolute difference is above 0.5 for raw: {percentage_raw}"
)


# calculating the number of instances where the absolute difference is above 1 for each of the models
number_of_instances_pred = results_df[results_df["diff_pred"].abs() > 1].shape[0]
number_of_instances_baseline = results_df[results_df["diff_baseline"].abs() > 1].shape[
    0
]
number_of_instances_raw = results_df[results_df["diff_raw"].abs() > 1].shape[0]
# printing the results
print(
    f"Number of instances where the absolute difference is above 1 for prediction: {number_of_instances_pred}"
)
print(
    f"Number of instances where the absolute difference is above 1 for baseline: {number_of_instances_baseline}"
)
print(
    f"Number of instances where the absolute difference is above 1 for raw: {number_of_instances_raw}"
)

# calculate the percentage of instances for each of the models
percentage_pred = number_of_instances_pred / results_df.shape[0]
percentage_baseline = number_of_instances_baseline / results_df.shape[0]
percentage_raw = number_of_instances_raw / results_df.shape[0]
# printing the results
print(
    f"Percentage of instances where the absolute difference is above 1 for prediction: {percentage_pred}"
)
print(
    f"Percentage of instances where the absolute difference is above 1 for baseline: {percentage_baseline}"
)
print(
    f"Percentage of instances where the absolute difference is above 1 for raw: {percentage_raw}"
)


# %%
results_df = results_df.reset_index()
results_df["date"] = results_df["Time"].dt.date
results_df["week"] = results_df["Time"].dt.strftime("%W")
dates = results_df["date"].unique()
weeks = results_df["week"].unique()
locations = results_df["Location"].unique()
results_df = results_df.set_index(["Time", "Location"])

# %%
results_df.head()

# %% making the final figure

# calculate the MAPE for predection and baseline for each location individually
mape_pred = results_df.groupby(
    results_df.index.get_level_values("Location"), observed=True
)["APE"].mean()
mape_baseline_per_location = results_df.groupby(
    results_df.index.get_level_values("Location"), observed=True
)["APE_baseline"].mean()

# calculate the MPE for predection and baseline for each location individually
mpe_pred = results_df.groupby(
    results_df.index.get_level_values("Location"), observed=True
)["PE"].mean()
mpe_baseline = results_df.groupby(
    results_df.index.get_level_values("Location"), observed=True
)["PE_baseline"].mean()

# sort the locations by: Oslo, Trondheim, Bergen, Stavanger, Tromsø
mape_pred = mape_pred.reindex(["oslo", "trondheim", "bergen", "stavanger", "tromsø"])
mape_baseline_per_location = mape_baseline_per_location.reindex(
    ["oslo", "trondheim", "bergen", "stavanger", "tromsø"]
)
mpe_pred = mpe_pred.reindex(["oslo", "trondheim", "bergen", "stavanger", "tromsø"])
mpe_baseline = mpe_baseline.reindex(
    ["oslo", "trondheim", "bergen", "stavanger", "tromsø"]
)

# make capital letter for location
mape_pred.index = mape_pred.index.str.capitalize()
mape_baseline_per_location.index = mape_baseline_per_location.index.str.capitalize()
mpe_pred.index = mpe_pred.index.str.capitalize()
mpe_baseline.index = mpe_baseline.index.str.capitalize()

print(mape_pred)

# make a variable called "locations" for the locations which is a list
locations = ["Oslo", "Trondheim", "Bergen", "Stavanger", "Tromsø"]
# make a list called "mape_pred_list" which is the MAPE for prediction for each location
mape_pred_list = mape_pred.tolist()
print(mape_pred_list)
# make a list called "mape_baseline_per_location_list" which is the MAPE for baseline for each location
mape_baseline_per_location_list = mape_baseline_per_location.tolist()

df = pd.DataFrame(
    {
        "Location": locations * 2,
        "Values": mape_pred_list + mape_baseline_per_location_list,
        "": ["MAPE (model)"] * len(locations) + ["MAPE (baseline)"] * len(locations),
    }
)
# Create a side-by-side bar plot
plt.figure(figsize=(12, 7))
# Apply color palette
plt.gca().set_prop_cycle(color=colors)

# Create bar plot and set color to the same as the line plot
sns.barplot(x="Location", y="Values", hue="", data=df, palette=colors, linewidth=2)

# Assuming you have two bars per group ('MAPE (model)' and 'MAPE (baseline)')
n_groups = len(locations)
n_bars_per_group = 2
bar_width = (
    0.4  # This is the default in seaborn, but it might be different in your plot
)
total_width = n_bars_per_group * bar_width

# Calculate the offset
offset = total_width / 2 - bar_width / 2

# Adjusted x-coordinates for the scatter plot
adjusted_x_pred = np.arange(n_groups) - offset  # For MPE (model)
adjusted_x_baseline = np.arange(n_groups) + offset  # For MPE (baseline)

# make sns scatter plot for the MPE for prediction and baseline
sns.scatterplot(
    x=adjusted_x_pred,
    y=mpe_pred,
    color=colors[0],
    s=250,
    marker="o",
    label="MPE (model)",
)
sns.scatterplot(
    x=adjusted_x_baseline,
    y=mpe_baseline,
    color=colors[1],
    s=250,
    marker="o",
    label="MPE (baseline)",
)

# set legend fontsize to 12 and to upper right corner
plt.legend(fontsize=12, loc="upper right")

# plt.legend(fontsize=12, labels=["MAPE", "MAPE (baseline)", "MPE", "MPE (baseline)"])
sns.set_style("whitegrid")
# set title to size 18 and Arial font
plt.title(
    "MAPE and MPE for Forecasting Model and Baseline", size=20, fontname="Arial", y=1.02
)
# set x and y labels to size 14 and Arial font
plt.xlabel("Location", size=18, fontname="Arial")
# give the X label a little space
plt.gca().xaxis.labelpad = 10
plt.ylabel("Percentage Error (%)", size=18, fontname="Arial")

# set the x values to %
plt.gca().set_yticklabels(["{:.0f}%".format(x) for x in plt.gca().get_yticks()])
# set ticks to size 16 and Arial font
plt.xticks(size=16, fontname="Arial")
plt.yticks(size=16, fontname="Arial")
# make the legend a square
# Save figure
plt.tight_layout()
plt.savefig("analysis/results_mape_mpe_per_city.png")
plt.show()

# %% Making one for only mape for the presentation
locations = ["Oslo", "Trondheim", "Bergen", "Stavanger", "Tromsø"]
# make a list called "mape_pred_list" which is the MAPE for prediction for each location
mape_pred_list = mape_pred.tolist()
print(mape_pred_list)
# make a list called "mape_baseline_per_location_list" which is the MAPE for baseline for each location
mape_baseline_per_location_list = mape_baseline_per_location.tolist()

df = pd.DataFrame(
    {
        "Location": locations * 2,
        "Values": mape_pred_list + mape_baseline_per_location_list,
        "": ["MAPE (model)"] * len(locations) + ["MAPE (baseline)"] * len(locations),
    }
)
# Create a side-by-side bar plot
plt.figure(figsize=(9, 7))
# make y-axis go from 0 to 20 with 5 steps between each tick
plt.yticks(np.arange(0, 21, 5))

# Apply color palette
plt.gca().set_prop_cycle(color=colors)

# Create bar plot and set color to the same as the line plot, make bars thin
sns.barplot(x="Location", y="Values", hue="", data=df, palette=colors, width=0.6)
# make bars thinner

# set legend fontsize to 12 and to upper right corner
plt.legend(fontsize=12, loc="upper right")

# plt.legend(fontsize=12, labels=["MAPE", "MAPE (baseline)", "MPE", "MPE (baseline)"])
sns.set_style("whitegrid")
# set title to size 18 and Arial font
plt.title("MAPE Forecasting Model vs. Baseline", size=18, fontname="Arial", y=1.02)
# set x and y labels to size 14 and Arial font
plt.xlabel("Location", size=16, fontname="Arial")
# give the X label a little space
plt.gca().xaxis.labelpad = 10
# set y label to nothing
plt.ylabel("", size=16, fontname="Arial")

# set the x values to %
plt.gca().set_yticklabels(["{:.0f}%".format(x) for x in plt.gca().get_yticks()])
# set ticks to size 16 and Arial font
plt.xticks(size=14, fontname="Arial")
plt.yticks(size=14, fontname="Arial")
# make the legend a square
# Save figure
plt.tight_layout()
plt.savefig("analysis/results_mape_per_city.png")
plt.show()

# %%
# Alt.
# Your existing DataFrame preparation code goes here...

# Create a DataFrame for MPE values
mpe_df = pd.DataFrame(
    {
        "Location": locations * 2,
        "Values": mpe_pred.tolist() + mpe_baseline.tolist(),
        "Type": ["MPE (model)"] * len(locations) + ["MPE (baseline)"] * len(locations),
    }
)

# Make the plot larger to accommodate both sets of bars
plt.figure(figsize=(12, 7))

# Plot MAPE bars
sns.barplot(x="Location", y="Values", hue="", data=df, palette=colors, linewidth=2)


# Plot MPE bars on top of the MAPE bars
# Adjust the width of the MPE bars to be narrower than the MAPE bars
# Darken the colors for MPE bars
def darken_color(color, factor=0.5):
    """
    Darkens a given color by blending it with black.

    Parameters:
    - color: The color to darken. Can be a hex code or a named color.
    - factor: A value between 0 and 1. The higher the value, the darker the color.

    Returns:
    - A darkened color.
    """
    # Convert the color to RGB
    rgb = mcolors.to_rgb(color)
    # Darken the RGB components
    darkened_rgb = [component * (1 - factor) for component in rgb]
    # Convert back to hex and return
    return mcolors.to_hex(darkened_rgb)


darker_colors = [
    darken_color(color, 0.5) for color in colors
]  # Function to darken the colors
sns.barplot(
    x="Location",
    y="Values",
    hue="Type",
    data=mpe_df,
    palette=darker_colors,
    linewidth=2,
    edgecolor="white",
    width=0.5,
    # Add gap between bars
    zorder=3,
)

# Set legend, style, title, labels, and other formatting as per your requirement
# ...

# Show the plot
plt.show()


# %%
# create a CSV file with the results
results_df.to_csv("results.csv")

# %%
# Loop through each week and create a separate line graph
locations = ["oslo", "trondheim", "bergen", "stavanger", "tromsø"]
for week in weeks:
    for location in locations:
        # Filter data for the current week
        subset = results_df[
            (results_df["week"] == week)
            & (results_df.index.get_level_values("Location") == location)
        ]

        # Skip if subset is empty
        if subset.empty:
            continue

        y_max = 1.5 * max(
            subset["actual"].max(), subset["prediction"].max(), subset["baseline"].max()
        )
        for label, key, fname_suffix, color in [
            ("Predicted", "prediction", "", colors[1]),
            ("Baseline", "baseline", "_baseline", colors[2]),
        ]:
            # Create a new figure and plot the data
            for figsize, typ in [
                ((12, 6), ""),
                ((8, 6), "_presentation"),
            ]:
                plt.figure(figsize=figsize)
                # Apply color palette
                plt.gca().set_prop_cycle(color=colors)
                sns.lineplot(
                    x=subset.index.get_level_values("Time"),
                    y=subset["actual"],
                    label="Actual",
                    linewidth=2,
                )
                sns.lineplot(
                    x=subset.index.get_level_values("Time"),
                    y=subset[key],
                    label=label,
                    linewidth=2,
                    color=color,
                )

                # Set format for the x-axis
                plt.gca().xaxis.set_major_formatter(
                    matplotlib.dates.DateFormatter("%a %H:%M")
                )

                # Customize the plot
                plt.title(
                    f"Week {week} - Actual vs. {label} Consumption in {location[0].upper()+location[1:]}",
                    size=18,
                    fontname="Arial",
                    y=1.02,
                )
                plt.xlabel("Hour of Day", size=16, fontname="Arial")
                plt.ylabel("Consumption (avg. MW in hour)", size=16, fontname="Arial")
                plt.legend(fontsize=12, loc="upper left")

                # Use whitegrid style
                sns.set_style("whitegrid")

                # Start y-axis at 0 and the top of y-axis to 1.5 times the maximum value
                plt.ylim(bottom=0, top=y_max)

                # Display the plot or save it as an image
                plt.grid(True)
                plt.xticks(rotation=45, size=16, fontname="Arial")
                # set ticks to size 16 and Arial font
                plt.yticks(size=16, fontname="Arial")
                plt.tight_layout()
                plt.savefig(
                    f"analysis/Test data predictions/week_{week}_{location}{fname_suffix}{typ}.png"
                )
                plt.show()

# %%
# Create one line graph for entire period in each location
for location in locations:
    # Filter data for the current location
    subset = results_df[results_df.index.get_level_values("Location") == location]

    # Skip if subset is empty
    if subset.empty:
        continue

    # Create a new figure and plot the data
    plt.figure(figsize=(12, 6))
    # Apply color palette
    plt.gca().set_prop_cycle(color=colors)
    sns.lineplot(
        x=subset.index.get_level_values("Time"),
        y=subset["actual"],
        label="Actual",
        linewidth=2,
    )
    sns.lineplot(
        x=subset.index.get_level_values("Time"),
        y=subset["prediction"],
        label="Predicted",
        linewidth=2,
    )

    # Customize the plot
    plt.title(
        f"Actual vs. Predicted Consumption in {location[0].upper()+location[1:]} (Test Data)",
        size=18,
        fontname="Arial",
        y=1.02,
    )
    plt.xlabel("Time", size=16, fontname="Arial")
    plt.ylabel("Consumption (avg. MW in hour)", size=16, fontname="Arial")
    plt.legend(fontsize=12, loc="upper left")

    # Use whitegrid style
    sns.set_style("whitegrid")

    # Start y-axis at 0
    plt.ylim(bottom=0)

    # Set top of y-axis to 1.5 times the maximum value
    plt.ylim(top=1.5 * max(subset["actual"].max(), subset["prediction"].max()))

    # Display the plot or save it as an image
    plt.grid(True)
    plt.xticks(rotation=45, size=16, fontname="Arial")
    # set ticks to size 16 and Arial font
    plt.yticks(size=16, fontname="Arial")
    plt.tight_layout()
    plt.savefig(f"analysis/Test data predictions/{location}.png")
    plt.show()

# %%
# Create a 5 x 1 grid graph where each subfigure is a line graph for a location
fig, axes = plt.subplots(5, 1, figsize=(14, 16))

# Use one y axis label for all charts
fig.text(
    -0,
    0.5,
    "Consumption (avg. MW in hour)",
    va="center",
    rotation="vertical",
    size=16,
    fontname="Arial",
)

for i, location in enumerate(sorted(locations)):
    # Filter data for the current location
    subset = results_df[results_df.index.get_level_values("Location") == location]

    # Skip if subset is empty
    if subset.empty:
        continue

    # Create a new figure and plot the data
    sns.lineplot(
        x=subset.index.get_level_values("Time"),
        y=subset["actual"],
        label="Actual",
        linewidth=2,
        ax=axes[i],
        color=colors[2],
    )
    sns.lineplot(
        x=subset.index.get_level_values("Time"),
        y=subset["prediction"],
        label="Predicted",
        linewidth=2,
        ax=axes[i],
        color=colors[0],
    )

    # Customize the plot
    # Don't use a title, but display the name of the city inside the chart
    axes[i].text(
        0.01,
        0.85,
        location[0].upper() + location[1:],
        horizontalalignment="left",
        transform=axes[i].transAxes,
        size=18,
        fontname="Arial",
        weight="bold",
    )
    # Hide x axis lables unless it's the bottom plot
    if i != 4:
        axes[i].set_xticklabels([])
        axes[i].set_xlabel("")
    else:
        axes[i].set_xlabel("Time", size=16, fontname="Arial")

    # Hide y-axis label for all charts
    axes[i].set_ylabel(" ")

    # Modify tick parameters with labelsize and pad
    axes[i].tick_params(axis="y", labelsize=16, pad=10, rotation=30)

    # Use #.# format for y-axis ticks
    axes[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

    # Force all y tick labels to be visible
    axes[i].set_yticks(axes[i].get_yticks())

    if i == 0:
        axes[i].legend(fontsize=12, loc="upper right")
    else:
        axes[i].legend().set_visible(False)

    # Use whitegrid style
    sns.set_style("whitegrid")

    # Start y-axis at 0
    # axes[i].set_ylim(bottom=0)

    # # Set top of y-axis to 1.5 times the maximum value
    # axes[i].set_ylim(top=1.5 * max(subset["actual"].max(), subset["prediction"].max()))

    # Display the plot or save it as an image
    axes[i].grid(True)
    axes[i].set_xticklabels(
        axes[i].get_xticklabels(), rotation=45, size=16, fontname="Arial"
    )
plt.tight_layout()
plt.savefig("analysis/Test data predictions/all_locations.png")

# %%
# Plot feature importance of size 10x10
for figsize, typ in [
    ((12, 7), ""),
    ((8 * 1.3, 6 * 1.3), "_presentation"),
]:
    fig, ax = plt.subplots(figsize=figsize)
    xgboost.plot_importance(model, ax=ax, color=colors[0])
    for text in ax.texts:
        text.set_fontsize(14)  # Adjust the font size as needed
    # Use whitegrid style
    sns.set_style("whitegrid")
    # set title to size 18 and Arial font
    plt.title(
        "Feature Importance",
        size=18,
        fontname="Arial",
        y=1.02,
    )
    # set x and y labels to size 14 and Arial font
    plt.xlabel("F score", size=16, fontname="Arial")
    plt.ylabel("Feature", size=16, fontname="Arial")
    # set ticks to size 16 and Arial font
    plt.xticks(size=16, fontname="Arial")
    plt.yticks(size=16, fontname="Arial")

    # Save figure
    plt.tight_layout()
    plt.savefig(f"analysis/feature_importance{typ}.png")

# Save figure
plt.tight_layout()
plt.savefig("analysis/importance.png")

# %%
# Plot correlation between features
all_data = pd.concat([X_train, X_test])
all_target = pd.concat([y_train, y_test])
all_data["Consumption"] = all_target
corr = all_data.corr()
plt.figure(figsize=(13, 12))
# Create gradient cmap based on colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    [
        colors[1],
        colors[0],
    ],
)
sns.heatmap(corr, annot=True, cmap=cmap, annot_kws={"size": 15, "fontname": "Arial"})
plt.title(
    "Feature Correlation",
    fontname="Arial",
    fontsize=20,
    y=1.02,
)
plt.xlabel("Feature", fontname="Arial", fontsize=16)
plt.ylabel("Feature", fontname="Arial", fontsize=16)
# set ticks to size 16 and Arial font
plt.xticks(size=14, fontname="Arial")
plt.yticks(size=14, fontname="Arial")
# Wrap words in x-axis labels
plt.xticks(
    rotation=45,
    ha="right",
)
plt.tight_layout()
plt.savefig("analysis/correlation.png")
plt.show()

# %% Bar plot standard deviation
# make a sns barplot for the standard deviations for pred and baseline
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(
    x=["Forecasting Model", "Baseline"],
    y=[std_pred, std_baseline],
    palette=colors,
    width=0.6,
)
# set title to size 18 and Arial font
plt.title(
    "Standard Deviation of Error for Forecasting Model vs. Baseline",
    size=18,
    fontname="Arial",
    y=1.02,
)
# set x and y labels to size 14 and Arial font
plt.ylabel("Standard Deviation (σ)", size=16, fontname="Arial")
# Get the values y values for the 2 bars and add text to the top of the bars
y1 = std_pred
y2 = std_baseline
plt.text(x=0, y=y1 + 0.01, s=round(y1, 2), size=16, fontname="Arial", ha="center")
plt.text(x=1, y=y2 + 0.01, s=round(y2, 2), size=16, fontname="Arial", ha="center")
# set ticks to size 16 and Arial font
# give the x label a little space
plt.gca().xaxis.labelpad = 10
plt.xticks(size=16, fontname="Arial")
plt.yticks(size=16, fontname="Arial")
# Save figure
plt.tight_layout()
plt.savefig("analysis/std.png")
plt.show()

# %% Bar plot MAPE
# make a sns barplot for the standard deviations for pred and baseline
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(
    x=["Forecasting Model", "Baseline"],
    y=[mape, mape_baseline],
    palette=colors,
    width=0.6,
)
# set title to size 18 and Arial font
plt.title("MAPE for Forecasting Model vs. Baseline", size=18, fontname="Arial", y=1.02)
# set x and y labels to size 14 and Arial font
plt.ylabel("MAPE (%)", size=16, fontname="Arial")
# Get the values y values for the 2 bars and add text to the top of the bars
y1 = mape
y2 = mape_baseline
plt.text(x=0, y=y1 + 0.01, s=f"{round(y1, 2)}%", size=16, fontname="Arial", ha="center")
plt.text(x=1, y=y2 + 0.01, s=f"{round(y2, 2)}%", size=16, fontname="Arial", ha="center")
# set ticks to size 16 and Arial font
# give the x label a little space
plt.gca().xaxis.labelpad = 10
plt.xticks(size=16, fontname="Arial")
plt.yticks(size=16, fontname="Arial")
# Save figure
plt.tight_layout()
plt.savefig("analysis/mape.png")
plt.show()

# %%
