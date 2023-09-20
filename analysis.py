# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost
from denormalize import denormalize_predictions

from preprocessing import preprocess_consumption_data, read_consumption_data
from train_test_split import split_into_training_validation_and_test
from xgb import preprocess_X_test_for_xgb, preprocess_X_train_for_xgb, train_xgb
from xgb import predict_xgb

df = read_consumption_data()

# %%
# Split, train and predict on dataset
raw_df = read_consumption_data()
print("Preprocessing data...")
processed_df = preprocess_consumption_data(raw_df)
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


# %%
# Line graph of prediction vs. validattion for each week

# Create a DataFrame to hold the week number, actual values, and predicted values
normalized_results = pd.DataFrame(
    {
        "date": X_validation.index.get_level_values("time").strftime("%A %-d. %b %Y"),
        "actual": y_validation,
        "prediction": y_validation_pred,
    },
    index=X_validation.index,
)
results = denormalize_predictions(
    normalized_results,
    raw_df.groupby("location")["consumption"].std(),
    raw_df.groupby("location")["consumption"].mean(),
)

locations = results.index.get_level_values("location").unique()
dates = results["date"].unique()

# %%
# Loop through each week and create a separate line graph
for date_string in dates:
    for location in locations:
        # Filter data for the current week
        subset = results[
            (results["date"] == date_string)
            & (results.index.get_level_values("location") == location)
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

        # Use HH:MM format for the x-axis
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))

        # Customize the plot
        plt.title(
            f"{date_string} - Actual vs. Predicted Consumption in {location[0].upper()+location[1:]}"
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
