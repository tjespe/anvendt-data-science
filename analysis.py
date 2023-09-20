# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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
results = pd.DataFrame(
    {
        "Week": X_validation.index.get_level_values("time").strftime("%U"),
        "Actual": y_validation,
        "Predicted": y_validation_pred,
    },
    index=X_validation.index,
)

locations = results.index.get_level_values("location").unique()

# Loop through each week and create a separate line graph
for week_num in results["Week"].unique():
    for location in locations:
        # Filter data for the current week
        subset = results[
            (results["Week"] == week_num)
            & (results.index.get_level_values("location") == location)
        ]

        # Create a new figure and plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(
            subset.index.get_level_values("time"),
            subset["Actual"],
            label="Actual",
        )
        plt.plot(
            subset.index.get_level_values("time"),
            subset["Predicted"],
            label="Predicted",
        )

        # Customize the plot
        plt.title(f"Week {week_num} - Actual vs. Predicted Consumption")
        plt.xlabel("Week")
        plt.ylabel("Consumption")
        plt.legend()

        # Display the plot or save it as an image
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
plt.show()

# %%
