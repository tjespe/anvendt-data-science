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

X_validation['time'] = pd.to_datetime(X_validation['time'])
X_validation['week'] = X_validation['time'].dt.strftime('%U')
# Create a DataFrame to hold the week number, actual values, and predicted values
week_data = pd.DataFrame(
    {'Week': X_validation['week'], 'Actual': y_validation, 'Predicted': y_validation_pred})
# Group the data by week and calculate the mean of actual and predicted values for each week
week_data = week_data.groupby('Week').mean().reset_index()


# Loop through each week and create a separate line graph
for week_num in week_data['Week'].unique():
    # Filter data for the current week
    week_subset = week_data[week_data['Week'] == week_num]

    # Create a new figure and plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(week_subset['Week'], week_subset['Actual'], label='Actual')
    plt.plot(week_subset['Week'], week_subset['Predicted'], label='Predicted')

    # Customize the plot
    plt.title(f'Week {week_num} - Actual vs. Predicted Consumption')
    plt.xlabel('Week')
    plt.ylabel('Consumption')
    plt.legend()

    # Display the plot or save it as an image
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

# %%
