import pandas as pd
from preprocessing import read_consumption_data


def test_feature_expectations_schema(df):
    # Check for non-negative consumption values
    assert df["consumption"].min() >= 0, "Consumption values should be non-negative."

    # Check for valid temperature range (e.g., -50 to 50 degrees Celsius)
    assert df["temperature"].min() >= -50, "Temperature values too low."
    assert df["temperature"].max() <= 50, "Temperature values too high."

    # Check for valid date and time
    assert pd.to_datetime(df["time"]).equals(
        df["time"]
    ), "Time column contains invalid datetime values."

    # Check for valid location values
    # **NB**: This list must be updated if new locations are added to the data
    valid_locations = [
        "oslo",
        "bergen",
        "trondheim",
        "tromsø",
        "stavanger",
        "helsingfors",
    ]
    assert set(df["location"].unique()).issubset(
        valid_locations
    ), "Invalid locations found in data."

    # Check for reasonable consumption range (based on historical data)
    # **NB**: If new cities are added to the data, this number might need to be updated
    max_reasonable_consumption = 100
    assert (
        df["consumption"].max() <= max_reasonable_consumption
    ), "Unreasonably high consumption values found."

    # Check for continuous time series data, no gap should be larger than 1 hour
    df["time"] = pd.to_datetime(df["time"])
    max_gap = pd.Timedelta(hours=1)
    time_diff = df["time"].sort_values().diff().max()
    assert (
        time_diff <= max_gap
    ), f"Found a gap larger than {max_gap} in time series data."

    print("Feature expectations schema test passed.")


if __name__ == "__main__":
    df = read_consumption_data()
    test_feature_expectations_schema(df)
    print("All tests passed.")
