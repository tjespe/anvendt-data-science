import pandas as pd


def read_consumption_data():
    """
    Reads the consumption data from the csv file.
    Resulting dataframe has columns:
    - time: datetime, the hour of the measurement
    - location: string, one of the 6 cities
    - consumption: float, avg. MW in the hour
    - temperature: float, avg. temperature in the hour
    """
    # Read the csv file
    df = pd.read_csv("data/consumption.csv")
    # Convert the date column to datetime
    df["time"] = pd.to_datetime(df["time"])
    # Ensure consumption and temperature are floats
    df["consumption"] = df["consumption"].astype(float)
    df["temperature"] = df["temperature"].astype(float)
    # Return the dataframe
    return df


def preprocess_consumption_data(df: pd.DataFrame, rolling_normalization_window_days=30):
    """
    Preprocesses the consumption data.

    Parameters
    ----------
    df: pd.DataFrame
        Consumption data. Columns:
        - time: datetime, the hour of the measurement
        - location: string, one of the 6 cities
        - consumption: float, avg. MW in the hour
        - temperature: float, avg. temperature in the hour
    rolling_normalization_window_days: int or None
        If set to a number, a rolling window will be used for normalizing consumption
        within each location (as opposed to using all data up until each data point).

    Returns
    -------
    df : pd.DataFrame
        Preprocessed consumption data. Columns:
        ### Time-based:
        - time: datetime, the hour of the measurement. **NB** Not intended to be used as a feature, but left in for
            convenience.
        - hour: string, the hour of the measurement ("00" - "23", can be used as categorical)
        - month: string, the month of the measurement ("01" - "12", can be used as categorical)
        - season: string, the season of the measurement ("winter", "spring", "summer", "fall", can be used as categorical)
        - weekend: bool, whether the measurement was on a weekend
        - weekday: string, the weekday of the measurement ("Monday" - "Sunday", can be used as categorical)
        - holiday: bool, whether the measurement was during a holiday NOT IMPLEMENTED
        - days_to_holiday: int, number of days to the next holiday NOT IMPLEMENTED
        ### Location-based:
        - location: string, one of the 6 cities
        ### Consumption-based:
        - consumption_normalized: float, the consumption normalized by the mean consumption of the location.
            **NB**: This is the target variable.
        - mean_consumption_7d: float, the mean (normalized) consumption at that hour last 7 days
            **NB**: Since there is a 5 day data lag, this feature is the mean of the 7 days preceding the
            5 days preceding the measurement, i.e. |---7d used for mean---|---5d gap---|---measurement---|
            For training, we only use training data points when calculating the mean.
            For validation, we only use training and validation data points when calculating the mean.
            For test, we only use training, validation, and test data points when calculating the mean.
        - mean_consumption_14d: float, the mean (normalized) consumption at that hour last 14 days (same system as 7d)
        - consumption_1w_ago: float, the consumption (normalized) at the same hour 1 week ago.
        ### Temperature-based:
        - temperature: float, avg. forecasted temperature in the hour
        - temperature_1h_ago: float, avg. forecasted temperature in the hour 1 hour ago
        - temperature_2h_ago: float, avg. forecasted temperature in the hour 2 hours ago
        - temperature_3h_ago: float, avg. forecasted temperature in the hour 3 hours ago
        - temperature_4_to_6h_ago: float, avg. forecasted temperature in the hours 4-6 hours ago
        - temperature_7_to_12h_ago: float, avg. forecasted temperature in the hours 7-12 hours ago
        - temperature_13_to_24h_ago: float, avg. forecasted temperature in the hours 13-24 hours ago
    """
    # %%

    # Merging holidays dataset into consumption dataset
    holiday_df = read_holiday_data()
    holiday_norway_df = holiday_df[holiday_df["country"] == "Norway"]
    holiday_finland_df = holiday_df[holiday_df["country"] == "Finland"]
    df["holiday_norway"] = df["time"].dt.date.isin(holiday_norway_df["date"]) & df[
        "location"
    ].isin(["oslo", "bergen", "trondheim", "tromsø", "stavanger"])
    df["holiday_finland"] = df["time"].dt.date.isin(holiday_finland_df["date"]) & df[
        "location"
    ].isin(["helsingfors"])
    df["holiday"] = df["holiday_norway"] | df["holiday_finland"]
    df = df.drop(["holiday_norway", "holiday_finland"], axis=1)

    # df["days_to_holiday"] = ...

    # Extract the hour, month, season, weekday, and weekend
    df["hour"] = df["time"].dt.strftime("%H")
    # df["month"] = df["time"].dt.strftime("%m") # MAPE reduced by removeing feature
    df["season"] = df["time"].dt.month.apply(
        lambda x: "winter"
        if x in [12, 1, 2]
        else "spring"
        if x in [3, 4, 5]
        else "summer"
        if x in [6, 7, 8]
        else "fall"
    )
    df["weekday"] = df["time"].dt.day_name()
    df["weekend"] = df["weekday"].isin(["Saturday", "Sunday"])

    # %%
    # Extract the location
    df["location"] = df["location"].astype("category")

    # %%
    # Normalize consumption using cumulative normalization per location
    if rolling_normalization_window_days:
        cumulative_stats = (
            df.groupby("location", observed=True)["consumption"]
            .shift(5 * 24)  # Shift by 5 days to account for lag on receiving data
            .rolling(rolling_normalization_window_days * 24)
            .agg(["mean", "std"])
        )
    else:
        cumulative_stats = (
            df.groupby("location", observed=True)["consumption"]
            .shift(5 * 24)  # Shift by 5 days to account for lag on receiving data
            .expanding()
            .agg(["mean", "std"])
        )
    cumulative_stats["std"].replace(0, 1, inplace=True)
    df["consumption_normalized"] = (
        df["consumption"] - cumulative_stats["mean"]
    ) / cumulative_stats["std"]

    # %%
    # Generate consumption features
    for lookback in [4, 7, 14]:
        df[f"mean_consumption_at_hour_{lookback}d"] = df.groupby(
            ["hour", "location"], observed=True
        )["consumption"].transform(lambda x: x.shift(5).rolling(lookback).mean())
        df[f"mean_consumption_at_hour_{lookback}d_normalized"] = (
            df[f"mean_consumption_at_hour_{lookback}d"] - cumulative_stats["mean"]
        ) / cumulative_stats["std"]
        df[f"mean_consumption_{lookback}d"] = df.groupby("location", observed=True)[
            "consumption"
        ].transform(lambda x: x.shift(5 * 24).rolling(lookback).mean())
        df[f"mean_consumption_{lookback}d_normalized"] = (
            df[f"mean_consumption_{lookback}d"] - cumulative_stats["mean"]
        ) / cumulative_stats["std"]

    df["consumption_1w_ago"] = df.groupby(
        ["location", "weekday", "hour"], observed=True
    )["consumption"].transform(lambda x: x.shift(1))
    df["consumption_1w_ago_normalized"] = (
        df.groupby(["location", "weekday", "hour"], observed=True)[
            "consumption"
        ].transform(lambda x: x.shift(1))
        - cumulative_stats["mean"]
    ) / cumulative_stats["std"]

    # Extract the temperature features
    df["temperature_1h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(1))
    df["temperature_2h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(2))
    df["temperature_3h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(3))
    df["temperature_4_to_6h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(4).rolling(3).mean())
    df["temperature_7_to_12h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(7).rolling(6).mean())
    df["temperature_13_to_24h_ago"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(13).rolling(12).mean())
    df["temperature_prev_week"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(25).rolling(6 * 24).mean())
    df["temperature_1w_ago"] = df.groupby(
        ["location", "hour", "weekday"], observed=True
    )["temperature"].transform(lambda x: x.shift(1))
    df["temperature_prev_prev_week"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(6 * 24 + 25).rolling(7 * 24).mean())
    df["temperature_2w_ago"] = df.groupby(
        ["location", "hour", "weekday"], observed=True
    )["temperature"].transform(lambda x: x.shift(2))
    df["temperature_prev_prev_prev_week"] = df.groupby("location", observed=True)[
        "temperature"
    ].transform(lambda x: x.shift(13 * 24 + 25).rolling(7 * 24).mean())
    df["temperature_3w_ago"] = df.groupby(
        ["location", "hour", "weekday"], observed=True
    )["temperature"].transform(lambda x: x.shift(3))

    # Some rows have NaN because of the lookback features, but since there are so few,
    # we drop them
    drop_mask = df.isna().sum(axis=1) > 0
    print(
        f"Dropping {drop_mask.sum()} of {df.shape[0]} rows ({round(100*drop_mask.sum()/df.shape[0])}%)",
    )
    df.drop(df[drop_mask].index, inplace=True)

    # %%
    return df


def read_holiday_data():
    """
    Reads the holiday data from the csv file.
    Resulting dataframe has columns:
    - date: datetime, the date of the holiday
    - event: string, name of the holiday
    - country: string, country of the holiday
    """
    # Read the csv file
    df = pd.read_csv("data/additional datasets/holidays_2022-2023.csv")
    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Return the dataframe
    return df


df = read_consumption_data()

preprocess_consumption_data(df)
