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
    # Capitalize column headers
    df.columns = [col.capitalize() for col in df.columns]
    # Convert the date column to datetime
    df["Time"] = pd.to_datetime(df["Time"])
    # Ensure consumption and temperature are floats
    df["Consumption"] = df["Consumption"].astype(float)
    df["Temperature"] = df["Temperature"].astype(float)
    # Return the dataframe
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


# %%
def add_time_features(df):
    df["Hour"] = df["Time"].dt.strftime("%H")
    df["Month"] = df["Time"].dt.strftime("%m")
    df["Season"] = df["Time"].dt.month.apply(
        lambda x: "winter"
        if x in [12, 1, 2]
        else "spring"
        if x in [3, 4, 5]
        else "summer"
        if x in [6, 7, 8]
        else "fall"
    )
    df["Weekday"] = df["Time"].dt.day_name()
    df["Weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    return df


def get_cumulative_stats(df, rolling_normalization_window_days):
    # df = df.copy().reset_index().sort_values(by="Time")
    if rolling_normalization_window_days:
        cumulative_stats = (
            df.groupby("Location")["Consumption"]
            .shift(5 * 24)  # Shift by 5 days to account for lag on receiving data
            .rolling(rolling_normalization_window_days * 24)
            .agg(["mean", "std"])
        )
    else:
        cumulative_stats = (
            df.groupby("Location")["Consumption"]
            .shift(5 * 24)  # Shift by 5 days to account for lag on receiving data
            .expanding()
            .agg(["mean", "std"])
        )
    cumulative_stats["std"].replace(0, 1, inplace=True)
    return cumulative_stats


def normalize_consumption(df, rolling_normalization_window_days):
    cumulative_stats = get_cumulative_stats(df, rolling_normalization_window_days)
    df["Normalized consumption"] = (
        df["Consumption"] - cumulative_stats["mean"]
    ) / cumulative_stats["std"]
    return df, cumulative_stats


def add_temperature_features(df):
    df["Temperature 1 hour ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(1))
    df["Temperature 2 hours ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(2))
    df["Temperature 3 hours ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(3))
    df["Average temperature 4 to 6 hours ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(4).rolling(3).mean())
    df["Average temperature 7 to 12 hours ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(7).rolling(6).mean())
    df["Average temperature 13 to 24 hours ago"] = df.groupby(
        "Location", observed=True
    )["Temperature"].transform(lambda x: x.shift(13).rolling(12).mean())
    df["Average temperature last week"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(25).rolling(6 * 24).mean())
    df["Temperature one week ago"] = df.groupby(
        ["Location", "Hour", "Weekday"], observed=True
    )["Temperature"].transform(lambda x: x.shift(1))
    df["Average temperature two weeks ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(6 * 24 + 25).rolling(7 * 24).mean())
    df["Temperature two weeks ago"] = df.groupby(
        ["Location", "Hour", "Weekday"], observed=True
    )["Temperature"].transform(lambda x: x.shift(2))
    df["Average temperature three weeks ago"] = df.groupby("Location", observed=True)[
        "Temperature"
    ].transform(lambda x: x.shift(13 * 24 + 25).rolling(7 * 24).mean())
    df["Temperature three weeks ago"] = df.groupby(
        ["Location", "Hour", "Weekday"], observed=True
    )["Temperature"].transform(lambda x: x.shift(3))
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
        - Time: datetime, the hour of the measurement. **NB** Not intended to be used as a feature, but left in for
            convenience.
        - Hour: string, the hour of the measurement ("00" - "23", can be used as categorical)
        - Month: string, the month of the measurement ("01" - "12", can be used as categorical)
        - Season: string, the season of the measurement ("winter", "spring", "summer", "fall", can be used as categorical)
        - Weekend: bool, whether the measurement was on a weekend
        - Weekday: string, the weekday of the measurement ("Monday" - "Sunday", can be used as categorical)
        - Holiday: bool, whether the measurement was during a holiday
        - Days to holiday: int, number of days to the next holiday NOT IMPLEMENTED
        ### Location-based:
        - Location: string, one of the 6 cities
        ### Consumption-based:
        - Normalized consumption: float, the consumption normalized by the mean consumption of the location.
            **NB**: This is the target variable.
        - Mean consumption last 7 days: float, the mean (normalized) consumption the last 7 days
            **NB**: Since there is a 5 day data lag, this feature is the mean of the 7 days preceding the
            5 days preceding the measurement, i.e. |---7d used for mean---|---5d gap---|---measurement---|
        - Mean consumption last 14 days: float, the mean (normalized) consumption the last 14 days (same system as 7d)
        - Consumption 1 week ago: float, the consumption at the same hour 1 week ago.
        - Mean consumption at same hour last 4 days: float, the mean consumption at that hour last 4 days
        - Mean consumption at same hour last 7 days: float, the mean consumption at that hour last 7 days
        - Mean consumption at same hour last 14 days: float, the mean consumption at that hour last 14 days
        - Normalized consumption 1 week ago: float, the consumption (normalized) at the same hour 1 week ago.
        - Normalized mean consumption at same hour last 4 days: float
        - Normalized mean consumption at same hour last 7 days: float
        - Normalized mean consumption at same hour last 14 days: float
        ### Temperature-based:
        - temperature: float, avg. forecasted temperature in the hour
        - Temperature 1 hour ago: float, avg. forecasted temperature in the hour 1 hour ago
        - Temperature 2 hours ago: float, avg. forecasted temperature in the hour 2 hours ago
        - Temperature 3 hours ago: float, avg. forecasted temperature in the hour 3 hours ago
        - Average temperature 4 to 6 hours ago: float, avg. forecasted temperature in the hours 4-6 hours ago
        - Average temperature 7 to 12 hours ago: float, avg. forecasted temperature in the hours 7-12 hours ago
        - Average temperature 13 to 24 hours ago: float, avg. forecasted temperature in the hours 13-24 hours ago
        - Average temperature last week: float, avg. forecasted temperature in the hour 1 week ago
        - Temperature one week ago: float, avg. forecasted temperature in the hour 1 week ago
        - Average temperature two weeks ago: float, avg. forecasted temperature in the hour 2 weeks ago
        - Temperature two weeks ago: float, avg. forecasted temperature in the hour 2 weeks ago
        - Average temperature three weeks ago: float, avg. forecasted temperature in the hour 3 weeks ago
        - Temperature three weeks ago: float, avg. forecasted temperature in the hour 3 weeks ago
    """
    # %%

    # Merging holidays dataset into consumption dataset
    holiday_df = read_holiday_data()
    holiday_norway_df = holiday_df[holiday_df["country"] == "Norway"]
    holiday_finland_df = holiday_df[holiday_df["country"] == "Finland"]
    df["holiday_norway"] = df["Time"].dt.date.isin(holiday_norway_df["date"]) & df[
        "Location"
    ].isin(["oslo", "bergen", "trondheim", "tromsø", "stavanger"])
    df["holiday_finland"] = df["Time"].dt.date.isin(holiday_finland_df["date"]) & df[
        "Location"
    ].isin(["helsingfors"])
    df["Holiday"] = df["holiday_norway"] | df["holiday_finland"]
    df = df.drop(["holiday_norway", "holiday_finland"], axis=1)

    # df["Days to holiday"] = ...

    # %%
    # Extract the time features
    add_time_features(df)

    # %%
    # Extract the location
    df["Location"] = df["Location"].astype("category")

    # %%
    # Normalize consumption using cumulative normalization per location
    df, cumulative_stats = normalize_consumption(df, rolling_normalization_window_days)

    # %%
    # Generate consumption features
    for lookback in [4, 7, 14]:
        df[f"Mean consumption at same hour last {lookback} days"] = df.groupby(
            ["Hour", "Location"], observed=True
        )["Consumption"].transform(lambda x: x.shift(5).rolling(lookback).mean())
        df[f"Normalized mean consumption at same hour last {lookback} days"] = (
            df[f"Mean consumption at same hour last {lookback} days"]
            - cumulative_stats["mean"]
        ) / cumulative_stats["std"]
        df[f"Mean consumption last {lookback} days"] = df.groupby(
            "Location", observed=True
        )["Consumption"].transform(lambda x: x.shift(5 * 24).rolling(lookback).mean())
        df[f"Normalized mean consumption last {lookback} days"] = (
            df[f"Mean consumption last {lookback} days"] - cumulative_stats["mean"]
        ) / cumulative_stats["std"]

    df["Consumption 1 week ago"] = df.groupby(
        ["Location", "Weekday", "Hour"], observed=True
    )["Consumption"].transform(lambda x: x.shift(1))
    df["Normalized consumption 1 week ago"] = (
        df.groupby(["Location", "Weekday", "Hour"], observed=True)[
            "Consumption"
        ].transform(lambda x: x.shift(1))
        - cumulative_stats["mean"]
    ) / cumulative_stats["std"]

    # Extract the temperature features
    df = add_temperature_features(df)

    # Some rows have NaN because of the lookback features, but since there are so few,
    # we drop them
    drop_mask = df.isna().sum(axis=1) > 0
    print(
        f"Dropping {drop_mask.sum()} of {df.shape[0]} rows ({round(100*drop_mask.sum()/df.shape[0])}%)",
    )
    df.drop(df[drop_mask].index, inplace=True)

    # %%
    return df
