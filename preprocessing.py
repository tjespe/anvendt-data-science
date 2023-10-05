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


def preprocess_consumption_data(df: pd.DataFrame):
    """
    Preprocesses the consumption data.

    Parameters
    ----------
    df : pd.DataFrame
        Consumption data. Columns:
        - time: datetime, the hour of the measurement
        - location: string, one of the 6 cities
        - consumption: float, avg. MW in the hour
        - temperature: float, avg. temperature in the hour

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
        - vacation: bool, whether the measurement was during a vacation NOT IMPLEMENTED
        - days_to_vacation: int, number of days to the next vacation NOT IMPLEMENTED
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
    # Define the type of each row (training, test, validation)
    # Most days are training
    df["type"] = "training"
    # Every 10th day is validation
    df.loc[df["time"].dt.day % 10 == 0, "type"] = "validation"
    # Every 10th day is test, but with an offset of 5 days
    df.loc[df["time"].dt.day % 10 == 5, "type"] = "test"

    # Extract the hour, month, season, weekday, and weekend
    df["hour"] = df["time"].dt.strftime("%H")
    # df["month"] = df["time"].dt.strftime("%m")
    # df["season"] = df["time"].dt.month.apply(
    #     lambda x: "winter"
    #     if x in [12, 1, 2]
    #     else "spring"
    #     if x in [3, 4, 5]
    #     else "summer"
    #     if x in [6, 7, 8]
    #     else "fall"
    # )
    df["weekday"] = df["time"].dt.day_name()
    df["weekend"] = df["weekday"].isin(["Saturday", "Sunday"])

    # Extract the location
    df["location"] = df["location"].astype("category")

    # Extract the consumption features
    df["consumption_normalized"] = df.groupby("location")["consumption"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    for lookback in [4, 7, 14]:
        df.loc[df["type"] == "training", f"mean_consumption_{lookback}d"] = (
            df.loc[df["type"] == "training"].groupby(["hour", "location"])[
                "consumption_normalized"
            ]
            # **NB**: This is a bit simplified. There might be a test of validation day
            # in the 7 days preceding the measurement, and since we have excluded those
            # from the mean, the mean is not exactly the mean of the 7 days preceding
            # the measurement. It might be the previous 8 or 9 days (but only a mean of
            # 7 of the days is used).
            .transform(lambda x: x.shift(5 * 24).rolling(lookback * 24).mean())
        )
        df.loc[df["type"] == "validation", f"mean_consumption_{lookback}d"] = (
            df.loc[(df["type"] == "validation") | (df["type"] == "training")]
            .groupby(["hour", "location"])["consumption_normalized"]
            .transform(lambda x: x.shift(5 * 24).rolling(lookback * 24).mean())
        )
        df.loc[df["type"] == "test", f"mean_consumption_{lookback}d"] = df.groupby(
            ["hour", "location"]
        )["consumption_normalized"].transform(
            lambda x: x.shift(5 * 24).rolling(lookback * 24).mean()
        )

    # TODO: Consider whether this data leak is a problem
    df["consumption_1w_ago"] = df.groupby("location")[
        "consumption_normalized"
    ].transform(lambda x: x.shift(7 * 24))
    for weekday in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        df.loc[
            (df["type"] == "training") & (df["weekday"] == weekday),
            f"consumption_1w_ago",
        ] = (
            df.loc[(df["type"] == "training") & (df["weekday"] == weekday)].groupby(
                "location"
            )["consumption_normalized"]
            # **NB**: This is a bit simplified. If the measurement is on a Monday,
            # and the last Monday was a test or validation day, this code will select
            # the Monday before that.
            .transform(lambda x: x.shift(24))
        )
        df.loc[
            (df["type"] == "validation") & (df["weekday"] == weekday),
            f"consumption_1w_ago",
        ] = (
            df.loc[
                ((df["type"] == "training") | (df["type"] == "validation"))
                & (df["weekday"] == weekday)
            ]
            .groupby("location")["consumption_normalized"]
            .transform(lambda x: x.shift(24))
        )
        df.loc[
            (df["type"] == "test") & (df["weekday"] == weekday),
            f"consumption_1w_ago",
        ] = (
            df.loc[df["weekday"] == weekday]
            .groupby("location")["consumption_normalized"]
            .transform(lambda x: x.shift(24))
        )

    # Remove original consumption column
    df = df.drop(columns=["consumption"])

    # Extract the temperature features
    # TODO: There is perhaps some potential for data leakage here, since the
    # temperature is averaged across hours that might be in the previous day,
    # and the previous day might be a test or validation day. However, since
    # the temperature is not a target variable, this should not be a problem.
    df["temperature_1h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(1)
    )
    df["temperature_2h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(2)
    )
    df["temperature_3h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(3)
    )
    df["temperature_4_to_6h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(4).rolling(3).mean()
    )
    df["temperature_7_to_12h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(7).rolling(6).mean()
    )
    df["temperature_13_to_24h_ago"] = df.groupby("location")["temperature"].transform(
        lambda x: x.shift(13).rolling(12).mean()
    )

    return df
