import pandas as pd
from preprocessing import get_cumulative_stats


def denormalize_predictions(
    predictions: pd.DataFrame,
    raw_df: pd.DataFrame,
    rolling_normalization_window_size=30,
):
    """
    Since we train on and predict consumption normalized per location, we need to
    denormalize the predictions. This function does that.

    Parameters
    ----------
    predictions : pd.DataFrame, a dataframe with the predictions and the actual values.
        Index:
        - time: datetime, the hour of the measurement
        - location: string, one of the 6 cities
        Columns:
        - prediction: float, the predicted consumption normalized by the mean consumption of the location.
        - actual: float, the actual consumption normalized by the mean consumption of the location.
    raw_df: pd.DataFrame, a dataframe with raw data (from read_consumption_data function)
        Columns:
        - time: datetime, the hour of the measurement
        - location: string, one of the 6 cities
        - consumption: float
    rolling_normalization_window_size: int or None
        If set to a number, it is assumed that a rolling window of that size was used
        for normalizing consumption within each location (as opposed to using all data
        up until each data point).
    """
    df = predictions.copy()
    stats_df = raw_df[["time", "location", "consumption"]].copy().sort_values(by="time")
    stats_df[["mean", "std"]] = get_cumulative_stats(
        stats_df, rolling_normalization_window_size
    )
    stats_df = stats_df.set_index(["time", "location"])
    df["prediction"] = df["prediction"] * stats_df["std"] + stats_df["mean"]
    df["actual"] = df["actual"] * stats_df["std"] + stats_df["mean"]
    return df
