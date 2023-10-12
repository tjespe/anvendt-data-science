import pandas as pd


def denormalize_predictions(predictions: pd.DataFrame, raw_df: pd.DataFrame):
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
    """
    df = predictions.copy()
    stats_df = raw_df[["time", "location", "consumption"]].copy().sort_values(by="time")
    stats_df[["mean", "std"]] = (
        stats_df.groupby("location")["consumption"]
        .expanding()
        .agg(["mean", "std"])
        .reset_index()
        .set_index("level_1")
        .sort_index()
    )[["mean", "std"]]
    stats_df = stats_df.set_index(["time", "location"])
    df["prediction"] = df["prediction"] * stats_df["std"] + stats_df["mean"]
    df["actual"] = df["actual"] * stats_df["std"] + stats_df["mean"]
    return df
