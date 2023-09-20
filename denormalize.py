import pandas as pd


def denormalize_predictions(
    predictions: pd.DataFrame,
    sd_per_location: pd.Series,
    mean_per_location: pd.Series,
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
    sd_per_location : pd.Series, a dataframe with the standard deviation of the consumption per location.
    mean_per_location : pd.Series, a dataframe with the mean of the consumption per location.
    """
    df = predictions.copy()
    df["location_sd"] = sd_per_location[df.index.get_level_values("location")].values
    df["location_mean"] = mean_per_location[
        df.index.get_level_values("location")
    ].values
    df["prediction"] = df["prediction"] * df["location_sd"] + df["location_mean"]
    df["actual"] = df["actual"] * df["location_sd"] + df["location_mean"]
    df.drop(["location_sd", "location_mean"], axis=1, inplace=True)
    return df
