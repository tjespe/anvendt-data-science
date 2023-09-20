import pandas as pd


def denormalize_predictions(
    predictions: pd.DataFrame, sd_per_location: dict, mean_per_location: dict
):
    """
    Since we train on and predict consumption normalized per location, we need to
    denormalize the predictions. This function does that.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions. Columns:
        - time: datetime, the hour of the measurement
        - location: string, one of the 6 cities
        - prediction: float, the consumption normalized by the mean consumption of the location.
    """
