from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_into_cv_folds_and_test_fold(df: pd.DataFrame, n_splits=5):
    """
    Splits data into folds.

    Parameters
    ----------
    df: pd.DataFrame
        The preprocessed data. Required columns:
        - time: datetime, the hour of the measurement. Will be used for splitting and
                then as index in the resulting dataframes.
        - location: string, one of the 6 cities. Will be used for splitting and then
                as index in the resulting dataframes.

    Returns
    -------
    List of folds where each fold is acontains two tuples:
    - 1st tuple: (X_train, y_train)
    - 2nd tuple: (X_val, y_val) or (X_test, y_test)

    Example usage
    -------
    ```
    folds = split_into_cv_folds_and_test_fold(df)
    cv_folds = folds[:-1]
    for training, validation in cv_folds:
        X_train, y_train = training
        X_val, y_val = validation
        ... your code for training and validation ...
    X_test, y_test = folds[-1]
    ... your code for testing ...
    ```
    """
    # Use time and location as index
    df = df.set_index(["time", "location"])
    # Use location as a categorical column as well
    df["location"] = df.index.get_level_values("location")
    # Split into folds
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = tscv.split(df)
    folds = []
    for training_indices, test_indices in splits:
        folds.append(
            (
                (
                    df.loc[training_indices].drop(columns=["consumption_normalized"]),
                    df.loc[training_indices, "consumption_normalized"],
                ),
                (
                    df.loc[test_indices].drop(columns=["consumption_normalized"]),
                    df.loc[test_indices, "consumption_normalized"],
                ),
            )
        )
    return folds
