from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_into_cv_folds_and_test_fold(
    df: pd.DataFrame, n_splits=5, target_variable="consumption_normalized"
):
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
    target_variables = ["consumption_normalized", "consumption"]
    if target_variable not in target_variables:
        raise Exception(
            f"Unexpected target variable {target_variable}, please choose one of {target_variables}"
        )
    for training_indices, test_indices in splits:
        folds.append(
            (
                (
                    df.iloc[training_indices].drop(
                        columns=target_variables, errors="ignore"
                    ),
                    df.iloc[training_indices][target_variable],
                ),
                (
                    df.iloc[test_indices].drop(
                        columns=target_variables, errors="ignore"
                    ),
                    df.iloc[test_indices][target_variable],
                ),
            )
        )
    return folds
