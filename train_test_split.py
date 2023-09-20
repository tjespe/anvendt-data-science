import pandas as pd


def split_into_training_validation_and_test(df: pd.DataFrame):
    """
    Splits the dataframe into training, validation and test sets.
    Uses every second day as training. Out of the 50% taken out, half is used for validation and half for test.

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
    X_train, y_train, X_validation, y_validation, X_test, y_test: pd.DataFrame
        The training, validation and test sets. y_* is the target variable, X_* are the features.
        All the dataframes have (time, location) as index.
    """
    # Use time and location as index
    df = df.set_index(["time", "location"])
    # Use location as a categorical column as well
    df["location"] = df.index.get_level_values("location")
    # Split into training and test
    df_train = df[df.index.get_level_values("time").day % 2 == 0]
    df_test = df[df.index.get_level_values("time").day % 2 == 1]
    # Split the test into validation and test
    df_validation = df_test[df_test.index.get_level_values("time").day % 4 == 1]
    df_test = df_test[df_test.index.get_level_values("time").day % 4 != 1]
    # Split each dataframe into X and y
    X_train, y_train = (
        df_train.drop(columns=["consumption_normalized"]),
        df_train["consumption_normalized"],
    )
    X_validation, y_validation = (
        df_validation.drop(columns=["consumption_normalized"]),
        df_validation["consumption_normalized"],
    )
    X_test, y_test = (
        df_test.drop(columns=["consumption_normalized"]),
        df_test["consumption_normalized"],
    )
    return X_train, y_train, X_validation, y_validation, X_test, y_test
