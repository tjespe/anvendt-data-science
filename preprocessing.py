import pandas as pd


def read_consumption_data():
    """
    Reads the consumption data from the csv file.
    Resulting dataframe has columns:
    - time: datetime
    - location: string
    - consumption: float
    - temperature: float
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
