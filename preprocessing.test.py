import pandas as pd
import unittest
from preprocessing import (
    add_time_features,
    normalize_consumption,
    add_temperature_features,
    read_consumption_data,
)


class TestPreprocessingFunctions(unittest.TestCase):
    def setUp(self):
        self.df = read_consumption_data()

    def test_add_time_features(self):
        df = add_time_features(self.df.copy())
        assert "Hour" in df.columns
        assert "Season" in df.columns
        assert "Weekday" in df.columns
        assert "Weekend" in df.columns

    def test_normalize_consumption(self):
        df, _ = normalize_consumption(self.df.copy(), None)
        assert "Normalized consumption" in df.columns
        assert df["Normalized consumption"].isna().sum() < df.shape[0] / 10

    def test_add_temperature_features(self):
        df = add_temperature_features(add_time_features(self.df.copy()))
        for feature in [
            "Temperature 1 hour ago",
            "Temperature 2 hours ago",
            "Temperature 3 hours ago",
            "Average temperature 4 to 6 hours ago",
            "Average temperature 7 to 12 hours ago",
            "Average temperature 13 to 24 hours ago",
            "Average temperature last week",
            "Temperature one week ago",
            "Average temperature two weeks ago",
            "Temperature two weeks ago",
            "Average temperature three weeks ago",
            "Temperature three weeks ago",
        ]:
            assert feature in df.columns
            assert df[feature].isna().sum() < df.shape[0] / 10


if __name__ == "__main__":
    unittest.main()
