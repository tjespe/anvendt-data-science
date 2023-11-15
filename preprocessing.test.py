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
        assert "hour" in df.columns
        assert "season" in df.columns
        assert "weekday" in df.columns
        assert "weekend" in df.columns

    def test_normalize_consumption(self):
        df, _ = normalize_consumption(self.df.copy(), None)
        assert "consumption_normalized" in df.columns
        assert df["consumption_normalized"].isna().sum() < df.shape[0] / 10

    def test_add_temperature_features(self):
        df = add_temperature_features(add_time_features(self.df.copy()))
        for feature in [
            "temperature_1h_ago",
            "temperature_2h_ago",
            "temperature_3h_ago",
            "temperature_4_to_6h_ago",
            "temperature_7_to_12h_ago",
            "temperature_13_to_24h_ago",
            "temperature_prev_week",
            "temperature_1w_ago",
            "temperature_prev_prev_week",
            "temperature_2w_ago",
            "temperature_prev_prev_prev_week",
            "temperature_3w_ago",
        ]:
            assert feature in df.columns
            assert df[feature].isna().sum() < df.shape[0] / 10


if __name__ == "__main__":
    unittest.main()
