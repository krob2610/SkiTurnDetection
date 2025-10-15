import concurrent.futures
from ast import literal_eval
from datetime import timedelta

import pandas as pd

from app.device_namespace import FINAL_DF_SENSORS, TIME
from app.labels_namespace import ORIENTATION, OTHER


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        # self.granular_data = {}

    def load_transform_data(self):
        res = pd.read_csv(self.data_path)

        # Define columns that should be kept as strings (not evaluated) for labeled data
        string_columns = ["STYLE", "SKIER_LEVEL", "SLOPE"]

        # Define the GPS column which contains array-like strings but may have missing values
        gps_column = "bearingAccuracy_speedAccuracy_verticalAccuracy_horizontalAccuracy_speed_bearing_altitude_longitude_latitude"

        for col in res.columns:
            if col == TIME:
                res[col] = pd.to_datetime(res[col])
            elif col == "Curve":
                # Handle missing Curve values
                res[col] = res[col].apply(
                    lambda x: x if pd.notna(x) and x in ["L", "R"] else False
                )
            elif col in ["Behavior", "Status"]:
                # Handle missing Behavior and Status values
                res[col] = res[col].fillna("")
            elif col in string_columns:
                # Keep these as strings, no evaluation needed, handle missing values
                res[col] = res[col].fillna("unknown")
            elif col == gps_column:
                # Handle GPS column which may have missing values
                res[col] = res[col].apply(
                    lambda x: (
                        literal_eval(x)
                        if pd.notna(x) and isinstance(x, str) and x.strip() != ""
                        else None
                    )
                )
            elif col not in [TIME, "Curve", "Behavior", "Status"]:
                # For sensor data columns, handle missing values before applying literal_eval
                res[col] = res[col].apply(
                    lambda x: (
                        literal_eval(x)
                        if pd.notna(x) and isinstance(x, str) and x.strip() != ""
                        else ([] if pd.isna(x) else x)
                    )
                )

        res.set_index(TIME, inplace=True)
        # save data for later use
        self.data = res
        return res


class SplittedLoader:
    def __init__(self, dataframe):
        self.data = dataframe
        self.splitted_data = {}
