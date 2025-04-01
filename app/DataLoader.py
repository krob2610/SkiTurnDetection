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
        for col in res.columns:
            if col not in [TIME, "Curve", "Behavior", "Status"]:
                res[col] = res[col].apply(
                    lambda x: literal_eval(x) if isinstance(x, str) else x
                )
            elif col == TIME:
                res[col] = pd.to_datetime(res[col])
            elif col == "Curve":
                res[col] = res[col].apply(lambda x: x if x in ["L", "R"] else False)
        res.set_index(TIME, inplace=True)
        # save data for later use
        self.data = res
        return res


class SplittedLoader:
    def __init__(self, dataframe):
        self.data = dataframe
        self.splitted_data = {}
