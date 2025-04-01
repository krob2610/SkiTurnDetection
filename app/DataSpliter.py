import numpy as np
import pandas as pd

from app.device_namespace import FINAL_DF_SENSORS, TIME
from app.labels_namespace import ORIENTATION, OTHER
from app.video_utils import select_data_range

CURVE = "Curve"
DIRECTION = "Direction"


class DataSplitter:
    def __init__(
        self,
        final_df: pd.DataFrame,
    ):
        self.data = final_df

    def split_data(self, start_time: str, end_time: str, offset: str) -> pd.DataFrame:
        # *First calculate the fixed time -> start_time|end_time - offset
        fixed_start = pd.to_datetime(start_time) - pd.Timedelta(offset)
        fixed_end = pd.to_datetime(end_time) - pd.Timedelta(offset)
        print(f"fixed_start: {fixed_start} | fixed_end: {fixed_end}")
        splitted = select_data_range(
            df=self.data, start=str(fixed_start), end=str(fixed_end)
        )
        return splitted

    def split_label_data(
        self,
        start_time: str,
        end_time: str,
        offset: str,
        curves_L: list,
        curves_R: list,
    ) -> pd.DataFrame:
        # firstly it splits data using the above splitter and keeps offset
        splitted = self.split_data(start_time, end_time, offset)
        # convert list of strings to list of pd.datetime
        curves_L = [
            (pd.to_datetime(label) - pd.Timedelta(offset)).time() for label in curves_L
        ]
        curves_R = [
            (pd.to_datetime(label) - pd.Timedelta(offset)).time() for label in curves_R
        ]
        # apply curves to the splitted data in the correct time
        splitted[CURVE] = [
            "L" if label in curves_L else "R" if label in curves_R else False
            for label in splitted.index.time
        ]

        return splitted


class GranularDataSplitter:
    """Splits data into granular lvl (input must be a final df with brackets)"""

    def __init__(
        self,
        final_df: pd.DataFrame,
    ):
        self.data = final_df
        self.granular_data = {}

    def split_into_granular(self):
        # if self.data is None:
        #     self.load_transform_data()
        df = self.data
        if df is None:
            raise ValueError("Data is None")
        data = df.copy()
        columns = list(data.columns)
        granular_columns = list(
            filter(lambda col: col not in FINAL_DF_SENSORS, columns)
        )
        for col in FINAL_DF_SENSORS:
            if col not in columns:
                continue
            else:
                # keep only the columns that are granular columns and col - rest should be droped in new datadframne
                granular_data = data[granular_columns].copy()
                granular_data[col] = data[col]
                sensor_df = granular_data[col].apply(pd.Series)
                # Check if all values in sensor_df are NaNs
                if sensor_df.isna().all().all():
                    # Create a DataFrame with 3 NaN columns
                    if col == "Orientation":
                        sensor_df = pd.DataFrame(
                            {
                                "col1": [np.nan],
                                "col2": [np.nan],
                                "col3": [np.nan],
                                "col4": [np.nan],
                                "col5": [np.nan],
                                "col6": [np.nan],
                                "col7": [np.nan],
                            }
                        )
                    else:
                        sensor_df = pd.DataFrame(
                            {"col1": [np.nan], "col2": [np.nan], "col3": [np.nan]}
                        )
                # Rename these columns using the names in `renames_orientation`
                sensor_df.columns = ORIENTATION if col == "Orientation" else OTHER

                # Drop the original "Orientation" column from `temp`
                granular_data = granular_data.drop(columns=[col])

                # Join the new orientation columns to `temp`
                granular_data = sensor_df.join(granular_data)
                self.granular_data[col] = granular_data
