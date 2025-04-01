import copy
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .device_namespace import DATA_PATH, FINAL_DF_SENSORS, INPUT_PATH, TIME


class DataTransformer:
    def __init__(self, date: str) -> None:
        self.date = date
        self.transformed_data_paths = {}
        folders = self._select_folders(date=date)
        for folder in tqdm(folders, desc="Loading dataframes"):
            self.transformed_data_paths[folder] = {}
            print(self._generate_path_for_final_df(date=date, device=folder))
            for file in self._select_csv_files(path=Path(INPUT_PATH, date, folder)):
                print(f"file = {file}")
                self.transformed_data_paths[folder][file[:-4]] = self._transform_save(
                    path=Path(INPUT_PATH, date, folder, file)
                )
        # load dataframes to memory
        self._load_DataFrames(self.transformed_data_paths)
        self.final_df_paths = {}
        # for each device generate final dataframe
        for key, value in self.device_dataframes.items():
            final_path_device = self._generate_path_for_final_df(date=date, device=key)
            self.final_df_paths[key] = final_path_device
            self._generate_final_df(value, final_path_device)

    def _select_folders(self, date: str) -> None:
        # open folder under INPUT_PATH/date
        path = Path(INPUT_PATH, date)
        # get all folders in INPUT_PATH/date
        return [file.name for file in path.iterdir() if file.is_dir()]

    def _select_csv_files(self, path: Path) -> list:
        # select csv files in folder
        return [file.name for file in path.iterdir() if file.suffix == ".csv"]

    def _transform_save(self, path: Path) -> Path:
        """transform and save single data file

        Parameters
        ----------
        path : Path
            path to single csv file
        """
        # print(f"Transforming {path}")
        last_parts = path.parts[-3:]
        final_path = Path(DATA_PATH, *last_parts)
        folders = final_path.parts[:-1]
        folders = Path(*folders)
        folders.mkdir(parents=True, exist_ok=True)

        if path.name != "Metadata.csv":
            try:
                df = pd.read_csv(path, sep=",")
                df_time = df.copy()
                df_time["time"] = pd.to_datetime(df["time"], unit="ns")

                # Add one hour to 'time' column
                df_time["time"] = df_time["time"] + pd.Timedelta(hours=1)

                offset_path = Path(path.parent, "offset.txt")

                df_real = self._apply_offset(offset_path, df_time)
                df_real = self._unify_dates(df_real)
                df_real.to_csv(final_path, index=False)
            except pd.errors.EmptyDataError:
                shutil.copy(path, final_path)
        return final_path

    def _apply_offset(self, offset_path: Path, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply an offset to the 'time' column of the DataFrame.

        Parameters:
        offset_path (Path): The path to the file containing the offset value.
        df (pd.DataFrame): The DataFrame to apply the offset to.

        Returns:
        pd.DataFrame: The DataFrame with the offset applied to the 'time' column.
        """
        f = open(offset_path, "r")
        offset = int(f.readline())
        df_copy = df.copy()
        df_copy["time"] = df_copy["time"] + pd.Timedelta(offset, unit="ms")
        return df_copy

    def _unify_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["time"] = pd.to_datetime(
            df_copy["time"], format="%Y-%m-%d %H:%M:%S.%f"
        )  # Convert to datetime
        df_copy["time"] = df_copy["time"].dt.round(
            "100L"
        )  # Round to nearest decisecond
        df_copy.drop_duplicates(subset="time", inplace=True)
        return df_copy

    def _load_DataFrames(self, transformed_data_paths: dict) -> None:
        """Load all dataframes to memory"""
        self.device_dataframes = {}
        for key, value in transformed_data_paths.items():
            self.device_dataframes[key] = {}
            for key2, value2 in value.items():
                if key2 in FINAL_DF_SENSORS:
                    try:
                        df = pd.read_csv(value2, parse_dates=[TIME])
                        self.device_dataframes[key][key2] = df
                    except pd.errors.EmptyDataError:
                        print(f"Empty file: {key2} in {key}")

    def _generate_final_df(self, device_dataframes: dict, final_df_path: Path) -> None:
        missing_sensors = set(FINAL_DF_SENSORS) - set(device_dataframes.keys())

        prepared_data = self._cut_dataframes_inside_device(device_dataframes)
        filled_data = self._fill_missing_times(prepared_data)
        final_df = self.prepare_final_df(filled_data)

        # Add columns for missing sensors and fill them with None
        for sensor in missing_sensors:
            final_df[sensor] = None

        final_df.to_csv(final_df_path, index=True)

    def _cut_dataframes_inside_device(self, device_dataframes: dict) -> dict:
        """Cut dataframes to same length"""
        device_dataframes_copy = copy.deepcopy(device_dataframes)
        min_time = max_time = None

        # Find the global min and max time
        for df in device_dataframes_copy.values():
            current_min = df[TIME].min()
            current_max = df[TIME].max()
            min_time = (
                current_min if min_time is None or current_min > min_time else min_time
            )
            max_time = (
                current_max if max_time is None or current_max < max_time else max_time
            )
        print(f"min_time: {min_time}, max_time: {max_time}")

        # Cut each dataframe to the global min and max time
        for key, df in device_dataframes_copy.items():
            device_dataframes_copy[key] = df[
                (df[TIME] >= min_time) & (df[TIME] <= max_time)
            ]

        return device_dataframes_copy

    def _fill_missing_times(self, device_dataframes: dict) -> dict:
        """Fill missing times in dataframes"""
        device_dataframes_copy = copy.deepcopy(device_dataframes)
        all_times = set()

        # Get all unique times
        for df in device_dataframes_copy.values():
            all_times.update(df[TIME].unique())

        # Fill missing times and interpolate other columns
        for key, df in device_dataframes_copy.items():
            unique_times = set(df[TIME].unique())
            missing_times = all_times - unique_times
            if missing_times:
                missing_df = pd.DataFrame(missing_times, columns=[TIME])
                df = pd.concat([df, missing_df])
                df.sort_values(TIME, inplace=True)
                df.set_index(TIME, inplace=True)
                df = (
                    df.interpolate(method=TIME)
                    .fillna(method="bfill")
                    .fillna(method="ffill")
                )
                device_dataframes_copy[key] = df.reset_index()

        return device_dataframes_copy

    @staticmethod
    def prepare_final_df(device_dataframes: dict) -> pd.DataFrame:
        """Prepare final dataframe that merges all dataframes"""
        dfs = []
        device_dataframes_copy = copy.deepcopy(device_dataframes)
        # For each dataframe, create a single column where the values are lists of the values in the original columns
        for key, df in device_dataframes_copy.items():
            df = df.drop(columns="seconds_elapsed", errors="ignore")
            df.set_index(TIME, inplace=True)
            df[key] = df.values.tolist()
            df = df[[key]]  # Keep only the new column
            dfs.append(df)

        # Merge all dataframes on the index
        final_df = pd.concat(dfs, axis=1)

        return final_df

    def _generate_path_for_final_df(self, date: str, device: str) -> Path:
        """Generate path for final dataframe"""
        return Path(DATA_PATH, date, device, "final_df.csv")
