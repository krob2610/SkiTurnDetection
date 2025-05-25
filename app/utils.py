from ast import literal_eval
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd


def calculate_camera_time_offset(PHONE_TIME: list, CAMERA_TIME: list) -> int:
    """Calculate offset for camera time

    Parameters
    ----------
    PHONE_TIME : list
        list with phone time
    PHONE_OFFSET : int
        offset for phone time
    CAMERA_TIME : int
        camera time

    Returns
    -------
    int
        offset for camera time
    """
    phone_delta_time = timedelta(
        hours=PHONE_TIME[0],
        minutes=PHONE_TIME[1],
        seconds=PHONE_TIME[2],
        milliseconds=PHONE_TIME[3] * 100,
    )
    camerta_delta_time = timedelta(
        hours=CAMERA_TIME[0],
        minutes=CAMERA_TIME[1],
        seconds=CAMERA_TIME[2],
    )
    # = phone_delta_time + timedelta(milliseconds=PHONE_OFFSET)
    return (phone_delta_time - camerta_delta_time).total_seconds() * 1000

# TODO: Is load_data_for_device actually used anywhere in the workflow?
# If yes, why does it expect a file/folder (data/{date}/{device}/final_df.csv) that may not exist by default?
def load_data_for_device(device_names: list, current_date: str) -> dict:
    """
    Load data for each device for a given date.

    Parameters
    ----------
    device_names : list
        List of device names for which data is to be loaded.
    current_date : str
        The date for which data is to be loaded, in the format 'YYYY-MM-DD'.

    Returns
    -------
    dict
        A dictionary where the keys are device names and the values are pandas DataFrames containing the data for each device.
    """
    df_dict = {}
    for device_name in device_names:
        res = None
        print(f"device_name: {device_name}")
        try:
            res = pd.read_csv(f"data/{current_date}/{device_name}/final_df.csv")
            for col in res.columns:
                if col != "time":
                    res[col] = res[col].apply(
                        lambda x: literal_eval(x) if isinstance(x, str) else x
                    )
                else:
                    res[col] = pd.to_datetime(res[col])
        except FileNotFoundError:
            print(f"data/{current_date}/{device_name}/")
        except pd.errors.EmptyDataError:
            print(f"data/{current_date}/{device_name}/")

        df_dict[device_name] = res
    return df_dict


# * for making yaw not crossing pi/-pi
def unwrap_column(data_column, period=2 * np.pi):
    """
    Unwraps a single column of phase data to avoid discontinuities (jumps) in the plot.

    :param data_column: A pandas Series of angle data (e.g., roll, pitch, yaw) in radians.
    :param period: The period of cyclic data, default is 2*pi for radian data.
    :return: Unwrapped data as a pandas Series.
    """
    unwrapped_data = [data_column.iloc[0]]  # Start with the first value
    cumulative_shift = 0  # Track cumulative shift applied due to wrap correction

    for i in range(1, len(data_column)):
        delta = data_column.iloc[i] - data_column.iloc[i - 1]

        # Detect if there is a jump forward or backward that crosses the boundary
        if delta > period / 2:  # period / 2 -> mozliwe ze musi byc mniej
            cumulative_shift -= period  # Correcting a forward jump
        elif delta < -period / 2:
            cumulative_shift += period  # Correcting a backward jump

        # Append the corrected data to the unwrapped list
        unwrapped_data.append(data_column.iloc[i] + cumulative_shift)

    return pd.Series(unwrapped_data, index=data_column.index)

# TODO: What is a trend? Why is it not defined?
# TODOI: In what situations is it better to enforce a fixed trend? Would adding a tolerance to avoid overcorrecting small fluctuations be a good idea?
def unwrap_column_v2(data_column, period=2 * np.pi):
    """
    Unwraps a column of cyclic data, enforcing a fixed trend (either "grow" or "descend")
    until it crosses another margin.

    :param data_column: A pandas Series of angle data (e.g., roll, pitch, yaw) in radians.
    :param period: The period of cyclic data, default is 2*pi for radian data.
    :param trend: Either "grow" or "descend" to enforce the desired trend direction.
    :return: A pandas Series of unwrapped data.
    """
    unwrapped_data = [data_column.iloc[0]]  # Start with the first value
    cumulative_shift = 0  # Track cumulative shift due to wrapping correction

    for i in range(1, len(data_column)):
        current_value = data_column.iloc[i]
        previous_value = unwrapped_data[-1]  # Use the last unwrapped value
        delta = current_value - previous_value

        # Detect wrap-around and apply correction
        if delta > period / 2:
            cumulative_shift -= period  # Forward jump
        elif delta < -period / 2:
            cumulative_shift += period  # Backward jump

        # Correct the value using the cumulative shift
        corrected_value = current_value + cumulative_shift

        # Enforce the trend
        if trend == "grow" and corrected_value < previous_value:
            # Force the value to grow by adding the period
            corrected_value += period
        elif trend == "descend" and corrected_value > previous_value:
            # Force the value to descend by subtracting the period
            corrected_value -= period

        # Append the corrected value
        unwrapped_data.append(corrected_value)

    return pd.Series(unwrapped_data, index=data_column.index)


# * for making yaw not crossing pi/-pi
def unwrap_angles(
    df_or_dfs, columns_to_unwrap=["roll", "pitch", "yaw"], period=2 * np.pi
):
    """
    Unwraps specified columns in a DataFrame or a list of DataFrames to avoid discontinuities (jumps) in the plot.

    :param df_or_dfs: pandas DataFrame or list of DataFrames containing time-series data.
    :param columns_to_unwrap: List of column names (e.g., ['roll', 'pitch', 'yaw']) to apply unwrapping.
    :param period: The period of cyclic data, default is 2*pi for radian data.
    :return: DataFrame or list of DataFrames with specified columns unwrapped.
    """

    def unwrap_df(df):
        df_unwrapped = df.copy()
        for column in columns_to_unwrap:
            df_unwrapped[column] = unwrap_column(df[column], period)
        return df_unwrapped

    if isinstance(df_or_dfs, list):
        return [unwrap_df(df) for df in df_or_dfs]
    else:
        return unwrap_df(df_or_dfs)


def calculate_euler_angles_list(l_dfs):
    return [calculate_euler_angles(df) for df in l_dfs]


def calculate_euler_angles(df):
    df_cp = df.copy()
    """
        Function transforms quaternions (qx, qy, qz, qw) into Euler angles (roll, pitch, yaw),
        without limiting them to the range of -pi to pi. Updates values in the dataframe.

        Args:
        - df (pd.DataFrame): DataFrame with columns 'qx', 'qy', 'qz', 'qw', 'roll', 'pitch', 'yaw'

        Returns:
        - pd.DataFrame: DataFrame updated with roll, pitch, yaw values
    """

    # function for converting quaternion to euler angles
    def quaternion_to_euler(w, x, y, z):
        # Yaw (ψ, Z axis)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        # Pitch (θ, Y axis)
        pitch = np.arcsin(2 * (w * y - z * x))

        # Roll (ϕ, X axis)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        return roll, pitch, yaw

    # Calculate Euler angles for each row
    rolls, pitches, yaws = [], [], []
    for index, row in df_cp.iterrows():
        qw = row["qw"]
        qx = row["qx"]
        qy = row["qy"]
        qz = row["qz"]

        # Calculate Euler angles
        roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)

        # Save the results
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    # Update the DataFrame with the new values
    df_cp["roll"] = rolls
    df_cp["pitch"] = pitches
    df_cp["yaw"] = yaws

    return df_cp
