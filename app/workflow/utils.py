import os
from copy import deepcopy
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.DataLoader import DataLoader
from app.DataSpliter import GranularDataSplitter
from app.transformers.smoother import Smoother
from app.utils import unwrap_angles, unwrap_column
from app.visualization_utils import draw_plotly


def read_files(path: str, snowplow: bool = False):
    files = os.listdir(path)
    dataframes = []
    for file in files:
        if file.endswith(".csv"):
            if "snow_plow" in file and not snowplow:
                print(f"snowflow run skiped {path}/{str(file)}")
            else:
                print(f"{path}/{str(file)}")
                loader = DataLoader(f"{path}/{str(file)}")
                dataframes.append(loader.load_transform_data())
    return dataframes


def read_files_recursive(
    path: str, snowplow: bool = False, return_filenames: bool = False
):
    """
    Recursively reads .csv files from folders and subfolders.

    Args:
        path (str): Main folder path.
        snowplow (bool): Whether to include files with "snow_plow" in the name. Default is False.
        return_filenames (bool): Whether to also return filenames. Default is False.


    Returns:
        list: List of DataFrames loaded from .csv files.
    """
    # Use pathlib to handle paths recursively
    base_path = Path(path)
    csv_files = base_path.rglob("*.csv")  # find all csv files recursively

    dataframes, processed_filenames = [], []
    for file in csv_files:
        file_name = str(file.name)
        file_path = str(file)

        # * no use as files ends with "sp"
        if "snow_plow" in file_name and not snowplow:
            print(f"snowplow run skipped: {file_path}")
        else:
            print(f"Processing: {file_path}")
            loader = DataLoader(file_path)
            dataframes.append(loader.load_transform_data())
            processed_filenames.append(file_path)

    if return_filenames:
        return dataframes, processed_filenames
    else:
        return dataframes

# TODO: How is granularity defined?
# Are data split based on time, fixed index ranges, or other criteria?
def split_data_granular(dataframe_list: list[pd.DataFrame], split_on: str):
    orientation_dfs = []
    for df in dataframe_list:
        Granular_spliter = GranularDataSplitter(df)
        Granular_spliter.split_into_granular()
        orientation_dfs.append(Granular_spliter.granular_data[split_on].copy())
    return orientation_dfs


def smooth_data(
    dataframe_list: list[pd.DataFrame],
    sensors: list[str] = ["roll", "pitch", "yaw"],
    window_size: int = 5,
):
    smoothed_dataframes = []
    for df in dataframe_list:
        smoother = Smoother(df)
        smoothed_dataframes.append(
            smoother.smooth_data_centered_ma(window_size=window_size, columns=sensors)
        )
    return smoothed_dataframes


def merge_close_integers(int_list, count_list, threshold=5):
    """
    * Used after we obtain extremas/counts. Need to be used twice for (min and max)
    t is merging close points
    so for example if there is 5 points :
    3, 6, 15, 30, 32 with counts 1, 10, 2, 30, 2 and te threshold is 5
    it will return the list of points 6, 15, 30 with counts 11, 2, 32
    """
    merged_ints = []
    merged_counts = []

    i = 0
    while i < len(int_list):
        current_int = int_list[i]
        current_count = count_list[i]

        # Merge all consecutive numbers within the threshold
        while i < len(int_list) - 1 and abs(int_list[i] - int_list[i + 1]) <= threshold:
            i += 1
            current_int = int_list[i] if count_list[i] > current_count else current_int
            current_count += count_list[i]

        merged_ints.append(current_int)
        merged_counts.append(current_count)
        i += 1

    return merged_ints, merged_counts


def weighted_percentile(data, weights, percentile):
    """
    calculate weighted percentile
    """
    sorted_indices = np.argsort(data)
    sorted_data = np.array(data)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]

    cumulative_weight = np.cumsum(sorted_weights)
    total_weight = cumulative_weight[-1]

    return np.interp(percentile * total_weight / 100.0, cumulative_weight, sorted_data)


def remove_outliers(
    extremas_pos, counts, df, multiplier=1.5, selected_col="yaw", boundary="both"
):
    """
    Removes outliers based on the specified boundary.

    Parameters:
    - extremas_pos: positions of extrema
    - counts: data weights
    - df: dataframe with data
    - multiplier: multiplier for IQR calculation, default is 1.5
    - selected_col: name of the data column
    - boundary: 'upper' (upper boundary only), 'lower' (lower boundary only), 'both' (both lower and upper boundaries)
    """
    data = df[selected_col][extremas_pos].values
    weights = np.array(counts)

    Q1 = weighted_percentile(data, weights, 25)
    Q3 = weighted_percentile(data, weights, 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    if boundary == "upper":
        mask = data <= upper_bound
    elif boundary == "lower":
        mask = data >= lower_bound
    elif boundary == "both":
        mask = (data >= lower_bound) & (data <= upper_bound)
    else:
        raise ValueError(
            "Invalid value for 'boundary'. Choose 'upper', 'lower', or 'both'."
        )

    first_pos = extremas_pos[0]
    last_pos = extremas_pos[-1]

    if first_pos == 0:
        mask[0] = True
    if last_pos == len(df[selected_col]) - 1:
        mask[-1] = True

    filtered_extremas_pos = np.array(extremas_pos)[mask]
    new_counts = np.array(counts)[mask]

    return filtered_extremas_pos.tolist(), new_counts.tolist()


def find_midpoints(mins_l, maxs_l):
    mins, maxs = deepcopy(mins_l), deepcopy(maxs_l)
    midpoints_mins = []
    midpoints_maxs = []

    def calculate_midpoints(source_list, target_list):
        midpoints = []
        for value in source_list:
            greater_value = next((x for x in target_list if x > value), None)
            if greater_value is not None:
                midpoints.append(int((value + greater_value) / 2))
        return midpoints

    midpoints_mins = calculate_midpoints(mins, maxs)
    midpoints_maxs = calculate_midpoints(maxs, mins)

    return midpoints_mins, midpoints_maxs


def find_midpoints_with_weights(mins_l, mins_counts, maxs_l, max_counts):
    mins, maxs = deepcopy(mins_l), deepcopy(maxs_l)
    midpoints_mins = []
    midpoints_maxs = []
    midpoints_mins_counts = []
    midpoints_maxs_counts = []

    def calculate_midpoints(source_list, source_counts, target_list, target_counts):
        midpoints = []
        midpoints_counts = []
        for i, value in enumerate(source_list):
            for j, target_value in enumerate(target_list):
                if target_value > value:
                    midpoints.append(int((value + target_value) / 2))
                    midpoints_counts.append((source_counts[i] + target_counts[j]) / 2)
                    break
        return midpoints, midpoints_counts

    midpoints_mins, midpoints_mins_counts = calculate_midpoints(
        mins, mins_counts, maxs, max_counts
    )
    midpoints_maxs, midpoints_maxs_counts = calculate_midpoints(
        maxs, max_counts, mins, mins_counts
    )

    return midpoints_mins, midpoints_mins_counts, midpoints_maxs, midpoints_maxs_counts

# TODO: Is the percentile threshold still necessary if 'remove_outliers' has already been applied beforehand?
def filter_extremes(final_positions, counts, threshold_int=75):
    # unique_positions, counts = np.unique(final_positions, return_counts=True)
    final_positions_array = np.array(final_positions)
    counts_array = np.array(counts)
    threshold = np.percentile(counts, threshold_int)
    important_extrema_indices = final_positions_array[
        counts_array >= threshold
    ].tolist()
    important_extrema_counts = counts_array[counts_array >= threshold].tolist()

    return important_extrema_indices, important_extrema_counts

# TODO: What is the difference between 'label_apex' and 'find_midpoints_with_weights'?
# Is 'label_apex' a simplified version of midpoint detection? When should each approach be used?
def label_apex(df: pd.DataFrame) -> pd.DataFrame:
    """Mark apex (most important point in the turn) based on behavior and status column."""
    df_cp = df.copy()

    df_cp["Apex"] = np.nan

    start_mask = df_cp["Status"] == "START"
    stop_mask = df_cp["Status"] == "STOP"

    starts = df_cp[start_mask].copy()
    stops = df_cp[stop_mask].copy()

    starts["next_stop_time"] = stops.index[stops.index.searchsorted(starts.index)]

    valid_pairs = starts[
        starts["Behavior"] == df_cp.loc[starts["next_stop_time"], "Behavior"].values
    ]

    mid_times = (
        valid_pairs.index + (valid_pairs["next_stop_time"] - valid_pairs.index) / 2
    )

    nearest_apex_indices = df_cp.index.searchsorted(mid_times)

    df_cp.loc[df_cp.index[nearest_apex_indices], "Apex"] = valid_pairs[
        "Behavior"
    ].values

    return df_cp


def label_apex_list(dfs: list[pd.DataFrame]):
    dfs_labeled = []
    for df in dfs:
        dfs_labeled.append(label_apex(df))
    return dfs_labeled


def align_min_max_lists(
    new_filtered_mins,
    new_filtered_mins_counts,
    new_filtered_max,
    new_filtered_maxs_counts,
):
    len_mins = len(new_filtered_mins)
    len_maxs = len(new_filtered_max)

    def trim_list_by_counts(values, counts, target_len):
        while len(values) > target_len:
            min_count_idx = counts.index(min(counts))
            values.pop(min_count_idx)
            counts.pop(min_count_idx)
        return values, counts

    if len_mins > len_maxs:
        new_filtered_mins, new_filtered_mins_counts = trim_list_by_counts(
            new_filtered_mins, new_filtered_mins_counts, len_maxs + 1
        )
    elif len_maxs > len_mins:
        new_filtered_max, new_filtered_maxs_counts = trim_list_by_counts(
            new_filtered_max, new_filtered_maxs_counts, len_mins + 1
        )

    return (
        new_filtered_mins,
        new_filtered_mins_counts,
        new_filtered_max,
        new_filtered_maxs_counts,
    )


def append_predictions_df(
    df: pd.DataFrame, positions_L: np.ndarray, positions_R: np.ndarray
):
    df_copy = df.copy()

    df_copy["Predicted_Turn"] = False
    # assign left turns by index
    df_copy.iloc[positions_L, df_copy.columns.get_loc("Predicted_Turn")] = "L"
    # assign right turns by index
    df_copy.iloc[positions_R, df_copy.columns.get_loc("Predicted_Turn")] = "R"
    return df_copy


def append_predictions_df_new(
    df: pd.DataFrame,
    positions_L: np.ndarray,
    positions_R: np.ndarray,
    positions_L_counts: np.ndarray = None,
    positions_R_counts: np.ndarray = None,
):
    df_copy = df.copy()

    df_copy["Predicted_Turn"] = False
    df_copy["Counts"] = 0  # Initialize the Counts column with zeros

    # Assign left turns by index
    df_copy.iloc[positions_L, df_copy.columns.get_loc("Predicted_Turn")] = "left"
    if positions_L_counts is not None:
        df_copy.iloc[positions_L, df_copy.columns.get_loc("Counts")] = (
            positions_L_counts
        )

    # Assign right turns by index
    df_copy.iloc[positions_R, df_copy.columns.get_loc("Predicted_Turn")] = "right"
    if positions_R_counts is not None:
        df_copy.iloc[positions_R, df_copy.columns.get_loc("Counts")] = (
            positions_R_counts
        )

    return df_copy


def find_closest_actuals(df_forecast: pd.DataFrame):
    def find_closest_datetime_index(actual_series, target_value, current_datetime):
        closest_index = None
        min_time_diff = pd.Timedelta.max

        for idx, value in actual_series.iteritems():
            if value == target_value:
                time_diff = abs(current_datetime - idx)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_index = idx

        return closest_index

    df_forecast_closest = df_forecast.copy()

    df_forecast_closest["Closest_Actual"] = df_forecast_closest.apply(
        lambda row: (
            find_closest_datetime_index(
                df_forecast_closest["Apex"], row["Predicted_Turn"], row.name
            )
            if row["Predicted_Turn"] in ["left", "right"]
            else None
        ),
        axis=1,
    )
    return df_forecast_closest


def extract_turns_and_predictions(df: pd.DataFrame):

    df_actual = df[df["Apex"].isin(["left", "right"])]
    df_predicted = df[df["Predicted_Turn"].isin(["left", "right"])]
    df_actual.drop(columns=["Predicted_Turn", "Closest_Actual"], inplace=True)
    df_predicted.drop(columns=["Apex"], inplace=True)
    # move index from actual to 'time of turn column'
    df_actual["Turn_Time"] = df_actual.index
    df_predicted["Turn_Time"] = df_predicted.index
    # merge predicted to actuals on Turn_Time column of actual and closest actual in predicted
    df_merged = pd.merge(
        df_actual,
        df_predicted,
        how="left",
        left_on="Turn_Time",
        right_on="Closest_Actual",
        suffixes=("_Actual", "_Predicted"),
    )
    return df_merged


def calculate_scores(df_forecast: pd.DataFrame, margin_of_error: float = 10):
    """
    Calculate the True Positives, False Positives, False Negatives, and True Negatives from the forecast DataFrame.
    Also calculate the precision, recall, F1 score, and accuracy.

    Parameters:
    df_forecast (pd.DataFrame): DataFrame containing the forecasted and actual turn times
    margin_of_error (float): Margin of error in deciseconds (0.1 second)

    Returns:
    dict: A dictionary containing the True Positives, False Positives, False Negatives, True Negatives, precision, recall, F1 score, and accuracy
    """
    df_TP, df_FP, df_FN, tn_count, percentage_score = find_tp_fp_fn_tn(
        df_forecast, margin_of_error
    )
    metrics = calculate_performance_metrics(df_TP, df_FP, df_FN, tn_count)
    metrics["Percentage Score"] = percentage_score
    return metrics


def find_tp_fp_fn_tn(df_forecast: pd.DataFrame, margin_of_error: float = 10):
    """
    Input should have columns: 'Turn_Time_Predicted', 'Turn_Time_Actual'
    margin_of_error - in deciseconds (0.1 second)
    """
    extracted_turns = extract_turns_and_predictions(df_forecast)
    df_TP = pd.DataFrame()  # True Positives
    df_FP = pd.DataFrame()  # False Positives
    df_FN = pd.DataFrame()  # False Negatives
    df_metrics = pd.DataFrame()  # New metrics for distance-based evaluation

    matched_actuals = set()  # Track matched actual events to avoid double-counting
    matched_predictions = set()  # Track matched predictions to avoid double-counting

    for index, row in extracted_turns.iterrows():
        if (
            pd.notnull(row["Turn_Time_Predicted"])
            and row["Turn_Time_Predicted"] not in matched_predictions
        ):
            time_diff = abs(row["Turn_Time_Predicted"] - row["Turn_Time_Actual"])

            # Convert time difference to deciseconds
            time_diff_ds = time_diff.total_seconds() * 10

            # If within margin of error and neither the actual nor predicted has been matched
            if (
                time_diff_ds <= margin_of_error
                and row["Turn_Time_Actual"] not in matched_actuals
            ):
                df_TP = df_TP.append(row)
                matched_actuals.add(row["Turn_Time_Actual"])  # Mark actual as matched
                matched_predictions.add(
                    row["Turn_Time_Predicted"]
                )  # Mark prediction as matched

                # Calculate score from 0 to 1 based on how close the prediction was
                score = 1 - (time_diff_ds / margin_of_error)
                df_metrics = df_metrics.append(
                    {"Index": index, "Score": score, "Type": "TP"}, ignore_index=True
                )

            else:
                # If not matched, it's a False Positive
                df_FP = df_FP.append(row)
                matched_predictions.add(
                    row["Turn_Time_Predicted"]
                )  # Ensure we don't reuse this prediction
                df_metrics = df_metrics.append(
                    {"Index": index, "Score": -1, "Type": "FP"}, ignore_index=True
                )

    # Now calculate False Negatives (Actual events with no predicted match)
    for index, row in extracted_turns.iterrows():
        if (
            pd.isnull(row["Turn_Time_Predicted"])
            or row["Turn_Time_Actual"] not in matched_actuals
        ):
            df_FN = df_FN.append(row)
            matched_actuals.add(row["Turn_Time_Actual"])  # Mark actual as matched
            df_metrics = df_metrics.append(
                {"Index": index, "Score": -1, "Type": "FN"}, ignore_index=True
            )

    # Calculate True Negatives (Total non-event points - FP)
    total_points = len(df_forecast)
    actual_events = len(df_forecast[df_forecast["Apex"].isin(["left", "right"])])
    non_event_points = total_points - actual_events
    tn_count = non_event_points - len(df_FP)

    total_actual_events = (
        actual_events + len(df_FP) + len(df_FN)
    )  # Total points that matter (all events)

    actual_score = df_metrics["Score"].sum()  # Sum of all scores from TP, FP, FN, TN
    percentage_score = (
        (actual_score / total_actual_events) * 100 if total_actual_events > 0 else 0
    )

    return (
        df_TP.reset_index(drop=True),
        df_FP.reset_index(drop=True),
        df_FN.reset_index(drop=True),
        tn_count,
        percentage_score,  # Returning new metric dataframe
    )


def calculate_score_between(df_forecast: pd.DataFrame):
    """
    Input should have columns: 'Turn_Time_Predicted', 'Turn_Time_Actual'
    margin_of_error - in deciseconds (0.1 second)
    """
    predictions = df_forecast[df_forecast["Predicted_Turn"].isin(["left", "right"])]
    actuals = df_forecast[df_forecast["Behavior"].isin(["left", "right"])]

    # between turn without error range
    int_TP_no_error = 0  # True Positives
    int_FP_no_error = 0  # False Positives
    int_FN_no_error = 0  # False Negatives
    matched = set()  # Track matched actual events to avoid double-counting

    for index, row in predictions.iterrows():
        # find closest row before that has status 'START'
        closest_start = df_forecast.loc[: row.name][
            df_forecast.loc[: row.name]["Status"] == "START"
        ].iloc[-1]
        # find closest row after that has status 'STOP'
        closest_stop = df_forecast.loc[row.name :][
            df_forecast.loc[row.name :]["Status"] == "STOP"
        ].iloc[0]
        if (
            closest_start["Behavior"]
            == closest_stop["Behavior"]
            == row["Predicted_Turn"]
            and closest_start.name not in matched
            and closest_stop.name not in matched
        ):
            int_TP_no_error += 1
            matched.add(closest_start.name)
            matched.add(closest_stop.name)
        else:
            int_FP_no_error += 1

    # Now calculate False Negatives (Actual events with no predicted match)
    for index, row in actuals.iterrows():
        if row.name not in matched:
            int_FN_no_error += 1
            matched.add(row.name)  # Mark actual as matched

    int_FN_no_error = int_FN_no_error / 2  # divided by 2 because we have pairs
    # Calculate True Negatives (Total non-event points - FP)
    total_points = len(df_forecast)
    actual_events = len(df_forecast[df_forecast["Apex"].isin(["left", "right"])])
    non_event_points = total_points - actual_events
    tn_count = non_event_points - int_FP_no_error

    return (
        int_TP_no_error,
        int_FP_no_error,
        int_FN_no_error,
        tn_count,
    )


def calculate_distance_between(df_forecast: pd.DataFrame):
    """
    Input should have columns: 'Turn_Time_Predicted', 'Turn_Time_Actual'
    margin_of_error - in deciseconds (0.1 second)
    """
    predictions = df_forecast[df_forecast["Predicted_Turn"].isin(["left", "right"])]
    actuals = df_forecast[df_forecast["Behavior"].isin(["left", "right"])]

    # between turn without error range
    int_TP_no_error = 0  # True Positives
    int_FP_no_error = 0  # False Positives
    int_FN_no_error = 0  # False Negatives
    matched = set()  # Track matched actual events to avoid double-counting

    for index, row in predictions.iterrows():
        # find closest row before that has status 'START'
        closest_start = df_forecast.loc[: row.name][
            df_forecast.loc[: row.name]["Status"] == "START"
        ].iloc[-1]
        # find closest row after that has status 'STOP'
        closest_stop = df_forecast.loc[row.name :][
            df_forecast.loc[row.name :]["Status"] == "STOP"
        ].iloc[0]
        if (
            closest_start["Behavior"]
            == closest_stop["Behavior"]
            == row["Predicted_Turn"]
            and closest_start.name not in matched
            and closest_stop.name not in matched
        ):
            int_TP_no_error += 1
            matched.add(closest_start.name)
            matched.add(closest_stop.name)
        else:
            int_FP_no_error += 1

    # Now calculate False Negatives (Actual events with no predicted match)
    for index, row in actuals.iterrows():
        if row.name not in matched:
            int_FN_no_error += 1
            matched.add(row.name)  # Mark actual as matched

    int_FN_no_error = int_FN_no_error / 2
    # Calculate True Negatives (Total non-event points - FP)
    total_points = len(df_forecast)
    actual_events = len(df_forecast[df_forecast["Apex"].isin(["left", "right"])])
    non_event_points = total_points - actual_events
    tn_count = non_event_points - int_FP_no_error

    return (
        int_TP_no_error,
        int_FP_no_error,
        int_FN_no_error,
        tn_count,
    )


def calculate_performance_metrics(
    df_TP: Union[pd.DataFrame, int],
    df_FP: Union[pd.DataFrame, int],
    df_FN: Union[pd.DataFrame, int],
    tn_count: int,
):
    """
    Calculate performance metrics: precision, recall, F1 score, and accuracy from the DataFrames of TP, FP, FN, and TN count.
    Also calculate metrics for specific types of turns (e.g., Left 'L', Right 'R') if available.

    Parameters:
    df_TP (pd.DataFrame): DataFrame containing True Positives
    df_FP (pd.DataFrame): DataFrame containing False Positives
    df_FN (pd.DataFrame): DataFrame containing False Negatives
    tn_count (int): Number of True Negatives

    Returns:
    dict: A dictionary containing precision, recall, F1 score, and accuracy
    """
    # Count True Positives, False Positives, False Negatives
    tp = len(df_TP) if isinstance(df_TP, pd.DataFrame) else df_TP
    fp = len(df_FP) if isinstance(df_FP, pd.DataFrame) else df_FP
    fn = len(df_FN) if isinstance(df_FN, pd.DataFrame) else df_FN

    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (
        (tp + tn_count) / (tp + tn_count + fp + fn)
        if (tp + tn_count + fp + fn) > 0
        else 0
    )

    # Metrics for Left and Right turns
    metrics = {
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1 Score": f1_score,
        "Overall Accuracy": accuracy,
    }
    if any(isinstance(df, int) for df in [df_TP, df_FP, df_FN]):
        return metrics
    # Check if 'Curve' column exists
    if (
        "Curve" in df_TP.columns
        and "Curve" in df_FP.columns
        and "Curve" in df_FN.columns
    ):
        # Separate metrics for Left and Right turns
        left_tp = df_TP[df_TP["Curve"] == "L"]
        right_tp = df_TP[df_TP["Curve"] == "R"]

        left_fp = df_FP[df_FP["Curve"] == "L"]
        right_fp = df_FP[df_FP["Curve"] == "R"]

        left_fn = df_FN[df_FN["Curve"] == "L"]
        right_fn = df_FN[df_FN["Curve"] == "R"]

        # Calculate metrics for Left turns
        left_precision = (
            len(left_tp) / (len(left_tp) + len(left_fp))
            if (len(left_tp) + len(left_fp)) > 0
            else 0
        )
        left_recall = (
            len(left_tp) / (len(left_tp) + len(left_fn))
            if (len(left_tp) + len(left_fn)) > 0
            else 0
        )
        left_f1_score = (
            2 * (left_precision * left_recall) / (left_precision + left_recall)
            if (left_precision + left_recall) > 0
            else 0
        )
        left_accuracy = (
            (len(left_tp) + tn_count)
            / (len(left_tp) + tn_count + len(left_fp) + len(left_fn))
            if (len(left_tp) + tn_count + len(left_fp) + len(left_fn)) > 0
            else 0
        )

        # Calculate metrics for Right turns
        right_precision = (
            len(right_tp) / (len(right_tp) + len(right_fp))
            if (len(right_tp) + len(right_fp)) > 0
            else 0
        )
        right_recall = (
            len(right_tp) / (len(right_tp) + len(right_fn))
            if (len(right_tp) + len(right_fn)) > 0
            else 0
        )
        right_f1_score = (
            2 * (right_precision * right_recall) / (right_precision + right_recall)
            if (right_precision + right_recall) > 0
            else 0
        )
        right_accuracy = (
            (len(right_tp) + tn_count)
            / (len(right_tp) + tn_count + len(right_fp) + len(right_fn))
            if (len(right_tp) + tn_count + len(right_fp) + len(right_fn)) > 0
            else 0
        )

        metrics.update(
            {
                "Left Precision": left_precision,
                "Left Recall": left_recall,
                "Left F1 Score": left_f1_score,
                "Left Accuracy": left_accuracy,
                "Right Precision": right_precision,
                "Right Recall": right_recall,
                "Right F1 Score": right_f1_score,
                "Right Accuracy": right_accuracy,
            }
        )

    return metrics


def round_df(df):
    rounded_times = []

    def round_and_adjust(value):
        rounded = round(value, 1)

        while rounded in rounded_times:
            rounded += 0.1

        rounded_times.append(rounded)
        return rounded

    df["rounded_time"] = df["Time"].apply(round_and_adjust)


# !method distance from middle
def process_turns_with_restricted_distances(df):
    df_copy = df.copy()
    df_copy.drop(
        columns=[
            "qz",
            "qy",
            "qx",
            "qw",
            "roll",
            "pitch",
            "yaw",
            "Apex",
            "Closest_Actual",
        ],
        inplace=True,
    )
    """
    Process a DataFrame to extract turn details, corresponding predictions, and compute distances.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['Behavior', 'Status', 'Predicted_Turn'].
                           Index should be a timestamp.

    Returns:
        pd.DataFrame: A new DataFrame with columns:
                      ['turn', 'start', 'stop', 'predicted turn type', 'predicted time',
                       'max_valid_distances', 'distance start/stop'].
    """
    # Initialize an empty list to store rows for the new DataFrame
    rows = []

    # Keep track of the current turn's start time, actual turn type, and predicted turn
    current_turn = None
    start_time = None

    # Iterate through the rows of the DataFrame
    for i, (timestamp, row) in enumerate(df_copy.iterrows()):
        if row["Status"] == "START":  # Start of a turn
            # Store the start time and actual turn type
            start_time = timestamp
            current_turn = row["Behavior"]
        elif row["Status"] == "STOP":  # End of a turn
            # Store the stop time
            stop_time = timestamp

            # Calculate the total duration of the turn
            total_duration = (stop_time - start_time).total_seconds()
            half_duration = total_duration / 2  # Calculate the maximum allowed distance

            # Find predictions within this turn
            predictions = df_copy.loc[start_time:stop_time]
            valid_predictions = predictions[
                predictions["Predicted_Turn"].notna()
                & (predictions["Predicted_Turn"] != False)
            ]
            predicted_turns = valid_predictions[
                "Predicted_Turn"
            ].tolist()  # List of predicted turns
            predicted_times = (
                valid_predictions.index.tolist()
            )  # Corresponding timestamps
            counts = valid_predictions["Counts"].tolist()  # Corresponding counts

            # Compute distances for each prediction
            max_valid_distances = []
            if predicted_times:  # If there are valid predictions
                for pred_time, pred_turn in zip(predicted_times, predicted_turns):
                    dist_to_start = abs(pred_time - start_time).total_seconds()
                    dist_to_stop = abs(pred_time - stop_time).total_seconds()

                    # If the predicted turn does not match the actual turn, set distance to 0
                    if pred_turn != current_turn:
                        max_valid_distances.append(0)
                    else:
                        # Select the greater distance, or switch to the smaller one if greater exceeds half duration
                        max_dist = max(dist_to_start, dist_to_stop)
                        if max_dist > half_duration:
                            max_dist = min(
                                dist_to_start, dist_to_stop
                            )  # Use the smaller valid distance

                        max_valid_distances.append(max_dist)

            # Add a new row to the output DataFrame
            rows.append(
                {
                    "turn": current_turn,
                    "start": start_time,
                    "stop": stop_time,
                    "predicted turn type": (
                        predicted_turns if predicted_turns else ["No prediction"]
                    ),
                    "predicted time": (
                        predicted_times if predicted_times else ["No prediction"]
                    ),
                    "max_valid_distances": (
                        max_valid_distances
                        if max_valid_distances
                        else ["No prediction"]
                    ),
                    "half distance start/stop": half_duration,
                    "Counts": counts if counts else ["No prediction"],
                }
            )

            # Reset for the next turn
            current_turn = None
            start_time = None

    return pd.DataFrame(rows)

# TODO: What exactly does 'calculate_scores_sec_metric' compute?
# How is a "correct prediction" defined in this context?
def calculate_scores_sec_metric(df):
    df_cp = df.copy()

    def compute_row_score(row):
        # Ensure that 'max_valid_distances' and 'distance start/stop' are numeric
        if isinstance(row["max_valid_distances"], list) and isinstance(
            row["Counts"], list
        ):
            try:
                # Calculate scores for all valid distances
                scores = [
                    float(dist) / float(row["half distance start/stop"])
                    for dist in row["max_valid_distances"]
                ]

                # If there are multiple scores, select the one based on the highest count
                if scores and row["Counts"]:
                    max_count_index = row["Counts"].index(max(row["Counts"]))
                    selected_score = scores[max_count_index]
                    return [
                        selected_score if i == max_count_index else 0
                        for i in range(len(scores))
                    ]
                else:
                    return [0]  # If no valid scores or counts, return [0]
            except (ValueError, TypeError):
                return [0]  # Return [0] if any invalid value is encountered
        return [0]

    df_cp["scores"] = df_cp.apply(compute_row_score, axis=1)

    # Apply the computation for each row
    return df_cp


def calculate_overall_score(df):
    """
    Calculate the overall score by summing all scores (including zeros) in the 'scores' column
    and dividing by the total number of scores.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'scores' column that contains lists of scores.

    Returns:
        float: The overall average score, including zeros.
    """
    # Flatten all the lists of scores into a single list
    all_scores = [score for sublist in df["scores"] for score in sublist]

    return sum(all_scores), len(all_scores)


def find_max_el_between_current_next(current, next_el, searched_list, counts_list):
    # Find the element with the highest counts between current and next element
    max_count = -1
    max_el = None
    for i in range(len(searched_list)):
        if (current < searched_list[i] < next_el) and (i < len(counts_list)):
            if counts_list[i] > max_count:
                max_count = counts_list[i]
                max_el = searched_list[i]
    return max_el


def get_next_el(current, searched_list, add_index=1):
    if add_index > 20:
        return None
    if searched_list.index(current) + add_index < len(searched_list):
        return searched_list[searched_list.index(current) + add_index]
    else:
        return get_next_el(current, searched_list, add_index=add_index + 1)

# TODO: How are points to be removed selected?
# How should the count threshold be chosen in practice?
def align_min_max_lists_counts_order(
    new_filtered_mins, new_filtered_max, min_counts, max_counts
):
    mins_cp = new_filtered_mins.copy()
    maxs_cp = new_filtered_max.copy()
    first = min(mins_cp[0], maxs_cp[0])
    last = max(mins_cp[-1], maxs_cp[-1])

    new_mins = []
    new_maxs = []

    if first in mins_cp:
        new_mins.append(first)
        left_search = True
    else:
        new_maxs.append(first)
        left_search = False

    current = first
    while mins_cp or maxs_cp:
        if left_search:
            next_el = get_next_el(current, mins_cp)
            if next_el is None:
                break
            current_t = find_max_el_between_current_next(
                current, next_el, maxs_cp, max_counts
            )
            while current_t is None:
                next_el = get_next_el(next_el, mins_cp)
                if next_el is None:
                    break
                current_t = find_max_el_between_current_next(
                    current, next_el, maxs_cp, min_counts
                )

            current = current_t
            if current != -1 and current is not None:
                new_maxs.append(current)
            if next_el is None:
                break
            maxs_cp = [x for x in maxs_cp if x <= current or x >= next_el]
            left_search = False
        else:
            next_el = get_next_el(current, maxs_cp)
            if next_el is None:
                break

            current_t = find_max_el_between_current_next(
                current, next_el, mins_cp, min_counts
            )

            while current_t is None:
                next_el = get_next_el(next_el, maxs_cp)
                if next_el is None:
                    break
                current_t = find_max_el_between_current_next(
                    current, next_el, mins_cp, min_counts
                )

            current = current_t

            if current != -1 and current is not None:
                new_mins.append(current)
            if next_el is None:
                break
            mins_cp = [x for x in mins_cp if x <= current or x >= next_el]
            left_search = True

    if last in mins_cp and last not in new_mins:
        new_mins.append(last)
    elif last in maxs_cp and last not in new_maxs:
        new_maxs.append(last)

    return new_mins, new_maxs


def add_predicted_transitions_to_df(df, predicted_mins, predicted_maxs):
    df_cp = (
        df.copy()
    )  # Create a copy of the DataFrame to avoid modifying the original one

    # Iterate over the predicted_mins indices (points where 'left' should be assigned)
    for idx in predicted_mins:
        # Access the 'time' column to get the datetime value for the corresponding index
        time_value = df_cp.index[idx]
        df_cp.loc[time_value, "Predicted"] = "left"

    # Iterate over the predicted_maxs indices (points where 'right' should be assigned)
    for idx in predicted_maxs:
        # Access the 'time' column to get the datetime value for the corresponding index
        time_value = df_cp.index[idx]
        df_cp.loc[time_value, "Predicted"] = "right"

    # fill Nan with 'right' or 'left' based on the above one. So if first is 'right' we will fill all NaN with 'right' until we find 'left'
    df_cp["Predicted"] = df_cp["Predicted"].fillna(method="ffill")
    df_cp["Behavior"] = df_cp["Behavior"].fillna(method="ffill")
    return df_cp


def get_score_for_intervals(df):
    # check when bevaviour is the same as predicted and return one score
    score = 0
    for index, row in df.iterrows():
        if row["Behavior"] == row["Predicted"]:
            score += 1

    return score, len(df)


def get_thr_metrick(
    new_filtered_mins,
    new_filtered_max,
    new_filtered_mins_counts,
    new_filtered_maxs_counts,
    df,
):
    new_method_mins, new_method_max = align_min_max_lists_counts_order(
        new_filtered_mins,
        new_filtered_max,
        new_filtered_mins_counts,
        new_filtered_maxs_counts,
    )
    inteval_df = add_predicted_transitions_to_df(df, new_method_mins, new_method_max)
    score, df_lenght = get_score_for_intervals(inteval_df)
    return score, df_lenght


def get_sec_metrick(df):
    df_cp = process_turns_with_restricted_distances(df)
    df_scores = calculate_scores_sec_metric(df_cp)
    overall_score = calculate_overall_score(df_scores)
    return overall_score
