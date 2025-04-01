# import os
# import random
# from copy import deepcopy
from typing import Literal

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from app.DataLoader import DataLoader
# from app.DataSpliter import GranularDataSplitter
# from app.transformers.smoother import Smoother
# from app.utils import unwrap_angles, unwrap_column
# from app.visualization_utils import draw_plotly
from app.workflow.plots import plot_positions


def get_apex_points(df: pd.DataFrame, turn: Literal["left", "right"] = None):

    if turn:
        return np.where(df[f"Apex"] == turn)[0].tolist()


def get_curve_points(df: pd.DataFrame, turn: Literal["L", "R", "left", "right"] = None):
    op_turn = "left" if turn == "right" else "right"
    if turn and "Curve" in df.columns:
        return np.where(df[f"Curve"] == turn)[0].tolist()
    if turn and "Curve" not in df.columns:
        return np.where(
            ((df[f"Behavior"] == turn) & (df[f"Status"] == "START"))
            | (df[f"Behavior"] == op_turn) & (df[f"Status"] == "STOP")
        )[0].tolist()
    return np.where(
        ((df[f"Behavior"] == turn) & (df[f"Status"] == "START"))
        | (df[f"Behavior"] == op_turn) & (df[f"Status"] == "STOP")
    )[0].tolist()


def gradient_descent(
    df,
    series,
    normalized_series,
    normalized_positions,
    start_indices,
    learning_rate,
    steps,
    momentum,
    find_maxima,
    printing=False,
    tolerance=1e-5,
    initial_velocity_range=(0.1, 0.5),
):
    velocities = np.zeros(len(normalized_positions))
    previous_positions = np.array(normalized_positions)
    series_len = len(series) - 1

    for step in range(steps):
        ahead_idx = np.clip(
            (normalized_positions + 1 / series_len) * series_len, 0, series_len
        ).astype(int)
        behind_idx = np.clip(
            (normalized_positions - 1 / series_len) * series_len, 0, series_len
        ).astype(int)

        gradients = (normalized_series[ahead_idx] - normalized_series[behind_idx]) / 2

        if find_maxima:
            velocities = momentum * velocities + learning_rate * gradients
        else:
            velocities = momentum * velocities - learning_rate * gradients

        normalized_positions = np.clip(normalized_positions + velocities, 0, 1)

        if np.all(np.abs(normalized_positions - previous_positions) < tolerance):
            print(f"Early stopping at step {step}")
            break

        previous_positions = normalized_positions.copy()

        if step % 50 == 0 and printing:
            positions = (normalized_positions * series_len).astype(int)
            plot_positions(series, positions, step, get_curve_points(df))

    positions = (normalized_positions * series_len).astype(int)
    return positions


def gradient_descent_full(
    df,
    start_indices,
    learning_rate=0.01,
    steps=1000,
    momentum=0.98,
    printing=False,
    selected_col="roll",
):
    series = df[selected_col].values

    # Normalize series data to 0-1
    normalized_series = (series - series.min()) / (series.max() - series.min())

    # Normalize start_indices to 0-1 and convert to NumPy array
    normalized_positions = np.array(start_indices) / (len(series) - 1)

    # Perform gradient descent to find minima
    min_positions = gradient_descent(
        df,
        series,
        normalized_series,
        normalized_positions.copy(),  # Ensure independent copies
        start_indices,
        learning_rate,
        steps,
        momentum,
        False,  # False for finding minima
        printing,
    )

    # Perform gradient descent to find maxima
    max_positions = gradient_descent(
        df,
        series,
        normalized_series,
        normalized_positions.copy(),  # Ensure independent copies
        start_indices,
        learning_rate,
        steps,
        momentum,
        True,  # True for finding maxima
        printing,
    )

    return min_positions, max_positions
