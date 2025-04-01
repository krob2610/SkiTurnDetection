# import os
# from copy import deepcopy
# from typing import Literal

import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# from app.DataLoader import DataLoader
# from app.DataSpliter import GranularDataSplitter
# from app.transformers.smoother import Smoother
# from app.utils import unwrap_angles, unwrap_column
# from app.visualization_utils import draw_plotly


def plot_positions(
    series,
    positions,
    step,
    curves=None,
    find_maxima=None,
    positions_counts=None,
    title="",
    apex=True,
):
    plt.figure(figsize=(12, 6), dpi=1200)
    plt.plot(series, label="Yaw", color="blue")
    plt.scatter(
        positions,
        series[positions],
        color="red",
        alpha=0.6,
        label=(
            "Predicted turn apex" if apex else "Predicted turn transition"
        ),  # "Predicted turn beginning/ending",
    )

    if positions_counts is not None:
        for pos, count in zip(positions, positions_counts):
            plt.text(
                pos,
                series[pos],
                str(count),
                fontsize=12,
                ha="center",
                va="bottom",
                color="black",
            )

    if curves is not None:
        plt.scatter(
            curves,
            series[curves],
            color="green",
            alpha=0.6,
            label=(
                "Actual turn apex" if apex else "Actual turn transition"
            ),  # "Actual turn beginning/ending",
        )

    plt.xlabel("Deciseconds", fontsize=20, labelpad=10)
    plt.ylabel("Radians", fontsize=20, labelpad=10)

    if find_maxima is None:
        (
            plt.title("Positions at step {}".format(step))
            if title == ""
            else plt.title(title, fontsize=24, pad=25)
        )
    else:
        extremum = "maxima" if find_maxima else "minima"
        (
            plt.title("Positions at step {} for {}".format(step, extremum))
            if title == ""
            else plt.title(title, fontsize=24, pad=25)
        )

    plt.legend(
        fontsize=20,
        markerscale=2,
        # bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fancybox=True,
        shadow=True,
    )

    plt.show()


def investigate_plot(df_complete, df_filtered, col="yaw"):
    """
    * Should be used after obtaining results, df complete is a complete dataframe(before meregin forecasts for example) and df filtered is one of transformed dfs like a TP or FP dfs
    """
    # Extract the roll values from the complete DataFrame
    roll_complete = df_complete[col].values

    # Convert turn_time to integer indices
    turn_time_actual_filtered = df_complete.index.get_indexer(
        df_filtered["Turn_Time_Actual"]
    )
    turn_time_predicted_filtered = df_complete.index.get_indexer(
        df_filtered["Turn_Time_Predicted"]
    )

    # Extract the values of interest from the filtered DataFrame
    roll_actual_filtered = df_filtered[f"{col}_Actual"].values
    roll_predicted_filtered = df_filtered[f"{col}_Predicted"].values

    # Plot the values
    plt.figure(figsize=(10, 6))

    # Plot complete DataFrame roll values
    plt.plot(roll_complete, color="blue", label=f"{col} (Complete)", alpha=0.5)

    # Plot filtered DataFrame points using integer indices
    plt.scatter(
        turn_time_actual_filtered,
        roll_actual_filtered,
        color="red",
        label="Actual (Filtered)",
        alpha=0.8,
    )
    plt.scatter(
        turn_time_predicted_filtered,
        roll_predicted_filtered,
        color="orange",
        label="Predicted (Filtered)",
        alpha=0.8,
    )

    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title(f"{col} and Turn Time")
    plt.legend()
    plt.show()


def plot_filter_and_extract_points(df_complete, df_filtered, col="yaw"):
    # Extract the roll values from the complete DataFrame
    roll_complete = df_complete[col].values

    # Convert turn_time to integer indices
    turn_time_actual_filtered = df_complete.index.get_indexer(
        df_filtered["Turn_Time_Actual"]
    )
    turn_time_predicted_filtered = df_complete.index.get_indexer(
        df_filtered["Turn_Time_Predicted"]
    )

    # Extract the values of interest from the filtered DataFrame
    roll_actual_filtered = df_filtered[f"{col}_Actual"].values
    roll_predicted_filtered = df_filtered[f"{col}_Predicted"].values

    # Plot the values
    plt.figure(figsize=(10, 6))

    # Plot complete DataFrame roll values
    plt.plot(roll_complete, color="blue", label=f"{col} (Complete)", alpha=0.5)

    # Plot filtered DataFrame points using integer indices
    plt.scatter(
        turn_time_actual_filtered,
        roll_actual_filtered,
        color="red",
        label="Actual (Filtered)",
        alpha=0.8,
    )
    plt.scatter(
        turn_time_predicted_filtered,
        roll_predicted_filtered,
        color="orange",
        label="Predicted (Filtered)",
        alpha=0.8,
    )

    plt.xlabel("Index")
    plt.ylabel(col)
    plt.title(f"{col} and Turn Time")
    plt.legend()
    plt.show()
