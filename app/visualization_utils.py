import ast
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

from app.DataSpliter import CURVE
from app.labels_namespace import ORIENTATION, OTHER


def draw_plotly_new(df: pd.DataFrame, col: str, granular: bool = False, title=""):
    """It generates 3 plots with value over time based on Behaviour (Left/Right)"""

    # Determine labels based on the column name
    labels = ORIENTATION if col == "Orientation" else OTHER

    df_copy = df.copy()

    # Ensure the 'time' column exists or set it as the index if not
    if "time" not in df_copy.columns:
        df_copy.reset_index(inplace=True)

    if not granular:
        df_split = df_copy[col].apply(pd.Series)
    else:
        df_split = df_copy[labels]

    fig = go.Figure()
    # Define a list of colors
    colors = ["green", "blue", "orange"]
    # Add traces for each coordinate (e.g., X, Y, Z)
    for i, (cord, label) in enumerate(zip(df_split.columns, labels)):
        if col == "Orientation" and i < 4:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_copy["time"],
                y=df_split[cord],
                mode="lines",
                name=f"{label}",
                line=dict(color=colors[i % len(colors)]),  # Assign different colors
            )
        )
        # Check if df has the 'Behavior' column
        if "Behavior" in df_copy.columns:
            # Trace for 'Left' Behavior
            fig.add_trace(
                go.Scatter(
                    x=df_copy[
                        (df_copy["Behavior"] == "left") & (df_copy["Status"] == "START")
                    ]["time"],
                    y=df_split[
                        (df_copy["Behavior"] == "left") & (df_copy["Status"] == "START")
                    ][cord],
                    mode="markers",
                    marker=dict(color="red", size=6),
                    name=f"{label} Left turn beginning",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_copy[
                        (df_copy["Behavior"] == "left") & (df_copy["Status"] == "STOP")
                    ]["time"],
                    y=df_split[
                        (df_copy["Behavior"] == "left") & (df_copy["Status"] == "STOP")
                    ][cord],
                    mode="markers",
                    marker=dict(color="orange", size=6),
                    name=f"{label} Left turn ending",
                    showlegend=False,
                )
            )

            # Trace for 'Right' Behavior
            fig.add_trace(
                go.Scatter(
                    x=df_copy[
                        (df_copy["Behavior"] == "right")
                        & (df_copy["Status"] == "START")
                    ]["time"],
                    y=df_split[
                        (df_copy["Behavior"] == "right")
                        & (df_copy["Status"] == "START")
                    ][cord],
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    name=f"{label} Right turn beginning",
                    showlegend=False,
                )
            )
            # Trace for 'Right' Behavior
            fig.add_trace(
                go.Scatter(
                    x=df_copy[
                        (df_copy["Behavior"] == "right") & (df_copy["Status"] == "STOP")
                    ]["time"],
                    y=df_split[
                        (df_copy["Behavior"] == "right") & (df_copy["Status"] == "STOP")
                    ][cord],
                    mode="markers",
                    marker=dict(color="purple", size=6),
                    name=f"{label} Right turn ending",
                    showlegend=False,
                )
            )

        if "Apex" in df_copy.columns:
            # Trace for 'Left' Behavior
            fig.add_trace(
                go.Scatter(
                    x=df_copy[df_copy["Apex"] == "left"]["time"],
                    y=df_split[df_copy["Apex"] == "left"][cord],
                    mode="markers",
                    marker=dict(color="pink", size=6),
                    name=f"{label} Left apex",
                    showlegend=False,
                )
            )

            # Trace for 'Right' Behavior
            fig.add_trace(
                go.Scatter(
                    x=df_copy[df_copy["Apex"] == "right"]["time"],
                    y=df_split[df_copy["Apex"] == "right"][cord],
                    mode="markers",
                    marker=dict(color="indigo", size=6),
                    name=f"{label} Right apex",
                    showlegend=False,
                )
            )

    # Add legends for Behaviors
    if "Behavior" in df_copy.columns:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="red", size=6),
                legendgroup="Left",
                name="Left turn beginning",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="orange", size=6),
                legendgroup="Left",
                name="Left turn ending",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="blue", size=6),
                legendgroup="Right",
                name="Right turn beginning",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="purple", size=6),
                legendgroup="Right",
                name="Right turn ending",
                showlegend=True,
            )
        )
    if "Apex" in df_copy.columns:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="pink", size=6),
                legendgroup="Left",
                name="Left apex",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="indigo", size=6),
                legendgroup="Right",
                name="Right apex",
                showlegend=True,
            )
        )

    # Update layout with titles, axis labels, and legend position
    fig.update_layout(
        title=dict(
            text=(
                f"Marked beginnings/endings of turns along with their apexes on {col} over time"
                if title == ""
                else title
            ),
            font=dict(size=30),  # Increase title font size
        ),
        xaxis_title=dict(
            text="Time", font=dict(size=20)  # Increase x-axis label font size
        ),
        yaxis_title=dict(
            text="Radians", font=dict(size=20)  # Increase y-axis label font size
        ),
        legend=dict(
            x=0.01,  # Position the legend inside the plot
            y=0.99,
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=2,
            font=dict(size=30),  # Increase legend text size
            itemsizing="constant",  # Ensure consistent marker size in legend
            itemwidth=40,  # Increase width of legend items
        ),
        width=1200,  # Set the width of the plot
        height=600,  # Set the height of the plot
    )

    # Show the plot
    fig.show(
        config={
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "filename": "custom_image",
                "scale": 6,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
    )


def draw_plotly(df: pd.DataFrame, col: str, granular: bool = False):
    """it should generate a 3 plots with value over time"""
    labels = ORIENTATION if col == "Orientation" else OTHER

    df_copy = df.copy()
    # df_copy["time"] = pd.to_datetime(df_copy["time"])

    if "time" not in df_copy.columns:
        df_copy.reset_index(inplace=True)
    # df_copy[col] = df_copy[col].str.replace("nan", "0").apply(ast.literal_eval)
    if not granular:
        df_split = df_copy[col].apply(pd.Series)
    else:
        df_split = df_copy[labels]
        # df_split = df_split.rename(columns={0: 'X', 1: 'Y', 2: 'Z'})
    # df_split = df_split.rename(columns={0: 'X', 1: 'Y', 2: 'Z'})

    fig = go.Figure()

    for i, (cord, label) in enumerate(zip(df_split.columns, labels)):
        if col == "Orientation" and i < 4:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_copy["time"], y=df_split[cord], mode="lines", name=f"{label}"
            )
        )

        # check if df has curve column
        if CURVE in df_copy.columns:
            # Trace for 'Left' curve
            fig.add_trace(
                go.Scatter(
                    x=df_copy[df_copy[CURVE] == "L"]["time"],
                    y=df_split[df_copy[CURVE] == "L"][cord],
                    mode="markers",
                    marker=dict(color="red", size=6),
                    name=f"{label} Left Curve Points",
                    showlegend=False,
                )
            )

            # Trace for 'Right' curve
            fig.add_trace(
                go.Scatter(
                    x=df_copy[df_copy[CURVE] == "R"]["time"],
                    y=df_split[df_copy[CURVE] == "R"][cord],
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    name=f"{label} Right Curve Points",
                    showlegend=False,
                )
            )
    # add legends for Turns - dummy
    if CURVE in df_copy.columns:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="red", size=6),
                legendgroup="Left",
                name="Left Turn",
                showlegend=True,
            )
        )

        # Dummy trace for 'Right' curve legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="blue", size=6),
                legendgroup="Right",
                name="Right Turn",
                showlegend=True,
            )
        )
    fig.update_layout(title=f"{col} over time", xaxis_title="time", yaxis_title=col)

    fig.show()


def select_data_range(start: str, end: str, df: pd.DataFrame):
    """Select a range of data from a DataFrame based on the start and end times."""
    start_time = pd.to_datetime(start).time()
    end_time = pd.to_datetime(end).time()

    # throw an exception when the start time is greater than the end time
    if start_time > end_time:
        raise ValueError("The start time cannot be greater than the end time.")

    # Select data between start_time and end_time
    return df.between_time(start_time, end_time)


def draw_plot_plt(
    df: pd.DataFrame, start: str, end: str, col: str, window_size: int = 100
):
    """Function to create a plot for each value in a given column over time."""
    df_copy = df.copy()
    df_copy = select_data_range(df=df_copy, start=start, end=end)
    if "time" not in df_copy.columns:
        df_copy.reset_index(inplace=True)
    df_split = df_copy[col].apply(pd.Series)

    labels = ORIENTATION if col == "Orientation" else OTHER

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the y-axis limits
    ax.set_ylim(df_split.min().min(), df_split.max().max())

    def update(frame):
        ax.clear()
        for i, (cord, label) in enumerate(zip(df_split.columns, labels)):
            if col == "Orientation" and i < 4:
                continue
            ax.plot(
                df_copy["time"].iloc[: frame + 1],
                df_split[cord].iloc[: frame + 1],
                label=f"{label}",
            )
        ax.set_title(f"{col} over time")
        ax.set_xlabel("time")
        ax.set_ylabel(col)
        ax.legend()

        # Keep the y-axis limits fixed
        ax.set_ylim(df_split.min().min(), df_split.max().max())

        # Set the x-axis limits to create a moving window of window_size seconds
        if frame > window_size:
            ax.set_xlim(
                df_copy["time"].iloc[frame - window_size], df_copy["time"].iloc[frame]
            )

    duration = (df_copy["time"].iloc[-1] - df_copy["time"].iloc[0]).total_seconds()

    frames = len(df_copy)

    # Calculate the duration of one frame of the animation
    if frames > 1:
        frame_duration = (
            df_copy["time"].iloc[1] - df_copy["time"].iloc[0]
        ).total_seconds()
    else:
        frame_duration = duration

    # Prepare the animation
    ani = FuncAnimation(
        fig, update, frames=frames, repeat=False, interval=frame_duration * 1000
    )

    # Save the animation to a GIF file
    file_name = f"{col}_animation.gif"
    if os.path.exists(file_name):
        index = 1
        while os.path.exists(f"{col}_animation_{index}.gif"):
            index += 1
        file_name = f"{col}_animation_{index}.gif"

    ani.save(file_name, writer="pillow", fps=1 / frame_duration)

    plt.close()

    print(
        f"Animation saved as {file_name}. Please open the file to view the animation."
    )


def plot_predicted_transitions(interval_df):
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=interval_df.index, y=interval_df['yaw'], mode='lines', name='Yaw'))
    fig.add_trace(
        go.Scatter(
            x=interval_df[interval_df["Predicted"] == "left"].index,
            y=interval_df[interval_df["Predicted"] == "left"]["yaw"],
            mode="markers",
            marker=dict(color="green"),
            name="Predicted left",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=interval_df[interval_df["Predicted"] == "right"].index,
            y=interval_df[interval_df["Predicted"] == "right"]["yaw"],
            mode="markers",
            marker=dict(color="blue"),
            name="Predicted right",
        )
    )
    fig.update_layout(
        title=dict(
            text=(f"Predicted turns on yaw"),
            font=dict(size=30),  # Increase title font size
        ),
        xaxis_title=dict(
            text="Time", font=dict(size=20)  # Increase x-axis label font size
        ),
        yaxis_title=dict(
            text="Radians", font=dict(size=20)  # Increase y-axis label font size
        ),
        legend=dict(
            x=0.01,  # Position the legend inside the plot
            y=0.99,
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=2,
            font=dict(size=30),  # Increase legend text size
            itemsizing="constant",  # Ensure consistent marker size in legend
            itemwidth=40,  # Increase width of legend items
        ),
        width=1200,  # Set the width of the plot
        height=600,  # Set the height of the plot
    )
    fig.show(
        config={
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "filename": "custom_image",
                "scale": 6,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
    )
