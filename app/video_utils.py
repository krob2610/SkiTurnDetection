import datetime
import os
import re

import cv2
import ffmpy
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

from app.labels_namespace import ORIENTATION, OTHER


def select_data_range(start: str, end: str, df: pd.DataFrame):
    """Select a range of data from a DataFrame based on the start and end times."""
    start_time = pd.to_datetime(start).time()
    end_time = pd.to_datetime(end).time()

    # throw an exception when the start time is greater than the end time
    if start_time > end_time:
        raise ValueError("The start time cannot be greater than the end time.")

    # Select data between start_time and end_time
    return df.between_time(start_time, end_time)


def gif_to_mp4(input: str, output: str) -> None:
    """Convert a GIF file to an MP4 file."""
    ff = ffmpy.FFmpeg(inputs={input: None}, outputs={output: None})
    ff.run()


def draw_plot_plt(
    df: pd.DataFrame,
    start: str,
    end: str,
    col: str,
    output: str,
    window_size: int = 100,
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
    file_name = f"{output}.gif"

    ani.save(file_name, writer="pillow", fps=1 / frame_duration)

    plt.close()
    gif_to_mp4(input=file_name, output=file_name.replace(".gif", ".mp4"))
    print(
        f"Animation saved as {file_name}. Please open the file to view the animation."
    )


def draw_plotly_new_animation(
    df: pd.DataFrame,
    col: str,
    granular: bool = False,
    title="",
    output_dir="output_dir_anmation",
):
    """It generates 3 plots with value over time based on Behaviour (Left/Right)"""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    colors = ["green", "blue", "orange"]

    fig = go.Figure()
    fig.update_layout(
        title=(
            f"Marked beginnings/endings of turnson {col} over time"
            if title == ""
            else title
        ),
        xaxis_title="Time",
        yaxis_title="Radians",
    )

    # Define a list of colors
    # Loop through the time steps and create one frame per index
    for idx in range(len(df_copy)):
        fig.data = []  # Clear previous data for the new frame

        # Add traces for each coordinate (e.g., X, Y, Z)
        for i, (cord, label) in enumerate(zip(df_split.columns, labels)):
            if col == "Orientation" and i < 4:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_copy["time"][: idx + 1],  # Only plot up to the current index
                    y=df_split[cord][: idx + 1],  # Only plot up to the current index
                    mode="lines",
                    name=f"{label}",
                    line=dict(color=colors[i % len(colors)]),  # Assign different colors
                )
            )
            if label == "yaw":
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

            # Check if df has the 'Behavior' column and add markers as the plot progresses
            if "Behavior" in df_copy.columns:
                # Only add behavior markers if the current time is greater than or equal to 'time' of the behavior
                left_start_condition = (
                    (df_copy["Behavior"] == "left")
                    & (df_copy["Status"] == "START")
                    & (df_copy["time"] <= df_copy["time"].iloc[idx])
                )
                left_stop_condition = (
                    (df_copy["Behavior"] == "left")
                    & (df_copy["Status"] == "STOP")
                    & (df_copy["time"] <= df_copy["time"].iloc[idx])
                )
                right_start_condition = (
                    (df_copy["Behavior"] == "right")
                    & (df_copy["Status"] == "START")
                    & (df_copy["time"] <= df_copy["time"].iloc[idx])
                )
                right_stop_condition = (
                    (df_copy["Behavior"] == "right")
                    & (df_copy["Status"] == "STOP")
                    & (df_copy["time"] <= df_copy["time"].iloc[idx])
                )

                # Left turn behavior markers
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[left_start_condition]["time"],
                        y=df_split[left_start_condition][cord],
                        mode="markers",
                        marker=dict(color="red", size=6),
                        name=f"{label} Left turn beginning",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[left_stop_condition]["time"],
                        y=df_split[left_stop_condition][cord],
                        mode="markers",
                        marker=dict(color="orange", size=6),
                        name=f"{label} Left turn ending",
                        showlegend=False,
                    )
                )

                # Right turn behavior markers
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[right_start_condition]["time"],
                        y=df_split[right_start_condition][cord],
                        mode="markers",
                        marker=dict(color="blue", size=6),
                        name=f"{label} Right turn beginning",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[right_stop_condition]["time"],
                        y=df_split[right_stop_condition][cord],
                        mode="markers",
                        marker=dict(color="purple", size=6),
                        name=f"{label} Right turn ending",
                        showlegend=False,
                    )
                )

            # Handle Apex markers similarly, only show them when the trace reaches the corresponding time
            if "Apex" in df_copy.columns:
                left_apex_condition = (df_copy["Apex"] == "left") & (
                    df_copy["time"] <= df_copy["time"].iloc[idx]
                )
                right_apex_condition = (df_copy["Apex"] == "right") & (
                    df_copy["time"] <= df_copy["time"].iloc[idx]
                )

                # Left Apex marker
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[left_apex_condition]["time"],
                        y=df_split[left_apex_condition][cord],
                        mode="markers",
                        marker=dict(color="pink", size=6),
                        name=f"{label} Left apex",
                        showlegend=False,
                    )
                )

                # Right Apex marker
                fig.add_trace(
                    go.Scatter(
                        x=df_copy[right_apex_condition]["time"],
                        y=df_split[right_apex_condition][cord],
                        mode="markers",
                        marker=dict(color="indigo", size=6),
                        name=f"{label} Right apex",
                        showlegend=False,
                    )
                )

            frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
            fig.write_image(frame_path)

    # # Update layout with titles and axis labels
    # fig.update_layout(
    #     title=(
    #         f"Marked beginnings/endings of turns along with their apexes on {col} over time"
    #         if title == ""
    #         else title
    #     ),
    #     xaxis_title="Time",
    #     yaxis_title="Radians",
    # )

    # # Show the plot
    # # Save the current frame as a PNG image
    # frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
    # fig.write_image(frame_path)


def create_video_from_frames(frame_dir="frames", output_video="plot_video.mp4", fps=10):
    frames = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )

    if not frames:
        raise ValueError("No frames found in the specified directory.")

    # Read the first frame to get the image size
    frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, _ = frame.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' codec for mp4 file
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each frame to the video
    for frame_file in frames:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video}")


def cut_video(
    input_path: str, output_path: str, start: pd.Timestamp, end: pd.Timestamp
):
    if os.path.exists(output_path):
        os.remove(output_path)

    # Extract start and end time as HH:MM:SS.mmm
    start_time = f"{start.hour:02}:{start.minute:02}:{start.second:02}.{start.microsecond // 1000:03}"
    end_time = (
        f"{end.hour:02}:{end.minute:02}:{end.second:02}.{end.microsecond // 1000:03}"
    )

    # Build ffmpeg command
    ff = ffmpy.FFmpeg(
        inputs={input_path: None},
        outputs={output_path: f"-ss {start_time} -to {end_time}"},
    )

    ff.run()


def merge_videos(input1: str, input2: str, output: str):
    ff = ffmpy.FFmpeg(
        inputs={input1: None, input2: None},
        outputs={output: '-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]"'},
    )
    ff.run()


def merge_videos_3(
    input1: str, input2: str, input3: str, output: str, target_height: int = 1080
):
    print("Started merge!")

    # Check if the output file exists and delete it if it does
    if os.path.exists(output):
        os.remove(output)

    # Merge the three input videos with scaling to a common height
    ff = ffmpy.FFmpeg(
        inputs={input1: None, input2: None, input3: None},
        outputs={
            output: f'-filter_complex "[0:v]scale=-1:{target_height}[v0];'
            f"[1:v]scale=-1:{target_height}[v1];"
            f"[2:v]scale=-1:{target_height}[v2];"
            f'[v0][v1][v2]hstack=inputs=3[v]" -map "[v]"'
        },
    )

    # Run the command
    ff.run()
