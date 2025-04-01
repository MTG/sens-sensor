from lib.client import get_data_for_sensor_id
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json


# Define transformation functions based on the rules
def transform_sources(row, config):
    # Birds, dogs, human, music, siren
    for key in ["birds", "dogs", "human", "music", "siren"]:
        if row[key] > config[key]["threshold"]:
            row[key] = 1  # row[key]
        else:
            row[key] = 0

    # Construction, vehicles, and nature based on LAeq
    for key in ["construction", "vehicles", "nature"]:
        if row["LAeq"] > 50:
            if row[key] > config[key]["threshold"]:
                row[key] = 1  # row[key]
            else:
                row[key] = 0
        else:
            row[key] = 0

    return row


def transform_pleasantness_eventfulness(row):
    # P_intg and E_intg
    if row["P_intg"] > 0:
        row["P_intg"] = 1
    else:
        row["P_intg"] = -1

    if row["E_intg"] > 0:
        row["E_intg"] = 1
    else:
        row["E_intg"] = -1

    return row


def general_plots(df, saving_path, start_str, end_str, config, fontsizes):
    # Check if the folder exists, otherwise create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print(f"Folder created at: {saving_path}")

    # Go over each column
    for col in df.columns[:-2]:
        fig, ax = plt.subplots(figsize=(16, 9))  # Adjust to your screen's aspect ratio
        threshold = config[col]["threshold"]

        if col in [
            "P_inst",
            "P_intg",
            "E_inst",
            "E_intg",
        ]:
            window_size = 1
            plt.ylim(-1, 1)
            # Add a horizontal dotted line at threshold
            plt.axhline(y=threshold, color="gray", linestyle="--", linewidth=0.5)
            # Color the background based on threshold
            ax.fill_between(
                df["elapsed_time"],
                df[col],  # 1
                threshold,
                where=(df[col] >= threshold),
                color="green",
                alpha=0.3,
            )
            ax.fill_between(
                df["elapsed_time"],
                df[col],  # -1
                threshold,
                where=(df[col] < threshold),
                color="red",
                alpha=0.3,
            )

        elif col in ["leq", "LAeq"]:
            window_size = 1
            # Y-axis limits
            plt.ylim(40, 90)
            # Add a horizontal dotted line at threshold
            plt.axhline(y=threshold, color="gray", linestyle="--", linewidth=0.5)
            # Color fill
            ax.fill_between(
                df["elapsed_time"],
                df[col],  # 90 or df[col]
                0,
                where=(df[col] >= threshold),
                color="red",
                alpha=0.3,
            )
        else:
            window_size = 1
            # Add a horizontal dotted line at threshold
            plt.axhline(y=threshold, color="gray", linestyle="--", linewidth=0.5)
            # Y-axis limits
            plt.ylim(0, 1)
            # Color fill
            print("col ", col)
            ax.fill_between(
                df["elapsed_time"],
                df[col],  # 1  or df[col]
                0,
                where=(df[col] >= threshold),
                color=config[col]["color"],
                alpha=0.6,
            )

        ax.plot(
            df["elapsed_time"],
            df[col].rolling(window=window_size, center=True).mean(),
            linewidth=0.6,
            color=config[col]["color-line"],
        )
        # Set sizes
        ax.set_title(
            f"Presence of {col}", fontsize=fontsizes["title"]
        )  # Title font size
        ax.set_xlabel(
            "Elapsed Time (minutes)", fontsize=fontsizes["label"]
        )  # X-axis label font size
        ax.set_ylabel(
            f"Probability of {col}", fontsize=fontsizes["label"]
        )  # Y-axis label font size
        ax.tick_params(
            axis="both", labelsize=fontsizes["axis"]
        )  # Tick labels font size

        # Add a text box with original datetime range
        text_box = f"Start: {start_str}\nEnd: {end_str}"
        plt.text(
            0.98,
            0.02,
            text_box,
            transform=plt.gcf().transFigure,
            fontsize=fontsizes["box"],
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
        )

        # Add a text box with percentage above threshold
        percentage = (df[col] > threshold).mean() * 100
        text_box = f"{percentage:.2f}%"
        plt.text(
            0.9,
            0.88,
            text_box,
            transform=plt.gcf().transFigure,
            fontsize=fontsizes["percentage"],
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
        )

        plt.margins(x=0)  # Removes white gaps on the x-axis
        plt.tight_layout()
        # Save the plot to a file (for example, 'plot.png')
        name = "time_vs_" + col + ".png"
        plt.savefig(os.path.join(saving_path, name), dpi=300, bbox_inches="tight")
    # plt.show()


def soundlight_plot_PE(df, fontsizes, saving_path):

    fig, ax = plt.subplots(figsize=(16, 9))  # Adjust to your screen's aspect ratio

    # soundlights graph P and E
    """ ax.plot(
        df["elapsed_time"],
        df["P_intg"].rolling(window=1, center=True).mean(),
        linewidth=1,
        color=config["P_intg"]["color-line"],
        label="P_intg",
        linestyle="solid",
    ) 
    ax.plot(
        df["elapsed_time"],
        df["E_intg"].rolling(window=1, center=True).mean(),
        linewidth=1,
        color=config["E_intg"]["color-line"],
        label="E_intg",
        linestyle="dashed",
    )"""

    # Color the background based on threshold
    # GREEN
    ax.fill_between(
        df["elapsed_time"],
        1,  # 1
        -1,
        where=((df["P_intg"] >= 0) & (df["E_intg"] <= 0)),
        color="green",
        alpha=0.2,
        label="Pleasant and Uneventful",
    )
    # Light Green
    ax.fill_between(
        df["elapsed_time"],
        1,  # 1
        -1,
        where=((df["P_intg"] >= 0) & (df["E_intg"] >= 0)),
        color="green",
        alpha=0.5,
        label="Pleasant and Eventful",
    )
    # RED
    ax.fill_between(
        df["elapsed_time"],
        1,  # 1
        -1,
        where=((df["P_intg"] <= 0) & (df["E_intg"] >= 0)),
        color="red",
        alpha=0.5,
        label="Unpleasant and Eventful",
    )
    # Light red
    ax.fill_between(
        df["elapsed_time"],
        1,  # 1
        -1,
        where=((df["P_intg"] <= 0) & (df["E_intg"] <= 0)),
        color="red",
        alpha=0.2,
        label="Unpleasant and Uneventful",
    )

    # Set sizes
    ax.set_title(f"Time vs P-E", fontsize=fontsizes["title"])  # Title font size
    ax.set_xlabel(
        "Elapsed Time (minutes)", fontsize=fontsizes["label"]
    )  # X-axis label font size
    ax.set_ylabel(
        "Prediction value", fontsize=fontsizes["label"]
    )  # Y-axis label font size
    ax.tick_params(axis="both", labelsize=fontsizes["axis"])  # Tick labels font size
    ax.legend(fontsize=fontsizes["legend"])

    # Add a text box with percentage above threshold
    """ percentage_P = (df["P_intg"] > 0).mean() * 100
    percentage_E = (df["E_intg"] > 0).mean() * 100
    text_box = f"Pleasantness: {percentage_P:.2f}%\nEventfulness: {percentage_E:.2f}%"
    plt.text(
        0.93,
        0.86,
        text_box,
        transform=plt.gcf().transFigure,
        fontsize=fontsizes["percentage"],
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
    ) """

    plt.ylim(-1, 1)
    plt.axhline(
        y=0, color="gray", linestyle="--", linewidth=0.5
    )  # Horizontal dotted line at y=0
    plt.margins(x=0)  # Removes white gaps on the x-axis
    plt.title(f"Predictions P-E - Soundlight")
    plt.xlabel("Elapsed Time (minutes)")
    plt.tight_layout()
    # Save the plot to a file (for example, 'plot.png')
    name = "soundlight_time_vs_PE.png"
    plt.savefig(os.path.join(saving_path, name), dpi=300, bbox_inches="tight")


def percetages_plot(df, config, fontsizes, saving_path):
    list_sources = [
        "birds",
        "construction",
        "dogs",
        "human",
        "music",
        "nature",
        "siren",
        "vehicles",
    ]
    # Calculate percentages using the configuration dictionary
    percentages_sources = {
        col: (df[col] > config[col]["threshold"]).mean() * 100 for col in list_sources
    }

    # Extract colors for each bar
    colors = [config[col]["color-line"] for col in percentages_sources.keys()]

    # Create the bar graph
    plt.figure(figsize=(16, 9))
    bars = plt.barh(
        list(percentages_sources.keys()), percentages_sources.values(), color=colors
    )
    plt.title("Time % of sources", fontsize=16)
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(0, 100)

    # Annotate the bar values
    for bar in bars:
        plt.text(
            bar.get_width() + 1,  # Position the text slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Center the text vertically on the bar
            f"{bar.get_width():.1f}%",  # Format the value as a percentage
            va="center",  # Vertical alignment
            ha="left",  # Horizontal alignment
            color="gray",  # Text color
            fontsize=fontsizes["box"],  # Font size
        )

    # Show the plot
    plt.tight_layout()
    # plt.show()
    # Save the plot to a file (for example, 'plot.png')
    name = "all_sources_percentages.png"
    plt.savefig(os.path.join(saving_path, name), dpi=300, bbox_inches="tight")


def quadrands_PE_plot(df, fontsizes, saving_path):
    fig, ax = plt.subplots(figsize=(16, 9))

    # Extract the columns
    x = df["P_intg"]
    y = df["E_intg"]

    # Determine axis limits
    x_limit = max(abs(x.min()), abs(x.max()), 0.5)
    y_limit = max(abs(y.min()), abs(y.max()), 0.5)
    if y_limit > x_limit:
        x_limit = y_limit
    else:
        y_limit = x_limit

    # First quadrant (X > 0, Y > 0)
    ax.scatter(
        x[(x > 0) & (y > 0)],
        y[(x > 0) & (y > 0)],
        color="green",
        alpha=0.5,
        label="Pleasant and Eventful",
    )

    # Fourth quadrant (X > 0, Y < 0)
    ax.scatter(
        x[(x > 0) & (y < 0)],
        y[(x > 0) & (y < 0)],
        color="green",
        alpha=0.2,
        label="Pleasant and Unventful",
    )

    # Second quadrant (X < 0, Y > 0)
    ax.scatter(
        x[(x < 0) & (y > 0)],
        y[(x < 0) & (y > 0)],
        color="red",
        alpha=0.5,
        label="Unpleasant and Eventful",
    )

    # Third quadrant (X < 0, Y < 0)
    ax.scatter(
        x[(x < 0) & (y < 0)],
        y[(x < 0) & (y < 0)],
        color="red",
        alpha=0.2,
        label="Unpleasant and Uneventful",
    )

    # Add labels and limits
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # X-axis
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)  # Y-axis
    ax.set_xlabel("Pleasantness", fontsize=fontsizes["label"])
    ax.set_ylabel("Eventfulness ", fontsize=fontsizes["label"])
    ax.set_title("Pleasantness vs Eventfulness")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    # Show plot
    plt.tight_layout()
    # plt.show()
    # Save the plot to a file (for example, 'plot.png')
    name = "PE_quadrants.png"
    plt.savefig(os.path.join(saving_path, name), dpi=300, bbox_inches="tight")


def get_data_from_sensor(start, end, sensorId, saving_path):

    config = {
        "P_inst": {
            "threshold": 0,
            "color-line": "#111111",
        },
        "P_intg": {
            "threshold": 0,
            "color-line": "#111111",
        },
        "E_inst": {
            "threshold": 0,
            "color-line": "#111111",
        },
        "E_intg": {
            "threshold": 0,
            "color-line": "#111111",
        },
        "leq": {
            "threshold": 90,
            "color-line": "#111111",
        },
        "LAeq": {
            "threshold": 65,
            "color-line": "#111111",
        },
        "birds": {
            "threshold": 0.5,
            "color": "#8F7E8A",
            "color-line": "#111111",
        },
        "construction": {
            "threshold": 0.5,
            "color": "#EE9E2E",
            "color-line": "#111111",
        },
        "dogs": {
            "threshold": 0.5,
            "color": "#84B66F",
            "color-line": "#111111",
        },
        "human": {
            "threshold": 0.5,
            "color": "#FABA32",
            "color-line": "#111111",
        },
        "music": {
            "threshold": 0.5,
            "color": "#0DB2AC",
            "color-line": "#111111",
        },
        "nature": {
            "threshold": 0.5,
            "color": "#A26294",
            "color-line": "#111111",
        },
        "siren": {
            "threshold": 0.5,
            "color": "#FC694D",
            "color-line": "#111111",
        },
        "vehicles": {
            "threshold": 0.8,
            "color": "#CF6671",
            "color-line": "#111111",
        },
    }
    config_processed = {
        "P_inst": {
            "threshold": 0,
            "color-line": "#000000",
        },
        "P_intg": {
            "threshold": 0,
            "color-line": "#000000",
        },
        "E_inst": {
            "threshold": 0,
            "color-line": "#000000",
        },
        "E_intg": {
            "threshold": 0,
            "color-line": "#000000",
        },
        "leq": {
            "threshold": 90,
            "color-line": "#000000",
        },
        "LAeq": {
            "threshold": 65,
            "color-line": "#000000",
        },
        "birds": {
            "threshold": 0.5,
            "color": "#8F7E8A",
            "color-line": "#8F7E8A",
        },
        "construction": {
            "threshold": 0.5,
            "color": "#EE9E2E",
            "color-line": "#EE9E2E",
        },
        "dogs": {
            "threshold": 0.5,
            "color": "#84B66F",
            "color-line": "#84B66F",
        },
        "human": {
            "threshold": 0.5,
            "color": "#FABA32",
            "color-line": "#FABA32",
        },
        "music": {
            "threshold": 0.5,
            "color": "#0DB2AC",
            "color-line": "#0DB2AC",
        },
        "nature": {
            "threshold": 0.5,
            "color": "#A26294",
            "color-line": "#A26294",
        },
        "siren": {
            "threshold": 0.5,
            "color": "#FC694D",
            "color-line": "#FC694D",
        },
        "vehicles": {
            "threshold": 0.8,
            "color": "#CF6671",
            "color-line": "#CF6671",
        },
    }

    fontsizes = {
        "title": 20,
        "legend": 14,
        "label": 14,
        "axis": 12,
        "box": 14,
        "percentage": 20,
    }

    # Check if the folder exists, otherwise create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print(f"Folder created at: {saving_path}")

    # region 1) Download data and generate dataframe
    data_list = []
    all_data = get_data_for_sensor_id(
        sensor_id=sensorId, start_date=start, end_date=end
    )
    for data in all_data:
        json_data = data["data"]
        # Extraemos los datos planos
        row = {
            "P_inst": float(json_data["P_inst"]),
            "P_intg": float(json_data["P_intg"]),
            "E_inst": float(json_data["E_inst"]),
            "E_intg": float(json_data["E_intg"]),
            "leq": float(json_data["leq"]),
            "LAeq": float(json_data["LAeq"]),
        }
        sources = {key: float(value) for key, value in json_data["sources"].items()}
        row.update(sources)
        row["datetime"] = json_data["datetime"]
        # Add row to list
        data_list.append(row)

    # List --> dataframe
    df = pd.DataFrame(data_list)
    # Convert 'datetime' column to pandas datetime format
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%dT%H:%M:%S%z")

    # Calculate elapsed time in minutes and add to column
    start_time = df["datetime"].min()
    end_time = df["datetime"].max()
    df["elapsed_time"] = (
        df["datetime"] - start_time
    ).dt.total_seconds() / 60  # Convert to minutes

    # Apply processing to dataframe copy
    df_processed = df.copy()
    df_processed = df_processed.apply(
        lambda row: transform_sources(row, config=config), axis=1
    )
    df_processed = df_processed.apply(transform_pleasantness_eventfulness, axis=1)

    df.to_csv(os.path.join(saving_path, "data.csv"), index=False)
    df_processed.to_csv(os.path.join(saving_path, "data_processed.csv"), index=False)

    # Original start and end times as strings
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # endregion

    # 2) Generate raw plots - Individual plots: Time vs each column separately
    general_plots(
        df=df,
        saving_path=os.path.join(saving_path, "raw"),
        start_str=start_str,
        end_str=end_str,
        config=config,
        fontsizes=fontsizes,
    )

    # 3) Generate processed plots - Individual plots: Time vs each column separately
    general_plots(
        df=df_processed,
        saving_path=os.path.join(saving_path, "processed"),
        start_str=start_str,
        end_str=end_str,
        config=config_processed,
        fontsizes=fontsizes,
    )

    # 4) Generate other interesting plots
    soundlight_plot_PE(df=df_processed, fontsizes=fontsizes, saving_path=saving_path)

    percetages_plot(
        df=df_processed,
        config=config_processed,
        fontsizes=fontsizes,
        saving_path=saving_path,
    )

    quadrands_PE_plot(df=df, fontsizes=fontsizes, saving_path=saving_path)


def main_get_data_from_file(csv_file):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, delimiter=";")

    # Iterate over rows in DataFrame
    for index, row in df.iterrows():
        # Access values of each row by column name
        # Example: Access value of the 'column_name' in the row
        value = row["Date"]
        print(value)

        start_array = row["Start_time"].split(":")
        end_array = row["End_time"].split(":")
        date_array = row["Date"].split("-")

        start = datetime.datetime(
            int(date_array[0]),
            int(date_array[1]),
            int(date_array[2]),
            int(start_array[0]),
            int(start_array[1]),
            int(start_array[2]),
        )
        end = datetime.datetime(
            int(date_array[0]),
            int(date_array[1]),
            int(date_array[2]),
            int(end_array[0]),
            int(end_array[1]),
            int(end_array[2]),
        )
        sensorId = row["Sensor_Id"]
        saving_path = os.path.join(row["Saving_Path"], row["Name"])
        get_data_from_sensor(start, end, sensorId, saving_path)

        # Save metadata in txt
        # Step 1: Create your dictionary
        data_dict = {
            "Name": row["Name"],
            "Sensor_Id": row["Sensor_Id"],
            "Location": row["Location"],
            "Date": row["Date"],
            "Start_time": row["Start_time"],
            "End_time": row["End_time"],
        }

        # Step 2: Save the dictionary as a JSON file
        with open(os.path.join(saving_path, "point_info.json"), "w") as json_file:
            json.dump(data_dict, json_file, indent=4)


def main_get_data_specific(
    sensor_id,
    start_year,
    start_month,
    start_day,
    start_hour,
    start_minute,
    start_second,
    end_year,
    end_month,
    end_day,
    end_hour,
    end_minute,
    end_second,
    saving_path,
):

    start = datetime.datetime(
        start_year,
        start_month,
        start_day,
        start_hour,
        start_minute,
        start_second,
    )
    end = datetime.datetime(
        end_year,
        end_month,
        end_day,
        end_hour,
        end_minute,
        end_second,
    )
    get_data_from_sensor(start, end, sensor_id, saving_path)

    # Save metadata in txt
    # Step 1: Create your dictionary
    """ data_dict = {
        "Name": row["Name"],
        "Sensor_Id": row["Sensor_Id"],
        "Location": row["Location"],
        "Date": row["Date"],
        "Start_time": row["Start_time"],
        "End_time": row["End_time"],
    }

    # Step 2: Save the dictionary as a JSON file
    with open(os.path.join(saving_path, "point_info.json"), "w") as json_file:
        json.dump(data_dict, json_file, indent=4) """
