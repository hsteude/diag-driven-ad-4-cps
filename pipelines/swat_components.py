from kfp import dsl
from kfp.dsl import HTML, Input, Output, Dataset, Artifact
from typing import List


@dsl.component(
    packages_to_install=["pandas==1.5.3", "plotly"],
    base_image="python:3.9",
)
def show_results(
    metrics_dict_vanilla: dict,
    metrics_dict_multi_latent: dict,
    metrics_dict_combined: dict,
    fc1_table: Output[HTML],
) -> None:
    import pandas as pd
    import plotly.graph_objects as go

    result_df = pd.concat(
        [
            pd.DataFrame(metrics_dict_vanilla).T,
            pd.DataFrame(metrics_dict_combined).T,
            pd.DataFrame(metrics_dict_multi_latent).T,
        ],
        axis=1,
        keys=["fc1_vanilla", "fc1_univar", "fc1_multi-latend"],
    )
    fc1_results_df = result_df.xs("fc1", level=1, axis=1)

    # Round the dataframe values to two decimal places
    df_plot = fc1_results_df.round(2)

    # Add the index as a column named "subsystem"
    df_plot["subsystem"] = df_plot.index

    # Reorder columns to make 'subsystem' the first column
    df_plot = df_plot[["subsystem", "fc1_vanilla", "fc1_univar", "fc1_multi-latend"]]

    # Create the Plotly table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=df_plot.columns,
                ),
                cells=dict(
                    values=[
                        df_plot["subsystem"],
                        df_plot["fc1_vanilla"],
                        df_plot["fc1_univar"],
                        df_plot["fc1_multi-latend"],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.write_html(fc1_table.path)


@dsl.component(
    packages_to_install=["pandas==1.5.3", "pyarrow", "loguru"],
    base_image="python:3.9",
)
def get_metrics(
    attack_residuals_df_in: Input[Dataset],
    labels_df_in: Input[Dataset],
    labels_time_series_in: Input[Dataset],
    detrend_window: int,
) -> dict:
    import numpy as np
    import pandas as pd
    from numpy.typing import ArrayLike
    from loguru import logger

    def moving_average(data: ArrayLike, window_size: int) -> ArrayLike:
        """
        Compute the moving average of a dataset.
        """
        return (
            pd.Series(data)
            .rolling(window=window_size, min_periods=1)
            .median()
            .to_numpy()
        )

    def detrend_signal(data: ArrayLike, window_size: int) -> ArrayLike:
        """
        Detrend a signal using moving average.
        """
        trend = moving_average(data, window_size)
        detrended = [
            val - trend[i] if val - trend[i] > 0 else 0 for i, val in enumerate(data)
        ]
        return np.array(detrended)

    def harmonic_mean(a: float, b: float) -> float:
        """
        Compute the harmonic mean of two quantities.
        """
        if (
            a + b == 0
        ):  # Handle the case where both a and b are zero to prevent division by zero
            return 0
        return 2 * a * b / (a + b)

    def get_best_metrics(
        detrended_df: pd.DataFrame,
        pred_col: str,
        label_col: str,
        labels_df: pd.DataFrame,
        labels_time_series: pd.DataFrame,
    ) -> tuple:
        """Find and return the best metrics by iterating over potential thresholds."""
        logger.info(
            f"Optimizing threshold for pred_col: {pred_col} and label col: {label_col}"
        )
        best_fc1 = 0.0
        best_threshold = 0.0
        best_precision_time_point = 0.0
        best_recall_event_based = 0.0
        thresholds = np.linspace(0, 20, 100)

        for threshold in thresholds:
            predicted_labels = (detrended_df[pred_col] > threshold).astype(int).values

            # Time Point-based Precision and Recall
            TP = np.sum((predicted_labels == 1) & (labels_time_series[label_col] == 1))
            FP = np.sum((predicted_labels == 1) & (labels_time_series[label_col] == 0))
            FN = np.sum((predicted_labels == 0) & (labels_time_series[label_col] == 1))

            if TP + FP == 0 or TP + FN == 0:
                continue

            precision_time_point = TP / (TP + FP)

            # Event-based Precision and Recall
            true_positive_events = 0
            false_negative_events = 0

            for _, row in labels_df.iterrows():
                start_idx = np.where(labels_time_series.index == row["Start Time"])[0][
                    0
                ]
                end_idx = np.where(labels_time_series.index == row["End Time"])[0][0]
                if 1 in predicted_labels[start_idx : end_idx + 1]:
                    true_positive_events += 1
                else:
                    false_negative_events += 1

            recall_event_based = true_positive_events / (
                true_positive_events + false_negative_events
            )

            fc1 = harmonic_mean(precision_time_point, recall_event_based)
            if fc1 > best_fc1:
                best_fc1 = fc1
                best_threshold = threshold
                best_precision_time_point = precision_time_point
                best_recall_event_based = recall_event_based
        return (
            best_fc1,
            best_precision_time_point,
            best_recall_event_based,
            best_threshold,
        )

    def compute_metrics(
        attack_residuals_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        labels_time_series: pd.DataFrame,
        window_size: int,
    ) -> dict:
        """
        Compute and return metrics based on optimal thresholds for event-based recall
        and time-point precision.
        """
        detrended_df = pd.DataFrame()
        for col in attack_residuals_df.columns:
            detrended_df[col] = detrend_signal(
                attack_residuals_df[col].values, window_size
            )

        pred_cols = ["mse_full"] + [f"mse_comp_{c+1}" for c in range(6)]
        label_cols = ["label_full"] + [f"label_comp_{c+1}" for c in range(6)]
        result_dct_keys = ["full"] + [f"comp_{c+1}" for c in range(6)]
        result_dct = {}

        for pred_col, label_col, key in zip(pred_cols, label_cols, result_dct_keys):
            fc1, precision_time_point, recall_event_based, threshold = get_best_metrics(
                detrended_df, pred_col, label_col, labels_df, labels_time_series
            )
            result_dct[key] = {
                "fc1": fc1,
                "precision_time_point": precision_time_point,
                "recall_event_based": recall_event_based,
                "threshold": threshold,
            }
            logger.info(f"Best f1 for {key}:  {result_dct[key]}")

        return result_dct

    attack_residuals_df = pd.read_parquet(attack_residuals_df_in.path)
    labels_df = pd.read_parquet(labels_df_in.path)
    labels_time_series = pd.read_parquet(labels_time_series_in.path)
    result_dct = compute_metrics(
        attack_residuals_df, labels_df, labels_time_series, detrend_window
    )
    return result_dct


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
    ],
    base_image="python:3.9",
)
def format_labels_df_for_metric_computation(
    attack_df_in: Input[Dataset],
    labels_df: Input[Dataset],
    seq_len: int,
    anomaly_time_series: Output[Dataset],
):
    import pandas as pd
    import numpy as np

    label_df = pd.read_parquet(labels_df.path)
    attack_df = pd.read_parquet(attack_df_in.path)

    # Function to check if a timestamp is within any attack period
    def is_in_attack_period(timestamp, attack_periods):
        for start, end in attack_periods:
            if start <= timestamp <= end:
                return 1
        return 0

    # Filter rows where 'Actual Change' column is 'Yes'
    filtered_label_df = label_df[label_df["Actual Change"] == "Yes"]

    # Determine all unique subsystems
    all_subsystems = set(
        item for sublist in filtered_label_df["subsystem"].tolist() for item in sublist
    )

    # Dictionary to store anomaly label arrays for each subsystem
    subsystem_anomaly_arrays = {}

    # For the entire system
    attack_periods_all = list(
        zip(filtered_label_df["Start Time"], filtered_label_df["End Time"])
    )
    anomaly_array_all = np.array(
        [
            is_in_attack_period(ts, attack_periods_all)
            for ts in attack_df.index[seq_len:]
        ]
    )
    subsystem_anomaly_arrays["label_full"] = anomaly_array_all

    # For each unique subsystem, filter relevant rows and generate anomaly label array
    for subsystem in all_subsystems:
        relevant_rows = filtered_label_df[
            filtered_label_df["subsystem"].apply(lambda x: subsystem in x)
        ]
        attack_periods = list(
            zip(relevant_rows["Start Time"], relevant_rows["End Time"])
        )
        anomaly_array = np.array(
            [
                is_in_attack_period(ts, attack_periods)
                for ts in attack_df.index[seq_len:]
            ]
        )
        subsystem_anomaly_arrays[f"label_comp_{subsystem}"] = anomaly_array

    # Convert dictionary to DataFrame
    anomaly_ts = pd.DataFrame(subsystem_anomaly_arrays, index=attack_df.index[seq_len:])
    anomaly_ts.to_parquet(anomaly_time_series.path)


@dsl.container_component
def compute_residuals(
    df: Input[Dataset],
    model_path: str,
    model_name: str,
    data_module_name: str,
    config_path: str,
    seq_len: int,
    inference_batch_size: int,
    residual_df: Output[Dataset],
):
    return dsl.ContainerSpec(
        image="hsteude/diag-driven-ad-models:v60",
        command=["python", "main.py"],
        args=[
            "compute-residuals",
            "--df-path",
            df.path,
            "--model-path",
            model_path,
            "--data-module-name",
            data_module_name,
            "--model-name",
            model_name,
            "--config-path",
            config_path,
            "--seq-len",
            seq_len,
            "--result-df-path",
            residual_df.path,
            "--inference-batch-size",
            inference_batch_size,
        ],
    )


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "openpyxl",
    ],
    base_image="python:3.9",
)
def basic_cleanup_time_series_data(
    normal_data_in: Input[Dataset],
    attack_data_in: Input[Dataset],
    normal_data_out: Output[Dataset],
    attack_data_out: Output[Dataset],
):
    import pandas as pd

    df_normal_v1, df_attack_v0 = [
        pd.read_excel(path, header=1)
        for path in (normal_data_in.path, attack_data_in.path)
    ]

    # convert to pandas time series
    for df in [df_normal_v1, df_attack_v0]:
        df["Timestamp"] = pd.to_datetime(df[" Timestamp"])
    df_normal_v1 = df_normal_v1.set_index("Timestamp", drop=True)
    df_attack_v0 = df_attack_v0.set_index("Timestamp", drop=True)
    df_normal_v1 = df_normal_v1.drop(" Timestamp", axis=1)
    df_attack_v0 = df_attack_v0.drop(" Timestamp", axis=1)

    # fix column names (some begin with white spaces)
    df_normal_v1.columns = [s.replace(" ", "") for s in df_normal_v1.columns]
    df_attack_v0.columns = [s.replace(" ", "") for s in df_attack_v0.columns]

    df_normal_v1.to_parquet(normal_data_out.path)
    df_attack_v0.to_parquet(attack_data_out.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "openpyxl",
    ],
    base_image="python:3.9",
)
def basic_cleanup_label_data(
    label_data_in: Input[Dataset],
    label_data_out: Output[Dataset],
):
    from datetime import datetime
    import pandas as pd
    import re

    # read raw labels files
    df_label = pd.read_excel(label_data_in.path)

    # filter labels df to the attack that have a end date attached
    # transofrm end time to full timestmap
    df_label_time = df_label[df_label["End Time"].notna()].copy()
    df_label_time.loc[:, "End Time"] = [
        datetime.combine(datetime.date(a), b)
        for a, b in zip(df_label_time["Start Time"], df_label_time["End Time"])
    ]
    df_label_time = df_label_time.reset_index(drop=True)

    # ok, lets remove everything smaller than min_date and larger than max datefrom
    # the attacks and labels
    # See EDA notebook for why we do so!
    SWAT_MIN_DATE = datetime(2015, 12, 22)
    SWAT_MAX_DATE = datetime(2016, 1, 2)
    df_label_time = df_label_time[
        (df_label_time["Start Time"] > SWAT_MIN_DATE)
        & (df_label_time["Start Time"] < SWAT_MAX_DATE)
    ]

    def extract_first_digits_of_numbers(s):
        numbers = re.findall(r"\d+", s)
        if numbers:
            unique_first_digits = list(
                set([num[0] for num in numbers])
            )  # Nur eindeutige erste Ziffern
            return [
                int(digit) for digit in unique_first_digits
            ]  # Konvertiere die Ziffern in Integers
        return None  # Oder eine leere Liste, wenn keine Zahlen gefunden werden

    df_label_time["subsystem"] = df_label_time["Attack Point"].apply(
        extract_first_digits_of_numbers
    )

    df_label_time.to_parquet(label_data_out.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "scikit-learn",
    ],
    base_image="python:3.9",
)
def fit_scaler(
    in_df: Input[Dataset],
    scaler: Output[Artifact],
    cols: List[str],
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from joblib import dump
    import random

    df_train = pd.read_parquet(in_df.path)
    df_train = df_train[cols]
    scaler_obj = StandardScaler().fit(df_train.values)
    dump(scaler_obj, scaler.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "scikit-learn",
    ],
    base_image="python:3.9",
)
def scale_data(
    df_in: Input[Dataset],
    scaler: Input[Artifact],
    cols: List[str],
    df_out: Output[Dataset],
):
    import pandas as pd
    from joblib import load

    df = pd.read_parquet(df_in.path)
    df = df[cols]

    scaler_obj = load(scaler.path)
    df_sc = pd.DataFrame(
        scaler_obj.fit_transform(df.values),
        columns=df.columns,
        index=df.index,
    )

    df_sc.to_parquet(df_out.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "scikit-learn",
    ],
    base_image="python:3.9",
)
def split_data(
    normal_df_in: Input[Dataset],
    train_df_out: Output[Dataset],
    val_df_out: Output[Dataset],
):
    import pandas as pd
    from joblib import load

    df_normal = pd.read_parquet(normal_df_in.path)
    df_train = df_normal["2015-12-22 16:30:00":"2015-12-27 23:59:59"]
    df_val = df_normal["2015-12-28 00:00:00":"2015-12-28 09:59:55"]

    df_train.to_parquet(train_df_out.path)
    df_val.to_parquet(val_df_out.path)
