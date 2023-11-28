from kfp import dsl
from kfp.dsl import HTML, Input, Output, Dataset, Artifact, Model
from typing import List, Dict


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "plotly",
        "loguru",
        "scikit-learn",
    ],
    base_image="python:3.9",
)
def analyse_results(
    residuals_df_in: Input[Dataset],
    distro_plot: Output[HTML],
    metrics_table: Output[HTML],
):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import (
        f1_score,
        roc_auc_score,
        roc_curve,
        precision_score,
        recall_score,
    )
    import numpy as np
    import pandas as pd

    df_residuals = pd.read_parquet(residuals_df_in.path)

    from sklearn.metrics import f1_score, roc_auc_score, roc_curve

    def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute F1 scores, ROC-AUC, and the optimal threshold based on F1 score
        optimization for each model and metric.

        Parameters:
        - df: DataFrame containing labels and predictions.

        Returns:
        - metrics_df: DataFrame with metrics for each model and metric.
        """
        subsystem_string = ["a", "b", "full"]
        results = []

        for model in df["model"].unique():
            for subsystem in subsystem_string:
                labels = df[df["model"] == model][f"label_{subsystem}"]
                scores = df[df["model"] == model][f"mse_{subsystem}"]

                auc_roc = roc_auc_score(labels, scores)

                _, _, thresholds = roc_curve(labels, scores)
                # Calculate F1 score for each threshold
                f1_scores = [
                    f1_score(labels, scores > threshold) for threshold in thresholds
                ]
                # Find the optimal threshold (max F1 score)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]

                # Make predictions using the optimal threshold
                predictions = scores > optimal_threshold
                f1 = f1_score(labels, predictions)
                precision = precision_score(labels, predictions)
                recall = recall_score(labels, predictions)

                results.append(
                    {
                        "model": model,
                        "subsystem": subsystem,
                        "f1_score": f1,
                        "roc_auc": auc_roc,
                        "optimal_threshold": optimal_threshold,
                        "precision": precision,
                        "recall": recall,
                    }
                )

        metrics_df = pd.DataFrame(results)
        return metrics_df

    def create_plotly_table(df):
        """
        Create a plotly table from a dataframe and round the values to two decimal places.

        Args:
        - df (pd.DataFrame): The dataframe to convert to a plotly table.

        Returns:
        - plotly.graph_objects.Figure: The plotly table figure.
        """

        # Round numerical columns to two decimal places
        for col in df.columns:
            if df[col].dtype in ["float64", "float32"]:
                df[col] = df[col].round(3)

        table = go.Figure(
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        )

        return table

    metrics_df = compute_metrics(df_residuals)
    metrics_df = metrics_df.sort_values("subsystem")
    table = create_plotly_table(metrics_df)
    table.write_html(metrics_table.path)

    ## box_plot
    FONT_SIZE = 10

    model_order = [
        "gmm",
        "combined-univariate-tcn-vae",
        "vanilla-tcn-vae",
        "multi-latent-tcn-vae",
    ]
    model_names = {
        "gmm": "GMM",
        "combined-univariate-tcn-vae": "univariate-tcn-vae",
        "vanilla-tcn-vae": "vanilla-tcn-vae",
        "multi-latent-tcn-vae": "our-model",
    }

    import plotly.express as px

    color_scale = px.colors.sequential.Viridis
    num_categories = 5
    colors = [
        color_scale[i * len(color_scale) // num_categories]
        for i in range(num_categories)
    ]

    dataset_colors = {
        "healthy": colors[0],
        "fault1": colors[1],
        "fault2": colors[2],
        "fault3": colors[3],
        "fault4": colors[4],
    }

    fig = make_subplots(
        rows=len(model_order),
        cols=3,
        vertical_spacing=0.03,
        horizontal_spacing=0.04,
        subplot_titles=["subsystem A", "subsystem B", "all signals"],
    )

    for idx, model in enumerate(model_order, start=1):
        subset = df_residuals[df_residuals["model"] == model]
        for j, mse_type in enumerate(["mse_a", "mse_b", "mse_full"], start=1):
            for dataset in subset["dataset"].unique():
                data_for_dataset = subset[subset["dataset"] == dataset][mse_type]

                show_legend = idx == 1 and j == 1

                fig.add_trace(
                    go.Box(
                        y=data_for_dataset,
                        name=dataset,
                        showlegend=show_legend,
                        line=dict(color=dataset_colors[dataset]),
                        boxpoints=False,
                    ),
                    row=idx,
                    col=j,
                )

            if idx == len(model_order):
                fig.update_xaxes(
                    title_text="scenario",
                    row=idx,
                    col=j,
                    titlefont=dict(size=FONT_SIZE, family="Times New Roman"),
                    tickfont=dict(size=FONT_SIZE, family="Times New Roman"),
                    title_standoff=0,
                )
            else:
                fig.update_xaxes(showticklabels=False, row=idx, col=j)

            fig.update_yaxes(
                title_text=f"{model_names[model]} score",
                row=idx,
                col=1,
                titlefont=dict(size=FONT_SIZE, family="Times New Roman"),
                tickfont=dict(size=FONT_SIZE, family="Times New Roman"),
                title_standoff=0,
            )

    # Adjust subplot title font size
    for annotation in fig["layout"]["annotations"]:
        annotation["font"]["size"] = FONT_SIZE
        annotation["font"]["family"] = "Times New Roman"

    fig.update_layout(
        # height=150 * len(model_order),
        # width=600,
        font=dict(size=FONT_SIZE, family="Times New Roman"),
        # legend_title_text='Scenario',
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=-0.15,
        legend_x=0.2,
    )

    fig.write_html(distro_plot.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "loguru",
        "scikit-learn",
    ],
    base_image="python:3.9",
)
def train_gmms(
    data_path: Input[Dataset],
    model_a_out: Output[Model],
    model_b_out: Output[Model],
    model_full_out: Output[Model],
    subsystems_map: dict,
    n_samples: int,
    n_components_full: int = 6,
    n_components_a: int = 3,
    n_components_b: int = 3,
    random_state: int = 42,
) -> None:
    """
    Train Gaussian Mixture Models (GMMs) on subsets of data from a specified Parquet file and save them.

    Parameters:
    - data_path: Path to the Parquet file containing the data.
    - mode_a_out: Path where the trained GMM for component 'a' will be saved.
    - model_b_out: Path where the trained GMM for component 'b' will be saved.
    - mode_full_out: Path where the trained GMM for both components will be saved.
    - subsystems_map: Dictionary mapping component names to column lists.
    - n_samples: Number of samples to draw from the dataframe for training.
    - n_components_full: Number of mixture components for the GMM for all signals.
    - n_components_a: Number of mixture components for the GMM for subsystem A.
    - n_components_b: Number of mixture components for the GMM for susbsystem B.
    - random_state: Random seed for the GMM initialization and sampling. Default is 42.

    Returns:
    None. The trained GMMs are saved to the specified paths.
    """
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    from joblib import dump

    # Read the data from the Parquet file
    df = pd.read_parquet(data_path.path)

    for subsystem, path, n_components in zip(
        ["a", "b", "full"],
        [model_a_out, model_b_out, model_full_out],
        [n_components_a, n_components_b, n_components_full],
    ):
        # Decide on columns based on component
        if subsystem == "a":
            columns = subsystems_map["a"]
        elif subsystem == "b":
            columns = subsystems_map["b"]
        elif subsystem == "full":
            columns = subsystems_map["a"] + subsystems_map["b"]
        else:
            raise ValueError("Component should be 'a', 'b', or 'full'.")

        # Sample the data
        df_sampled = df[columns].sample(n=n_samples, random_state=random_state)

        # Extract samples
        samples = df_sampled.values

        # Train the GMM
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(samples)

        # Save the trained GMM model
        dump(gmm, path.path)


@dsl.container_component
def compute_simulated_residuals(
    vanilla_model_path: str,
    multi_latent_model_path: str,
    univar_model_path: str,
    gmm_model_a: Input[Model],
    gmm_model_b: Input[Model],
    gmm_model_full: Input[Model],
    healthy_df: Input[Dataset],
    df_fault1: Input[Dataset],
    df_fault2: Input[Dataset],
    df_fault3: Input[Dataset],
    df_fault4: Input[Dataset],
    residuals_df: Output[Dataset],
):
    return dsl.ContainerSpec(
        image="hsteude/diag-driven-ad-models:v76",
        command=["python", "main.py"],
        args=[
            "compute-simulated-residuals",
            "--vanilla-model-path",
            vanilla_model_path,
            "--multi-latent-model-path",
            multi_latent_model_path,
            "--univar-model-path",
            univar_model_path,
            "--gmm-model-a-path",
            gmm_model_a.path,
            "--gmm-model-b-path",
            gmm_model_b.path,
            "--gmm-model-full-path",
            gmm_model_full.path,
            "--healthy-df-path",
            healthy_df.path,
            "--df-fault1-path",
            df_fault1.path,
            "--df-fault2-path",
            df_fault2.path,
            "--df-fault3-path",
            df_fault3.path,
            "--df-fault4-path",
            df_fault4.path,
            "--residuals-df-path",
            residuals_df.path,
        ],
    )


@dsl.container_component
def generate_data(
    min_lenght_causal_phase: int,
    max_lenght_causal_phase: int,
    component_b_lag: int,
    num_ber_phases: int,
    zeta: float,
    du: float,
    taup: float,
    tau: float,
    seed: int,
    df_out: Output[Dataset],
):
    return dsl.ContainerSpec(
        image="hsteude/diag-driven-ad-models:v67",
        command=["python", "main.py"],
        args=[
            "generate_data",
            "--min-lenght-causal-phase",
            min_lenght_causal_phase,
            "--max-lenght-causal-phase",
            max_lenght_causal_phase,
            "--num-ber-phases",
            num_ber_phases,
            "--component-b-lag",
            component_b_lag,
            "--zeta",
            zeta,
            "--du",
            du,
            "--taup",
            taup,
            "--tau",
            tau,
            "--seed",
            seed,
            "--df-healthy-path",
            df_out.path,
        ],
    )


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow",
        "scipy",
    ],
    base_image="python:3.9",
)
def generate_anomaly_dfs(
    healthy_df: Input[Dataset],
    df_fault_1: Output[Dataset],
    df_fault_2: Output[Dataset],
    df_fault_3: Output[Dataset],
    df_fault_4: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    from scipy import interpolate

    df_healthy = pd.read_parquet(healthy_df.path)

    # fault 1: set a1 to a rediculus constant value
    SIG_A1_CONST = -1
    df_fault_1_obj = df_healthy.copy()
    df_fault_1_obj["sig_a1"] = SIG_A1_CONST
    df_fault_1_obj.to_parquet(df_fault_1.path)

    # fault 2: shift sig_b3
    df_fault_2_obj = df_healthy.copy()
    df_fault_2_obj["sig_b3"] = df_fault_2_obj["sig_b3"] + 1
    df_fault_2_obj.to_parquet(df_fault_2.path)

    # fault 3
    SUBSYS_B_LAG = 150
    df_fault_3_obj = df_healthy.copy()
    for column in df_fault_3_obj.columns:
        if "sig_b" in column:
            df_fault_3_obj[column] = df_fault_3_obj[column].shift(SUBSYS_B_LAG)
    df_fault_3_obj.dropna(inplace=True)
    df_fault_3_obj.to_parquet(df_fault_3.path)

    # fault 4: Speed up signals by taking every second row
    df_fault_4_obj = df_healthy.iloc[::2].reset_index(drop=True)
    df_fault_4_obj.to_parquet(df_fault_4.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "s3fs",
        "pyarrow",
        "signals",
        "plotly",
        "scipy",
        "loguru",
    ],
    base_image="python:3.9",
)
def plot_causal_factors(
    input_data: Input[Dataset],
    causal_factors_plot: Output[HTML],
):
    import plotly.graph_objects as go
    import pandas as pd

    df = pd.read_parquet(input_data.path)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(y=df.comp_a_signal[0:15000], mode="lines", name="comp_a_signal")
    )
    fig.add_trace(
        go.Scatter(y=df.comp_b_signal[0:15000], mode="lines", name="comp_b_signal")
    )
    fig.update_layout(title_text="Underlying factor of change for subsystem a and b")
    fig.write_html(causal_factors_plot.path)


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "s3fs",
        "pyarrow",
        "signals",
        "plotly",
        "scipy",
        "loguru",
    ],
    base_image="python:3.9",
)
def plot_signals(
    input_data: Input[Dataset], signals_plot: Output[HTML], dataset_name: str
):
    import plotly.subplots as sp
    import plotly.graph_objects as go
    import pandas as pd

    START_IDX = 0
    END_IDX = 3000

    df = pd.read_parquet(input_data.path)

    fig = sp.make_subplots(rows=2, cols=1)

    # Add traces
    fig.add_trace(
        go.Scatter(y=df.sig_a1[START_IDX:END_IDX], mode="lines", name="sig_a1"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=df.sig_a2[START_IDX:END_IDX], mode="lines", name="sig_a2"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=df.sig_a3[START_IDX:END_IDX], mode="lines", name="sig_a3"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=df.sig_b1[START_IDX:END_IDX], mode="lines", name="sig_b1"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=df.sig_b2[START_IDX:END_IDX], mode="lines", name="sig_b2"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=df.sig_b3[START_IDX:END_IDX], mode="lines", name="sig_b3"),
        row=2,
        col=1,
    )
    fig.update_layout(
        title_text=f"{dataset_name}: Random area of the generated  signals"
    )

    fig.write_html(signals_plot.path)


@dsl.component(
    packages_to_install=["pandas==1.5.3", "pyarrow"],
    base_image="python:3.9",
)
def split_data(
    simulate_data_normal: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data_normal: Output[Dataset],
):
    import pandas as pd

    # read data
    df_data = pd.read_parquet(simulate_data_normal.path)

    total_length = len(df_data)
    start_idx_val = int(0.8 * total_length)
    start_idx_test = int(0.9 * total_length)

    df_train = df_data[:start_idx_val]
    df_val = df_data[start_idx_val:start_idx_test]
    df_test = df_data[start_idx_test:]

    df_train.to_parquet(train_data.path)
    df_val.to_parquet(val_data.path)
    df_test.to_parquet(test_data_normal.path)


@dsl.component(
    packages_to_install=["pandas==1.5.3", "pyarrow"],
    base_image="python:3.9",
)
def compute_metrics(
    residual_df_healthy: Input[Dataset],
    residual_df_sig_a2: Input[Dataset],
    residual_df_sig_b3: Input[Dataset],
    residual_df_a2b: Input[Dataset],
    residual_df_b2a: Input[Dataset],
    combined_residual_df: Output[Dataset],
):
    import pandas as pd

    df_healthy = pd.read_parquet(residual_df_healthy.path)
    df_sig_a2 = pd.read_parquet(residual_df_sig_a2.path)
    df_sig_b3 = pd.read_parquet(residual_df_sig_b3.path)
    df_sig_a2b = pd.read_parquet(residual_df_a2b.path)
    df_sig_b2a = pd.read_parquet(residual_df_b2a.path)

    df_healthy["source"] = "healthy"
    df_sig_a2["source"] = "sig_a2"
    df_sig_b3["source"] = "sig_b3"
    df_sig_a2b["source"] = "sig_a2b"
    df_sig_b2a["source"] = "sig_b2a"

    df_combined = pd.concat(
        [df_healthy, df_sig_a2, df_sig_b3, df_sig_a2b, df_sig_b2a],
        axis=0,
    )
    df_combined.to_parquet(combined_residual_df.path)
