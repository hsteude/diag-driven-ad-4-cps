import click
import toml
import ast
from loguru import logger
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from diag_driven_ad_models.multi_latent_tcn_vae import MultiLatentTcnVAE
from diag_driven_ad_models.vanilla_tcn_vae import VanillaTcnVAE
from diag_driven_ad_models.combined_univariate_tcn_vae import CombinedUnivariateTcnVae
from typing import Optional, Tuple
from loguru import logger
from torchinfo import summary
import copy
from pytorch_lightning import seed_everything
import numpy as np
from diag_driven_ad_models.utils import (
    read_df,
    upload_file_to_minio_bucket,
    create_s3_client,
)
from data_modules.swat_dm import SwatDataModule, SwatDataset
from data_modules.simulated_data_dm import SimDataModule, SimDataSet
from diag_driven_ad_models.callbacks import StdOutLoggerCallback
from pytorch_lightning.plugins.environments import KubeflowEnvironment
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
from data_modules.simulated_data_generator import SimulatedDataGenerator
from diag_driven_ad_models.gmm_wrapper import GMMWrapper


@click.group()
def cli():
    pass


@cli.command("generate_data")
@click.option("--min-lenght-causal-phase", default=500, type=int)
@click.option("--max-lenght-causal-phase", default=1000, type=int)
@click.option("--num-ber-phases", default=500, type=int, help="Number of phases.")
@click.option("--zeta", default=0.3, type=float, help="Zeta value.")
@click.option("--du", default=1.0, type=float, help="Du value.")
@click.option("--taup", default=50, type=float, help="Taup value.")
@click.option("--tau", default=20.0, type=float, help="Tau value.")
@click.option("--df-healthy-path", type=str)
@click.option("--component-b-lag", type=int)
@click.option("--seed", default=42, type=int, help="Seed value.")
def generate_anomaly_dfs(
    df_healthy_path: str,
    min_lenght_causal_phase: int = 500,
    max_lenght_causal_phase: int = 1000,
    num_ber_phases: int = 500,
    component_b_lag: int = 200,
    zeta: float = 0.3,
    du: float = 1.0,
    taup: float = 50,
    tau: float = 20.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Uses the SimulatedDataGenerator to generate data and stores it as parquet


    Please see the docstring of the SimulatedDataGenerator class for details.
    """
    sdg = SimulatedDataGenerator(
        min_lenght_causal_phase=min_lenght_causal_phase,
        max_lenght_causal_phase=max_lenght_causal_phase,
        component_b_lag=component_b_lag,
        number_phases=num_ber_phases,
        zeta=zeta,
        du=du,
        taup=taup,
        tau=tau,
        seed=seed,
    )
    df = sdg.run()
    df.to_parquet(df_healthy_path)


@cli.command("compute-simulated-residuals")
@click.option("--vanilla-model-path", type=str, required=True)
@click.option("--multi-latent-model-path", type=str, required=True)
@click.option("--univar-model-path", type=str, required=True)
@click.option("--gmm-model-a-path", type=str, required=True)
@click.option("--gmm-model-b-path", type=str, required=True)
@click.option("--gmm-model-full-path", type=str, required=True)
@click.option("--healthy-df-path", type=str, required=True)
@click.option("--df-fault1-path", type=str, required=True)
@click.option("--df-fault2-path", type=str, required=True)
@click.option("--df-fault3-path", type=str, required=True)
@click.option("--df-fault4-path", type=str, required=True)
@click.option("--residuals-df-path", type=str, required=True)
def compute_simulated_residuals(
    vanilla_model_path: str,
    multi_latent_model_path: str,
    univar_model_path: str,
    gmm_model_a_path: str,
    gmm_model_b_path: str,
    gmm_model_full_path: str,
    healthy_df_path: str,
    df_fault1_path: str,
    df_fault2_path: str,
    df_fault3_path: str,
    df_fault4_path: str,
    residuals_df_path: str,
):
    """
    Computes residuals for simulated data using various anomaly detection models.

    Parameters:
    - vanilla_model_path: Path to the Vanilla TCN-VAE model's weights file.
    - multi_latent_model_path: Path to the Multi Latent TCN-VAE model's weights file.
    - univar_model_path: Path to the Combined Univariate TCN-VAE model's weights file.
    - gmm_model_a_path: Path to the GMM model for subsystem A's weights file.
    - gmm_model_b_path: Path to the GMM model for subsystem B's weights file.
    - gmm_model_full_path: Path to the GMM model for the full system's weights file.
    - healthy_df_path: Path to the dataset file for the healthy state.
    - df_fault1_path: Path to the dataset file for fault scenario 1.
    - df_fault2_path: Path to the dataset file for fault scenario 2.
    - df_fault3_path: Path to the dataset file for fault scenario 3.
    - df_fault4_path: Path to the dataset file for fault scenario 4.
    - residuals_df_path: Path where the computed residuals dataframe will be saved.

    Returns:
    - None: Saves the computed residuals in a DataFrame at the specified path.
    """
    # load dfs
    df_healthy, df_fault_1, df_fault_2, df_fault3, df_fault4 = [
        pd.read_parquet(path)
        for path in (
            healthy_df_path,
            df_fault1_path,
            df_fault2_path,
            df_fault3_path,
            df_fault4_path,
        )
    ]

    # copy model to local disk
    s3 = create_s3_client()
    for model_path in (vanilla_model_path, multi_latent_model_path, univar_model_path):
        s3.get(model_path, model_path)

    # load models
    vanilla_model = VanillaTcnVAE.load_from_checkpoint(
        checkpoint_path=vanilla_model_path
    )
    multi_model = MultiLatentTcnVAE.load_from_checkpoint(
        checkpoint_path=multi_latent_model_path
    )
    univar_model = CombinedUnivariateTcnVae.load_from_checkpoint(
        checkpoint_path=univar_model_path
    )

    gmm_model = GMMWrapper(
        gmm_model_a_path=gmm_model_a_path,
        gmm_model_b_path=gmm_model_b_path,
        gmm_model_full_path=gmm_model_full_path,
    )

    # load_config
    config = toml.load("./sim-all-config.toml")

    def get_residuals_df(
        model=vanilla_model,
        df=df_healthy,
        model_name="vanilla-tcn-vae",
        df_name="sig_a2",
        num_samples=100,
    ):
        dataset = SimDataSet(
            dataframe=df,
            subsystems_map=config["SUBSYSTEM_MAP"],
            input_cols=config["SENSOR_COLS"],
            number_of_samples=num_samples,
            seq_len=500,
            seed=42,
        )
        dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

        preds = []
        for batch in dataloader:
            with torch.no_grad():
                pred = model.predict(batch)
            preds.append(pred)

        res_dct = {}
        for i, key in enumerate(config["SUBSYSTEM_MAP"].keys()):
            res_dct[f"mse_{key.lower()}"] = (
                torch.cat([pred[0][i] for pred in preds]).cpu().numpy()
            )
        res_dct["mse_full"] = torch.cat([pred[1] for pred in preds]).cpu().numpy()
        res_df = pd.DataFrame(res_dct)
        res_df["model"] = model_name
        res_df["dataset"] = df_name
        return res_df

    models = [vanilla_model, multi_model, univar_model, gmm_model]
    model_names = [
        "vanilla-tcn-vae",
        "multi-latent-tcn-vae",
        "combined-univariate-tcn-vae",
        "gmm",
    ]
    dfs = [df_healthy, df_fault_1, df_fault_2, df_fault3, df_fault4]
    df_names = ["healthy", "fault1", "fault2", "fault3", "fault4"]
    num_samples_list = [400, 100, 100, 100, 100]
    res_dfs = []
    for model, model_name in zip(models, model_names):
        for df, df_name, num_samples in zip(dfs, df_names, num_samples_list):
            res_dfs.append(
                get_residuals_df(
                    model=model,
                    df=df,
                    model_name=model_name,
                    df_name=df_name,
                    num_samples=num_samples,
                )
            )
    res_df = pd.concat(res_dfs, axis=0).reset_index(drop=True)

    mapping = {
        "healthy": {"label_full": 0, "label_a": 0, "label_b": 0},
        "fault1": {"label_full": 1, "label_a": 1, "label_b": 0},
        "fault2": {"label_full": 1, "label_a": 0, "label_b": 1},
        "fault3": {"label_full": 1, "label_a": 0, "label_b": 0},
        "fault4": {"label_full": 1, "label_a": 1, "label_b": 1},
    }

    res_df["label_full"] = res_df["dataset"].apply(lambda x: mapping[x]["label_full"])
    res_df["label_a"] = res_df["dataset"].apply(lambda x: mapping[x]["label_a"])
    res_df["label_b"] = res_df["dataset"].apply(lambda x: mapping[x]["label_b"])
    res_df.to_parquet(residuals_df_path)


@cli.command("compute-residuals")
@click.option(
    "--df-path",
    type=str,
    help="Path to the local parquet file holding the attack (test) data",
)
@click.option(
    "--model-path",
    type=str,
    help="Path to the local pytorch checkpoint file",
)
@click.option("--model-name", type=str, help="Name of the model.")
@click.option("--config-path", type=str, help="Path to config file")
@click.option("--seq-len", type=int, default="500", help="length of the sequence")
@click.option(
    "--data-module-name",
    type=str,
    default="SWaT",
    help="Name of the data module. (SWaT or sim)",
)
@click.option("--result-df-path", type=str, help="Path to the test result df parquet")
@click.option("--inference-batch-size", type=int, help="batch size for the prediction")
def compute_residuals_swat(
    df_path: str,
    model_path: str,
    config_path: str,
    seq_len: int,
    model_name: str,
    data_module_name: str,
    result_df_path: str,
    inference_batch_size: int,
):
    """
    Computes residuals for anomaly detection based on a pre-trained model.

    Parameters:
    - df_path: Path to the dataset file used for computing residuals.
    - model_path: Path to the pre-trained model's weights file.
    - config_path: Path to the configuration file for model and dataset parameters.
    - seq_len: Sequence length to be used for processing the data.
    - model_name: Name of the model architecture to be used for residual computation.
    - data_module_name: Name of the data module indicating the type of dataset.
    - result_df_path: Path where the computed residuals dataframe will be saved.
    - inference_batch_size: Batch size to be used during the inference process.

    Returns:
    - None: Saves the computed residuals in a DataFrame at the specified path.
    """
    # copy model to local disk
    s3 = create_s3_client()
    s3.get(model_path, model_path)
    # load model
    if model_name == "vanilla-tcn-vae":
        logger.info(f"Loading weights for {model_name}")
        model = VanillaTcnVAE.load_from_checkpoint(checkpoint_path=model_path)
    elif model_name == "multi-latent-tcn-vae":
        logger.info(f"Loading weights for {model_name}")
        model = MultiLatentTcnVAE.load_from_checkpoint(checkpoint_path=model_path)
    elif model_name == "combined-univariate-tcn-vae":
        logger.info(f"Loading weights for {model_name}")
        model = CombinedUnivariateTcnVae.load_from_checkpoint(
            checkpoint_path=model_path
        )
    else:
        raise Exception("Unknown model name: " + model_name)

    # read dataframe
    config = toml.load(config_path)
    if data_module_name == "SWaT":
        df_attack_sc = pd.read_parquet(df_path)
        dataset = SwatDataset(
            cols=config["SENSOR_COLS"],
            symbols_dct=config["SUBSYSTEM_MAP"],
            df=df_attack_sc,
            seq_len_x=seq_len,
            num_samples=False,
            random_samples=False,
        )
    elif data_module_name == "simulated":
        dataset = SimDataSet(
            dataframe=pd.read_parquet(df_path),
            subsystems_map=config["SUBSYSTEM_MAP"],
            input_cols=config["SENSOR_COLS"],
            number_of_samples=100,
            seq_len=seq_len,
            seed=42,
        )
    else:
        raise Exception("Unknown data module name: " + data_module_name)

    dataloader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False)

    preds = []
    len_dl = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            pred = model.predict(batch)
        preds.append(pred)
        if batch_idx % 10 == 0:
            progress_percentage = (batch_idx / len_dl) * 100
            logger.info(f"Inference progress: ({progress_percentage:.2f}%)")

    res_dct = {}
    for i, key in enumerate(config["SUBSYSTEM_MAP"].keys()):
        res_dct[f"mse_{key.lower()}"] = (
            torch.cat([pred[0][i] for pred in preds]).cpu().numpy()
        )
    res_dct["mse_full"] = torch.cat([pred[1] for pred in preds]).cpu().numpy()
    res_df = pd.DataFrame(res_dct)
    res_df.to_parquet(result_df_path)


@cli.command("train")
@click.option("--df-train-path", type=str, required=True, help="Path to training data.")
@click.option("--df-val-path", type=str, required=True, help="Path to validation data.")
@click.option(
    "--model-output-file", type=str, required=True, help="Path to save the model."
)
@click.option(
    "--run-as-pytorchjob",
    default=True,
    type=bool,
    help="Whether to run as a PyTorchJob on Kubeflow.",
)
@click.option("--batch-size", type=int, required=True, help="Batch size.")
@click.option(
    "--num-workers", type=int, required=True, help="Number of data loader workers."
)
@click.option(
    "--number-of-train-samples",
    type=int,
    required=True,
    help="Number of training samples.",
)
@click.option(
    "--number-of-val-samples",
    type=int,
    required=True,
    help="Number of validation samples.",
)
@click.option(
    "--data-module-name",
    type=str,
    default="SWaT",
    help="Name of the data module. (SWaT or sim)",
)
@click.option("--kernel-size", type=int, default=15, help="Size of the kernel.")
@click.option("--max-epochs", type=int, default=3, help="Maximum number of epochs.")
@click.option("--learning-rate", type=float, default=1e-3, help="Learning rate.")
@click.option(
    "--early-stopping-patience",
    type=int,
    default=5,
    help="Patience for early stopping.",
)
@click.option("--latent-dim", type=int, default=5, help="Dimension of latent space.")
@click.option("--dropout", type=float, default=0.0, help="Dropout rate.")
@click.option(
    "--model-name", type=str, default="multi-latent-tcn-vae", help="Name of the model."
)
@click.option("--seq-len", type=int, default=500, help="Length of the sequence.")
@click.option("--beta", type=float, default=0.0, help="Beta parameter value.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option(
    "--config-path", type=str, default="swat-config.toml", help="Path to config file"
)
@click.option(
    "--minio-model-bucket",
    default=None,
    type=str,
    help="An minio bucket for the trained model.",
)
@click.option(
    "--num-gpu-nodes",
    default=1,
    type=str,
    help="Number of GPUs available for training. One required for training job",
)
def train(
    df_train_path: str,
    df_val_path: str,
    model_output_file: str,
    minio_model_bucket: Optional[str],
    batch_size: int,
    num_workers: int,
    number_of_train_samples: int,
    number_of_val_samples: int,
    data_module_name: str = "SWaT",
    kernel_size: int = 15,
    max_epochs: int = 3,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 5,
    latent_dim: int = 5,
    dropout: float = 0.0,
    run_as_pytorchjob: bool = False,
    model_name: str = "multi-latent-tcn-vae",
    seq_len: int = 500,
    beta: float = 0.0,
    config_path: str = "swat-config.toml",
    seed: int = 42,
    num_gpu_nodes: int = 1,
) -> str:
    """
    Trains a deep learning model for anomaly detection in Cyber-Physical Systems (CPS).

    This function handles the entire training pipeline, including data preparation,
    model initialization, training, and saving the trained model. It supports different
    neural network architectures and configurations, allowing for flexible experimentation.

    Args:
        df_train_path (str): Path to the training dataset.
        df_val_path (str): Path to the validation dataset.
        model_output_file (str): File path to save the trained model.
        minio_model_bucket (Optional[str]): Name of the MinIO bucket to store the model.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        number_of_train_samples (int): Number of training samples to use.
        number_of_val_samples (int): Number of validation samples to use.
        data_module_name (str): Name of the data module to use. Defaults to 'SWaT'.
        kernel_size (int): Kernel size for the convolutional layers. Defaults to 15.
        max_epochs (int): Maximum number of training epochs. Defaults to 3.
        learning_rate (float): Learning rate for the optimizer. Defaults to 1e-3.
        early_stopping_patience (int): Patience for early stopping. Defaults to 5.
        latent_dim (int): Dimensionality of the latent space. Defaults to 5.
        dropout (float): Dropout rate. Defaults to 0.0.
        run_as_pytorchjob (bool): Whether to run as a PyTorchJob in Kubeflow. Defaults to False.
        model_name (str): Name of the model to be trained.
        seq_len (int): Length of the input sequences. Defaults to 500.
        beta (float): Regularization coefficient for the loss function. Defaults to 0.0.
        config_path (str): Path to the configuration file. Defaults to 'swat-config.toml'.
        seed (int): Seed for random number generation. Defaults to 42.
        num_gpu_nodes (int): Number of GPU nodes to use. Defaults to 1.

    Returns:
        str: A string representation of the model summary after training.

    This function orchestrates the process of training a neural network model by setting up the necessary
    configurations, initiating the appropriate model based on the provided name, and executing the training
    process with early stopping and checkpointing strategies. It also handles optional tasks like uploading
    the trained model to a specified MinIO bucket and logging detailed information when running as a PyTorchJob.
    """
    # general training settings
    seed_everything(seed)
    np.random.seed(seed)
    logger.info(f"Random seed in training script set to {seed}")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")

    # ds specific settings from config file
    config = toml.load(config_path)
    input_cols = config["SENSOR_COLS"]
    subsystems_map = config["SUBSYSTEM_MAP"]

    # load dataset and initiate data module
    train_df = read_df(df_train_path)
    val_df = read_df(df_val_path)
    data_module = None
    if data_module_name == "SWaT":
        logger.info(f"Loading SWaT data module")
        data_module = SwatDataModule(
            df_train=train_df,
            df_val=val_df,
            cols=input_cols,
            symbols_dct=subsystems_map,
            batch_size=batch_size,
            num_train_samples=number_of_train_samples,
            num_val_samples=number_of_val_samples,
            seq_len=seq_len,
            dl_workers=num_workers,
            seed=seed,
        )
    elif data_module_name == "simulated":
        logger.info(f"Loading simulated data module")
        train_df = read_df(df_train_path)
        val_df = read_df(df_val_path)
        data_module = SimDataModule(
            df_train=train_df,
            df_val=val_df,
            input_cols=input_cols,
            subsystems_map=subsystems_map,
            batch_size=batch_size,
            number_of_train_samples=number_of_train_samples,
            number_of_val_samples=number_of_val_samples,
            seq_len=seq_len,
            num_workers=num_workers,
            seed=seed,
        )
    else:
        raise Exception("Unknown data module name: " + data_module_name)

    # initiate model
    if model_name in [
        "vanilla-tcn-vae",
    ]:
        # read in config params for model architecture
        channel_dims_key = "VANILLA_TCN_VAE_CHANNEL_DIMENSIONS"
        enc_tcn1_in_dims = config[channel_dims_key]["enc_tcn1_in_dims"]
        enc_tcn1_out_dims = config[channel_dims_key]["enc_tcn1_out_dims"]
        enc_tcn2_in_dims = config[channel_dims_key]["enc_tcn2_in_dims"]
        enc_tcn2_out_dims = config[channel_dims_key]["enc_tcn2_out_dims"]
        dec_tcn1_in_dims = config[channel_dims_key]["dec_tcn1_in_dims"]
        dec_tcn1_out_dims = config[channel_dims_key]["dec_tcn1_out_dims"]
        dec_tcn2_in_dims = config[channel_dims_key]["dec_tcn2_in_dims"]
        dec_tcn2_out_dims = config[channel_dims_key]["dec_tcn2_out_dims"]
        # modify in and out channels
        enc_tcn1_in_dims[0] = len(input_cols)
        dec_tcn2_out_dims[-1] = len(input_cols)

        logger.info(f"Initializing Vanilla TCN VAE model")
        model = VanillaTcnVAE(
            kernel_size=kernel_size,
            enc_tcn1_in_dims=enc_tcn1_in_dims,
            enc_tcn1_out_dims=enc_tcn1_out_dims,
            enc_tcn2_in_dims=enc_tcn2_in_dims,
            enc_tcn2_out_dims=enc_tcn2_out_dims,
            dec_tcn1_in_dims=dec_tcn1_in_dims,
            dec_tcn1_out_dims=dec_tcn1_out_dims,
            dec_tcn2_in_dims=dec_tcn2_in_dims,
            dec_tcn2_out_dims=dec_tcn2_out_dims,
            lr=learning_rate,
            seq_len=seq_len,
            component_output_dims=[
                len(subsystems_map[k]) for k in sorted(subsystems_map.keys())
            ],
            latent_dim=latent_dim,
            dropout=dropout,
            beta=beta,
        )
    elif model_name == "combined-univariate-tcn-vae":
        # read in config params for model architecture
        channel_dims_key = "COMBINED_UNIVARIATE_TCN_VAE_CHANNEL_DIMENSIONS"
        enc_tcn1_in_dims = config[channel_dims_key]["enc_tcn1_in_dims"]
        enc_tcn1_out_dims = config[channel_dims_key]["enc_tcn1_out_dims"]
        enc_tcn2_in_dims = config[channel_dims_key]["enc_tcn2_in_dims"]
        enc_tcn2_out_dims = config[channel_dims_key]["enc_tcn2_out_dims"]
        dec_tcn1_in_dims = config[channel_dims_key]["dec_tcn1_in_dims"]
        dec_tcn1_out_dims = config[channel_dims_key]["dec_tcn1_out_dims"]
        dec_tcn2_in_dims = config[channel_dims_key]["dec_tcn2_in_dims"]
        dec_tcn2_out_dims = config[channel_dims_key]["dec_tcn2_out_dims"]
        # modify in and out channels
        enc_tcn1_in_dims[0] = len(input_cols)
        dec_tcn2_out_dims[-1] = len(input_cols)

        logger.info(f"Initializing combined univariate tcn vae model")
        # modify in and out channels
        combi_enc_tcn1_in_dims = copy.copy(enc_tcn1_in_dims)
        combi_dec_tcn2_out_dims = copy.copy(dec_tcn2_out_dims)
        combi_enc_tcn1_in_dims[0] = 1
        combi_dec_tcn2_out_dims[-1] = 1
        model = CombinedUnivariateTcnVae(
            kernel_size=kernel_size,
            enc_tcn1_in_dims=combi_enc_tcn1_in_dims,
            enc_tcn1_out_dims=enc_tcn1_out_dims,
            enc_tcn2_in_dims=enc_tcn2_in_dims,
            enc_tcn2_out_dims=enc_tcn2_out_dims,
            dec_tcn1_in_dims=dec_tcn1_in_dims,
            dec_tcn1_out_dims=dec_tcn1_out_dims,
            dec_tcn2_in_dims=dec_tcn2_in_dims,
            dec_tcn2_out_dims=combi_dec_tcn2_out_dims,
            lr=learning_rate,
            seq_len=seq_len,
            component_output_dims=[
                len(subsystems_map[k]) for k in sorted(subsystems_map.keys())
            ],
            latent_dim=latent_dim,
            dropout=dropout,
            beta=beta,
        )
    elif model_name == "multi-latent-tcn-vae":
        # read in config params for model architecture
        channel_dims_key = "MULTI_LATENT_TCN_VAE_CHANNEL_DIMENSIONS"
        enc_tcn1_in_dims = config[channel_dims_key]["enc_tcn1_in_dims"]
        enc_tcn1_out_dims = config[channel_dims_key]["enc_tcn1_out_dims"]
        enc_tcn2_in_dims = config[channel_dims_key]["enc_tcn2_in_dims"]
        enc_tcn2_out_dims = config[channel_dims_key]["enc_tcn2_out_dims"]
        dec_tcn1_in_dims = config[channel_dims_key]["dec_tcn1_in_dims"]
        dec_tcn1_out_dims = config[channel_dims_key]["dec_tcn1_out_dims"]
        dec_tcn2_in_dims = config[channel_dims_key]["dec_tcn2_in_dims"]
        dec_tcn2_out_dims = config[channel_dims_key]["dec_tcn2_out_dims"]
        # modify in and out channels
        enc_tcn1_in_dims[0] = len(input_cols)
        dec_tcn2_out_dims[-1] = len(input_cols)
        logger.info(f"Initializing multi latent tcn vae model")
        model = MultiLatentTcnVAE(
            kernel_size=kernel_size,
            enc_tcn1_in_dims=enc_tcn1_in_dims,
            enc_tcn1_out_dims=enc_tcn1_out_dims,
            enc_tcn2_in_dims=enc_tcn2_in_dims,
            enc_tcn2_out_dims=enc_tcn2_out_dims,
            dec_tcn1_in_dims=dec_tcn1_in_dims,
            dec_tcn1_out_dims=dec_tcn1_out_dims,
            dec_tcn2_in_dims=dec_tcn2_in_dims,
            dec_tcn2_out_dims=dec_tcn2_out_dims,
            lr=learning_rate,
            seq_len=seq_len,
            component_output_dims=[
                len(subsystems_map[k]) for k in sorted(subsystems_map.keys())
            ],
            component_latent_dims=[latent_dim] * len(subsystems_map.keys()),
            total_latent_dim=sum([latent_dim] * len(subsystems_map.keys())),
            dropout=dropout,
            beta=beta,
        )
    else:
        raise Exception("Unknown model name: " + model_name)

    # early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        strict=True,
    )

    # saves top-K checkpoints based on "val_loss" metric
    os.makedirs("data", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="data/",
        filename=model_output_file,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        plugins=[KubeflowEnvironment()] if run_as_pytorchjob else [],
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            StdOutLoggerCallback(),
        ],
        num_nodes=num_gpu_nodes,
        strategy="ddp" if run_as_pytorchjob else "auto",
    )

    # Log relevant trainer attributes
    if run_as_pytorchjob:
        logger.debug(f"Trainer accelerator: {trainer.accelerator}")
        logger.debug(f"Trainer strategy: {trainer.strategy}")
        logger.debug(f"Trainer global_rank: {trainer.global_rank}")
        logger.debug(f"Trainer local_rank: {trainer.local_rank}")

    trainer.fit(model=model, datamodule=data_module)

    trainer.save_checkpoint(model_output_file)
    if minio_model_bucket:
        upload_file_to_minio_bucket(minio_model_bucket, model_output_file)
    return str(summary(model))


if __name__ == "__main__":
    cli()
