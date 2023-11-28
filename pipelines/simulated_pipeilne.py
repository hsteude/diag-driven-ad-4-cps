# imports
from kfp import dsl
from kfp.client import Client
import os
from simulated_components import (
    generate_data,
    plot_causal_factors,
    plot_signals,
    split_data,
    compute_simulated_residuals,
    generate_anomaly_dfs,
    train_gmms,
    analyse_results,
)
from common_components import run_pytorch_training_job, run_hyper_parameter_tuning
import toml
from typing import Dict


@dsl.pipeline
def diag_tcn_simulated_pipeline(
    subsystem_map: Dict,
    seq_len: int = 500,
    early_stopping_patience: int = 50,
    max_epochs: int = 100,
    seed: int = 42,
):
    generate_healthy_data_task = generate_data(
        min_lenght_causal_phase=500,
        max_lenght_causal_phase=1000,
        num_ber_phases=500,
        component_b_lag=200,
        zeta=0.3,
        du=1.0,
        taup=50.0,
        tau=20.0,
        seed=seed,
    )

    plot_causal_factors_task = plot_causal_factors(
        input_data=generate_healthy_data_task.outputs["df_out"],
    )

    plot_healthy_signals_task = plot_signals(
        input_data=generate_healthy_data_task.outputs["df_out"],
        dataset_name="Healthy/normal data",
    )

    split_data_task = split_data(
        simulate_data_normal=generate_healthy_data_task.outputs["df_out"]
    )

    run_vanilla_hparam_tuning_task = run_hyper_parameter_tuning(
        df_train=split_data_task.outputs["train_data"],
        df_val=split_data_task.outputs["val_data"],
        experiment_name="vanilla-simulated",
        image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        seed=seed,
        max_epochs=100,
        model_name="vanilla-tcn-vae",
        max_trials=35,
        batch_size_list=["64", "128", "256"],
        beta_list=["0.0001", "0.00005", "0.00001"],
        learning_rate_list=["0.0005", "0.001", "0.005"],
        kernel_size_list=["5", "10", "15"],
        latent_dim=12,
    )

    run_multi_latent_hparam_tuning_task = run_hyper_parameter_tuning(
        df_train=split_data_task.outputs["train_data"],
        df_val=split_data_task.outputs["val_data"],
        experiment_name="multi-latent-simulated",
        image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        seed=seed,
        max_epochs=100,
        model_name="multi-latent-tcn-vae",
        max_trials=35,
        batch_size_list=["64", "128", "256"],
        beta_list=["0.0001", "0.00005", "0.00001"],
        learning_rate_list=["0.0005", "0.001", "0.005"],
        kernel_size_list=["5", "10", "15"],
        latent_dim=6,
    )

    run_univar_tuning_task = run_hyper_parameter_tuning(
        df_train=split_data_task.outputs["train_data"],
        df_val=split_data_task.outputs["val_data"],
        experiment_name="univar-simulated",
        image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        seed=seed,
        max_epochs=100,
        model_name="combined-univariate-tcn-vae",
        max_trials=35,
        batch_size_list=["64", "128", "256"],
        beta_list=["0.0001", "0.00005", "0.00001"],
        learning_rate_list=["0.0005", "0.001", "0.005"],
        kernel_size_list=["5", "10", "15"],
        latent_dim=2,
    )

    train_vanilla_tcn_vae_model_task = run_pytorch_training_job(
        train_df_in=split_data_task.outputs["train_data"],
        val_df_in=split_data_task.outputs["val_data"],
        model_name="vanilla-tcn-vae",
        data_module_name="simulated",
        config_path="./sim-all-config.toml",
        minio_model_bucket="hs-bucket",
        training_image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        num_dl_workers=12,
        number_trainng_samples=10_000,
        number_validation_samples=1_000,
        tuning_param_dct=run_vanilla_hparam_tuning_task.output,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        latent_dim=12,
        seq_len=seq_len,
        seed=seed,
        num_gpu_nodes=3,
    )

    train_multi_latent_tcn_vae_model_task = run_pytorch_training_job(
        train_df_in=split_data_task.outputs["train_data"],
        val_df_in=split_data_task.outputs["val_data"],
        model_name="multi-latent-tcn-vae",
        data_module_name="simulated",
        config_path="./sim-all-config.toml",
        minio_model_bucket="hs-bucket",
        training_image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        num_dl_workers=12,
        number_trainng_samples=10_000,
        number_validation_samples=1_000,
        tuning_param_dct=run_multi_latent_hparam_tuning_task.output,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        latent_dim=6,
        seq_len=seq_len,
        seed=seed,
        num_gpu_nodes=3,
    )

    train_combined_univar_tcn_vae_model_task = run_pytorch_training_job(
        train_df_in=split_data_task.outputs["train_data"],
        val_df_in=split_data_task.outputs["val_data"],
        model_name="combined-univariate-tcn-vae",
        data_module_name="simulated",
        config_path="./sim-all-config.toml",
        minio_model_bucket="hs-bucket",
        training_image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        num_dl_workers=12,
        number_trainng_samples=10_000,
        number_validation_samples=1_000,
        tuning_param_dct=run_univar_tuning_task.output,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        latent_dim=2,
        seq_len=seq_len,
        seed=seed,
        num_gpu_nodes=3,
    )

    train_gmms_task = train_gmms(
        data_path=split_data_task.outputs["train_data"],
        subsystems_map=subsystem_map,
        n_samples=10_000,
        n_components_full=6,
        n_components_a=3,
        n_components_b=3,
        random_state=seed,
    )

    generate_anomaly_dfs_task = generate_anomaly_dfs(
        healthy_df=split_data_task.outputs["test_data_normal"]
    )

    for fault_string in ("df_fault_1", "df_fault_2", "df_fault_3", "df_fault_4"):
        plot_faulty_signals_task = plot_signals(
            input_data=generate_anomaly_dfs_task.outputs[fault_string],
            dataset_name=f"{fault_string} data",
        )

    compute_residuals_task = (
        compute_simulated_residuals(
            vanilla_model_path=train_vanilla_tcn_vae_model_task.output,
            multi_latent_model_path=train_multi_latent_tcn_vae_model_task.output,
            univar_model_path=train_combined_univar_tcn_vae_model_task.output,
            gmm_model_a=train_gmms_task.outputs["model_a_out"],
            gmm_model_b=train_gmms_task.outputs["model_b_out"],
            gmm_model_full=train_gmms_task.outputs["model_full_out"],
            healthy_df=split_data_task.outputs["test_data_normal"],
            df_fault1=generate_anomaly_dfs_task.outputs["df_fault_1"],
            df_fault2=generate_anomaly_dfs_task.outputs["df_fault_2"],
            df_fault3=generate_anomaly_dfs_task.outputs["df_fault_3"],
            df_fault4=generate_anomaly_dfs_task.outputs["df_fault_4"],
        )
        .set_env_variable("AWS_SECRET_ACCESS_KEY", os.environ["AWS_SECRET_ACCESS_KEY"])
        .set_env_variable("AWS_ACCESS_KEY_ID", os.environ["AWS_ACCESS_KEY_ID"])
        .set_env_variable("S3_ENDPOINT", os.environ["S3_ENDPOINT"])
        .set_cpu_limit("20")
    )

    analyse_results_task = analyse_results(
        residuals_df_in=compute_residuals_task.outputs["residuals_df"]
    )


def run():
    # Load the subsystem map from the TOML file
    with open("../images/diag-driven-ad-models/sim-all-config.toml", "r") as f:
        config = toml.load(f)
    subsystem_map = config["SUBSYSTEM_MAP"]

    # compile and run pipeline
    client = Client()
    args = dict(
        seq_len=500,
        early_stopping_patience=50,
        max_epochs=1000,
        seed=42,
        subsystem_map=subsystem_map,
    )
    client.create_run_from_pipeline_func(
        diag_tcn_simulated_pipeline,
        arguments=args,
        experiment_name="diag_drive_ad_simulation",
        enable_caching=True,
    )


if __name__ == "__main__":
    run()
