# imports
from kfp.components import load_component_from_file
from kfp import dsl
from kfp.client import Client
import constants as const
import os
from swat_components import (
    basic_cleanup_time_series_data,
    basic_cleanup_label_data,
    fit_scaler,
    scale_data,
    split_data,
    compute_residuals,
    format_labels_df_for_metric_computation,
    get_metrics,
    show_results,
)

from common_components import run_pytorch_training_job


# define pipeline
@dsl.pipeline
def diag_tcn_swat_pipeline(
    normal_data_path: str = "minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/Physical/SWaT_Dataset_Normal_v1.xlsx",
    attack_data_path: str = "minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/Physical/SWaT_Dataset_Attack_v0.xlsx",
    label_data_path: str = "minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/List_of_attacks_Final.xlsx",
    detrend_window: int = 10800,
    seed: int = 42,
    seq_len: int = 500,
    beta: float = 0.0001,
    early_stopping_patience: int = 30,
    max_epochs: int = 1000,
):
    import_normal_task = dsl.importer(
        artifact_uri=normal_data_path, artifact_class=dsl.Dataset
    )

    import_attack_task = dsl.importer(
        artifact_uri=attack_data_path, artifact_class=dsl.Dataset
    )

    import_labels_task = dsl.importer(
        artifact_uri=label_data_path, artifact_class=dsl.Dataset
    )

    basic_ts_cleanup_task = basic_cleanup_time_series_data(
        normal_data_in=import_normal_task.output,
        attack_data_in=import_attack_task.output,
    )

    basic_labels_cleanup_task = basic_cleanup_label_data(
        label_data_in=import_labels_task.output,
    )

    compute_labels_time_series_task = format_labels_df_for_metric_computation(
        attack_df_in=basic_ts_cleanup_task.outputs["attack_data_out"],
        labels_df=basic_labels_cleanup_task.outputs["label_data_out"],
        seq_len=seq_len,
    )

    fit_scaler_task = fit_scaler(
        in_df=basic_ts_cleanup_task.outputs["normal_data_out"],
        cols=const.SWAT_SENSOR_COLS,
    )

    scale_normal_data_task = scale_data(
        df_in=basic_ts_cleanup_task.outputs["normal_data_out"],
        scaler=fit_scaler_task.outputs["scaler"],
        cols=const.SWAT_SENSOR_COLS,
    )
    scale_attack_data_task = scale_data(
        df_in=basic_ts_cleanup_task.outputs["attack_data_out"],
        scaler=fit_scaler_task.outputs["scaler"],
        cols=const.SWAT_SENSOR_COLS,
    )
    split_data_task = split_data(normal_df_in=scale_normal_data_task.outputs["df_out"])

    train_multi_latent_tcn_vae_model_task = run_pytorch_training_job(
        train_df_in=split_data_task.outputs["train_df_out"],
        val_df_in=split_data_task.outputs["val_df_out"],
        model_name="multi-latent-tcn-vae",
        data_module_name="SWaT",
        config_path="./swat-config.toml",
        minio_model_bucket="hs-bucket",
        training_image="hsteude/diag-driven-ad-models:v76",
        namespace="henrik-steude",
        num_dl_workers=12,
        number_trainng_samples=10_000,
        number_validation_samples=1_000,
        tuning_param_dct=const.SWAT_HPARAMS,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        latent_dim=17,
        seq_len=seq_len,
        seed=seed,
        num_gpu_nodes=3,
    )

    compute_multi_latent_residuals_task = (
        compute_residuals(
            df=scale_attack_data_task.outputs["df_out"],
            model_path=train_multi_latent_tcn_vae_model_task.output,
            model_name="multi-latent-tcn-vae",
            data_module_name="SWaT",
            config_path="./swat-config.toml",
            seq_len=seq_len,
            inference_batch_size=512,
        )
        .set_env_variable("AWS_SECRET_ACCESS_KEY", os.environ["AWS_SECRET_ACCESS_KEY"])
        .set_env_variable("AWS_ACCESS_KEY_ID", os.environ["AWS_ACCESS_KEY_ID"])
        .set_env_variable("S3_ENDPOINT", os.environ["S3_ENDPOINT"])
        .set_cpu_limit("20")
    )

    compute_multi_latent_metrics_task = get_metrics(
        attack_residuals_df_in=compute_multi_latent_residuals_task.outputs[
            "residual_df"
        ],
        labels_time_series_in=compute_labels_time_series_task.outputs[
            "anomaly_time_series"
        ],
        labels_df_in=basic_labels_cleanup_task.outputs["label_data_out"],
        detrend_window=detrend_window,
    )


#    show_results_task = show_results(
#        metrics_dict_vanilla=compute_vanilla_metrics_task.output,
#        metrics_dict_multi_latent=compute_multi_latent_metrics_task.output,
#        metrics_dict_combined=compute_combined_univar_metrics_task.output,
#    )


# compile and run pipeline
client = Client()
args = dict(
    normal_data_path="minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/Physical/SWaT_Dataset_Normal_v1.xlsx",
    attack_data_path="minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/Physical/SWaT_Dataset_Attack_v0.xlsx",
    label_data_path="minio://swat-dataset/SWaT.A1 _ A2_Dec 2015/List_of_attacks_Final.xlsx",
    detrend_window=10800,
    seed=42,
    seq_len=500,
    early_stopping_patience=30,
)
client.create_run_from_pipeline_func(
    diag_tcn_swat_pipeline, arguments=args, experiment_name="diag_drive_ad_swat"
)
