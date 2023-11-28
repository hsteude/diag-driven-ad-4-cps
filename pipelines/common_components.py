from kfp import dsl
from kfp.dsl import Input, Dataset
from typing import Dict

@dsl.component(
    packages_to_install=["kubernetes", "loguru"],
    base_image="python:3.9",
)
def run_pytorch_training_job(
    train_df_in: Input[Dataset],
    val_df_in: Input[Dataset],
    model_name: str,
    data_module_name: str,
    config_path: str,
    minio_model_bucket: str,
    training_image: str,
    namespace: str,
    num_dl_workers: int,
    number_trainng_samples: int,
    number_validation_samples: int,
    tuning_param_dct: Dict,
    max_epochs: int,
    early_stopping_patience: int,
    latent_dim: int,
    seq_len: int,
    seed: int,
    num_gpu_nodes: int,
) -> str:
    """Initiates a PyTorch training job.

    Parameters:
    train_df_in: KFP input for the training dataframe.
    val_df_in: KFP input for the validation dataframe.
    model_name: Model identifier string (e.g., 'vanilla-tcn-vae', 'multi-latent-tcn-vae', or 'combined-univariate-tcn-vae').
    data_module_name: Data module identifier string ('SWat' or 'sim').
    config_path: Path to the config within the training image (e.g., './swat-config.toml' or './sim-all-config.toml').
    minio_model_bucket: Bucket to store trained models.
    training_image: Docker image containing training code.
    namespace: Kubernetes namespace to run the training job in.
    num_dl_workers: Number of data loader processes.
    number_trainng_samples: Number of samples to draw from the training dataframe.
    number_validation_samples: Number of samples to draw from the validation dataframe.
    batch_size: Batch size during training.
    learning_rate: Learning rate parameter.
    kernel_size: Kernel size of convolutional layers.
    beta: Beta parameter for ELBO loss.
    max_epochs: Maximum epochs to train.
    early_stopping_patience: Number of epochs to wait for improved validation loss before early stopping.
    latent_dim: Number of latent variables.
    seq_len: Length of sequences used for training.
    seed: Seed for random number generation to ensure reproducibility.
    num_gpu_nodes: Number of GPU nodes to utilize during training.

    Returns:
    A string message indicating the status of the PytorchJob
    """

    import time
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    from datetime import datetime
    from loguru import logger

    batch_size, learning_rate, kernel_size, beta = [
        tuning_param_dct[k]
        for k in ("batch_size", "learning_rate", "kernel_size", "beta")
    ]

    pytorchjob_name = f"{model_name}-{data_module_name.lower()}"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_file = f"{pytorchjob_name}_{current_time}"

    command = [
        "python",
        "main.py",
        "train",
        f"--df-train-path={train_df_in.path}",
        f"--df-val-path={val_df_in.path}",
        f"--model-output-file={model_output_file}",
        f"--minio-model-bucket={minio_model_bucket}",
        f"--num-workers={num_dl_workers}",
        f"--number-of-train-samples={number_trainng_samples}",
        f"--number-of-val-samples={number_validation_samples}",
        f"--data-module-name={data_module_name}",
        f"--batch-size={batch_size}",
        f"--learning-rate={learning_rate}",
        f"--kernel-size={kernel_size}",
        f"--beta={beta}",
        "--dropout=0.0",
        f"--max-epochs={max_epochs}",
        f"--early-stopping-patience={early_stopping_patience}",
        f"--latent-dim={latent_dim}",
        f"--model-name={model_name}",
        f"--seq-len={seq_len}",
        f"--seed={seed}",
        f"--config-path={config_path}",
        "--run-as-pytorchjob=True",
        f"--num-gpu-nodes={num_gpu_nodes}",
    ]

    template = {
        "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
        "spec": {
            "containers": [
                {
                    "name": "pytorch",
                    "image": training_image,
                    "imagePullPolicy": "Always",
                    "command": command,
                    "env": [
                        {
                            "name": "AWS_ACCESS_KEY_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "s3creds",
                                    "key": "AWS_ACCESS_KEY_ID",
                                }
                            },
                        },
                        {
                            "name": "AWS_SECRET_ACCESS_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "s3creds",
                                    "key": "AWS_SECRET_ACCESS_KEY",
                                }
                            },
                        },
                        {"name": "S3_ENDPOINT", "value": "minio.minio"},
                        {"name": "S3_USE_HTTPS", "value": "0"},
                        {"name": "S3_VERIFY_SSL", "value": "0"},
                    ],
                    "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                }
            ],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        },
    }

    training_job_manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": f"{pytorchjob_name}", "namespace": f"{namespace}"},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": template,
                },
                "Worker": {
                    "replicas": num_gpu_nodes - 1,
                    "restartPolicy": "OnFailure",
                    "template": template,
                },
            }
        },
    }

    # Kubernetes API clients
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api_client = client.ApiClient()
    custom_api = client.CustomObjectsApi(api_client)

    # Common arguments for API calls
    args_dict = {
        "group": "kubeflow.org",
        "version": "v1",
        "namespace": namespace,
        "plural": "pytorchjobs",
    }

    def start_pytorchjob():
        """Starts the PyTorchJob by calling create_namespaced_custom_object."""
        try:
            custom_api.create_namespaced_custom_object(
                body=training_job_manifest, **args_dict
            )
            print(f"Job {pytorchjob_name} started.")
        except ApiException as e:
            print(f"Error starting job: {e}")

    def get_pytorchjob_status():
        """Retrieves and returns the last status of the PyTorchJob."""
        try:
            job = custom_api.get_namespaced_custom_object(
                name=pytorchjob_name, **args_dict
            )
            return job.get("status", {}).get("conditions", [])[-1].get("type", "")
        except ApiException as e:
            print(f"Error retrieving job status: {e}")
            return None

    def delete_pytorchjob():
        """Deletes the PyTorchJob once it's completed."""
        try:
            custom_api.delete_namespaced_custom_object(
                name=pytorchjob_name, **args_dict
            )
            print(f"Job {pytorchjob_name} deleted.")
        except ApiException as e:
            print(f"Error deleting job: {e}")

    # Start the job and wait for a while
    start_pytorchjob()
    time.sleep(20)
    # Periodically check the job status
    while True:
        status = get_pytorchjob_status()
        logger.info(f"Current job status: {status}")

        if status == "Succeeded":
            logger.info("Job succeeded!")
            delete_pytorchjob()
            break
        elif status in ["Failed", "Error"]:
            logger.info("Job did not succeed. Exiting.")
            delete_pytorchjob()
            break

        time.sleep(10)

    return f"{minio_model_bucket}/{model_output_file}"

@dsl.component(
    packages_to_install=["kubernetes", "loguru"],
    base_image="python:3.9",
)
def run_hyper_parameter_tuning(
    df_train: Input[Dataset],
    df_val: Input[Dataset],
    experiment_name: str,
    image: str,
    namespace: str,
    model_name: str,
    seed: int,
    max_epochs: int,
    max_trials: int,
    batch_size_list: List[str],
    beta_list: List[str],
    learning_rate_list: List[str],
    kernel_size_list: List[str],
    latent_dim: int,
) -> Dict:
    import time
    from kubernetes import client, config
    from loguru import logger

    group = "kubeflow.org"
    version = "v1beta1"
    plural = "experiments"
    # Command List
    command_list = [
        "python",
        "main.py",
        "train",
        f"--df-train-path={df_train.path}",
        f"--df-val-path={df_val.path}",
        "--model-output-file=tmp-file",
        "--num-workers=12",
        "--number-of-train-samples=10000",
        "--number-of-val-samples=1000",
        "--data-module-name=simulated",
        "--batch-size=${trialParameters.batchSize}",
        "--learning-rate=${trialParameters.learningRate}",
        "--kernel-size=${trialParameters.kernelSize}",
        "--beta=${trialParameters.beta}",
        "--dropout=0.0",
        f"--max-epochs={max_epochs}",
        "--early-stopping-patience=25",
        f"--latent-dim={latent_dim}",
        f"--model-name={model_name}",
        "--seq-len=1000",
        f"--seed={seed}",
        "--config-path=./sim-all-config.toml",
        "--run-as-pytorchjob False",
    ]

    # Environment Dictionary
    env_dict = [
        {
            "name": "AWS_ACCESS_KEY_ID",
            "valueFrom": {
                "secretKeyRef": {"name": "s3creds", "key": "AWS_ACCESS_KEY_ID"}
            },
        },
        {
            "name": "AWS_SECRET_ACCESS_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": "s3creds", "key": "AWS_SECRET_ACCESS_KEY"}
            },
        },
        {"name": "S3_ENDPOINT", "value": "minio.minio"},
        {"name": "S3_USE_HTTPS", "value": "0"},
        {"name": "S3_VERIFY_SSL", "value": "0"},
    ]

    # Spec Dictionary
    spec_dict = {
        "template": {
            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
            "spec": {
                "containers": [
                    {
                        "name": "training-container",
                        "image": image,
                        "resources": {"limits": {"nvidia.com/gpu": 1}},
                        "command": command_list,
                        "env": env_dict,
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    }
                ],
                "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
                "restartPolicy": "Never",
            },
        }
    }

    # Trial Template Dictionary
    trial_template_dict = {
        "primaryContainerName": "training-container",
        "trialParameters": [
            {
                "name": "learningRate",
                "description": "Learning rate for the training model",
                "reference": "learning_rate",
            },
            {
                "name": "kernelSize",
                "description": "Kernel size of convolutional layers",
                "reference": "kernel_size",
            },
            {
                "name": "batchSize",
                "description": "Batch size for training",
                "reference": "batch_size",
            },
            {
                "name": "beta",
                "description": "beta in beta vae loss function",
                "reference": "beta",
            },
        ],
        "trialSpec": {"apiVersion": "batch/v1", "kind": "Job", "spec": spec_dict},
    }

    experiment_config = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": experiment_name, "namespace": namespace},
        "spec": {
            "parallelTrialCount": 3,
            "maxTrialCount": max_trials,
            "maxFailedTrialCount": 3,
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "objective": {
                "type": "minimize",
                "goal": 0.0001,
                "objectiveMetricName": "val_loss",
                "additionalMetricNames": ["train_loss"],
            },
            "algorithm": {"algorithmName": "bayesianoptimization"},
            "parameters": [
                {
                    "name": "learning_rate",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": learning_rate_list},
                },
                {
                    "name": "kernel_size",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": kernel_size_list},
                },
                {
                    "name": "batch_size",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": batch_size_list},
                },
                {
                    "name": "beta",
                    "parameterType": "discrete",
                    "feasibleSpace": {"list": beta_list},
                },
            ],
            "trialTemplate": trial_template_dict,
        },
    }

    config.load_incluster_config()

    k8s_client = client.ApiClient()

    katib_api_instance = client.CustomObjectsApi(k8s_client)
    katib_api_instance.create_namespaced_custom_object(
        group, version, namespace, plural, experiment_config
    )
    time.sleep(60)
    logger.info(f"Experiment {experiment_name} submitted. Waiting for completion...")

    while True:
        response = katib_api_instance.get_namespaced_custom_object(
            group, version, namespace, plural, experiment_name
        )

        status = response["status"]["conditions"][-1]["type"]
        if status == "Succeeded":
            result = response
            break
        elif status == "Failed":
            raise Exception(f"Experiment {experiment_name} failed!")

        logger.info(f"Experiment {experiment_name} running. Waiting for completion...")
        time.sleep(60)

    best_trial = result["status"]["currentOptimalTrial"]["parameterAssignments"]
    return {b["name"]: b["value"] for b in best_trial}
