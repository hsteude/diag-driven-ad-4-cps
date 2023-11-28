# Model-Related Pipeline Components for Training and Inference

This directory contains images for custom [container components](https://www.kubeflow.org/docs/components/pipelines/v2/components/container-components/) used in Kubeflow pipelines.
These images are also utilized for Katib hyperparameter tuning experiments and Kubeflow PyTorch training jobs.
All complex Kubeflow components requiring more involved functions are defined here, while lightweight components are located in the `pipelines` directory at the project's root.
`main.py` serves as the entry point for the scripts.
The benchmark models and our proposed model architecture are located in the `diag_driven_ad_models` directory.

## Local Installation
For local development, install the necessary Python dependencies using Poetry. Run the following command in the directory containing this README.md file:
```sh
poetry install
```

## Run/Debug Locally

Once the Poetry packages are installed, you can execute functions defined in the `main.py` file. Example scripts are available in `run_data_gen_locally.sh` and `run_swat_training_locally.sh`.

Ensure you download the required input files from MinIO if the scripts necessitate local files.


## Build Image

To containerize the application for use in Kubeflow components as described in the Kubeflow docs, follow these steps:
1. Export the requirements from the Poetry environment:
```sh
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

2. Build the Docker image:
```sh
docker build -t <your_registry/image_name>:<tag> .
```

3. Push the image to an image registry (e.g., Docker Hub):
```sh
docker push <your_registry>/<image_name>:<tag>
```
