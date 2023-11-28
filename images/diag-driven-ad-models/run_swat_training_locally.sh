#!/bin/bash


poetry run python main.py train \
    --df-train-path "/minio/mlpipeline/v2/artifacts/diag-tcn-swat-pipeline/a75b802b-e6d1-4d2f-9d19-4c0c82d0f258/split-data/train_df_out" \
    --df-val-path "minio://mlpipeline/v2/artifacts/diag-tcn-swat-pipeline/60acdd14-2323-4cb2-a04f-5f226bf4495d/split-data/val_df_out" \
    --model-output-file "../../local_data/some_model" \
    --minio-model-bucket "hs-bucket"  \
    --batch-size 64 \
    --num-workers 8 \
    --number-of-train-samples 10_000 \
    --number-of-val-samples 1000 \
    --data-module-name "SWaT" \
    --kernel-size 15 \
    --max-epochs 3 \
    --learning-rate 0.001 \
    --early-stopping-patience 10 \
    --latent-dim 2 \
    --dropout 0.2 \
    --model-name "combined-univariate-tcn-vae" \
    --seq-len 100 \
    --beta 0.0001 \
    --seed 42 \
    --config-path "./swat-config.toml" \
    --run-as-pytorchjob False
   

 # vanilla-tcn-vae, combined-univariate-tcn-vae, multi-latent-tcn-vae
