#!/bin/bash

poetry run python main.py generate_anomaly_dfs \
    --min-lenght-causal-phase 500 \
    --max-lenght-causal-phase 1000 \
    --num-ber-phases 500 \
    --zeta 0.3 \
    --du 1.0 \
    --taup 50 \
    --tau 20.0 \
    --y-min 0.1 \
    --y-max 1.0 \
    --rate-b2 0.01 \
    --rate-b3 0.01 \
    --mixing-factor-b2 0.5 \
    --mixing-factor-a1 0.25 \
    --seed 42 \
    --df-sig-a2-path '../../local_data/tmp_df_a2' \
    --df-sig-b3-path '../../local_data/tmp_df_b3' \
    --df-a2b-path '../../local_data/tmp_df_a2b' \
    --df-b2a-path '../../local_data/tmp_df_b2a' \
    --df-b2a-path '../../local_data/tmp_df_b2a' \
    --df-healthy-path '../../local_data/tmp_df_healthy'

