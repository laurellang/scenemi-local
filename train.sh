#!/bin/bash

# ============================================================
# SceneMI Training Script for Cluster
# ============================================================

# --- Dataset Paths ---
export TRUMANS_DATA_ROOT=/home/dataset/xingyu/trumans/Data_release
export TRUMANS_META_DIR=./dataset         # normalization stats (mean/std) save here
export BODY_MODELS_PATH=./body_models/    # SMPL-X model path (keep in project dir)

# --- Run Training ---
python -m train.train_diffusion_scenemib \
    --device 0 \
    --arch unet \
    --no_wo_frame_feature \
    --data_rep smpl \
    --batch_size 32 \
    --num_steps 1200000 \
    --lr 1e-4 \
    --lr_anneal_steps 500000 \
    --grad_clip 1.0 \
    --save_interval 50000 \
    --log_interval 500 \
    --seed 10
