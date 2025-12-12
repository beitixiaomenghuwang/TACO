#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cuda  # 使用GPU

# 设置PYTHONPATH，优先使用项目中的lerobot和openpi
PROJECT_ROOT="/home/caslx/Robotics/TACO"
export PYTHONPATH="${PROJECT_ROOT}/third_party/lerobot/src:${PROJECT_ROOT}/openpi/src:${PYTHONPATH}"

DATASET_REPO_ID="/media/caslx/0E73-05CF/Data/cubestack2025_1126_merge_v30"
output_dir="./representation_collection/pi05_teleavatar_cubestack_jax/"
policy_path="/home/caslx/Robotics/openpi/checkpoints/pi05_teleavatar_cubestack/my_experiment/19999"
train_config_name="pi05_teleavatar"  # 根据你的实际训练配置修改

python scripts/collect_inernal_representation/pi05_robotwin2/collect_jax.py \
    --dataset_repo_id=$DATASET_REPO_ID \
    --output_dir=$output_dir \
    --policy_path=$policy_path \
    --train_config_name=$train_config_name \
    --batch_size=4 \
    --num_workers=16 \
    --noise_num=50 \
    --seed=42

