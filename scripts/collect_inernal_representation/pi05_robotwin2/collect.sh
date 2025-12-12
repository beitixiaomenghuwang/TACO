
export TOKENIZERS_PARALLELISM=false

DATASET_REPO_ID="/media/caslx/0E73-05CF/Data/cubestack2025_1126_merge_v30" # offered by lerobot, have to be lerobot dataset v3.0
output_dir="./representation_collection/pi05_teleavatar_cubestack/" # The path where you will store the collected features 
policy_path="/home/caslx/Robotics/openpi/checkpoints/pi05_teleavatar_cubestack/my_experiment/19999"  # model report id, or trained lerobot model checkpoint

python scripts/collect_inernal_representation/pi05_robotwin2/collect.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --output_dir=$output_dir \
    --batch_size=32 \
    --num_workers=16 \
    --policy.type=pi05 \
    --policy.pretrained_path=$policy_path \
    --policy.push_to_hub=false


