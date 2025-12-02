
export TOKENIZERS_PARALLELISM=false

DATASET_REPO_ID="your/lerobot/dataset/report/id" # have to be lerobot dataset v3.0, e.g. RoboTwin2/demo_clean/${task}_v30
output_dir="./representation_collection/pi05_robotwin2/" # The path where you will store the collected features 
policy_path="your/policy/path" # model report id, or trained lerobot model checkpoint

python scripts/collect_inernal_representation/pi05_robotwin2/collect.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --output_dir=$output_dir \
    --batch_size=32 \
    --num_workers=16 \
    --policy.type=pi05 \
    --policy.pretrained_path=$policy_path \
    --policy.push_to_hub=false


