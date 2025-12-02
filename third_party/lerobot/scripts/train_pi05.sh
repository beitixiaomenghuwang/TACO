
export TOKENIZERS_PARALLELISM=false

task=adjust_bottle

DATASET_REPO_ID="RoboTwin2/demo_clean/${task}_v30"
output_dir="./outputs/pi05_training/$task"

python ./third_party/lerobot/src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$DATASET_REPO_ID \
    --policy.type=pi05 \
    --output_dir=$output_dir \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=30000 \
    --save_freq=10000 \
    --policy.device=cuda \
    --batch_size=32 \
    --num_workers=16

