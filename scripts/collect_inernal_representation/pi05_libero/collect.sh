
export TOKENIZERS_PARALLELISM=false

DATASET_REPO_ID="HuggingFaceVLA/libero" # offered by lerobot, have to be lerobot dataset v3.0
output_dir="./representation_collection/pi05_libero/" # The path where you will store the collected features 
policy_path="lerobot/pi05_libero_finetuned" # model report id, or trained lerobot model checkpoint, e.g. huggingface_cache/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855

python scripts/collect_inernal_representation/pi05_libero/collect.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --output_dir=$output_dir \
    --batch_size=32 \
    --num_workers=16 \
    --policy.type=pi05 \
    --policy.pretrained_path=$policy_path \
    --policy.push_to_hub=false

