
cd ./third_party/lerobot/src/lerobot/datasets/v30

task=adjust_bottle

# skip push to hub
python augment_dataset_quantile_stats.py \
    --repo-id=RoboTwin2/demo_clean/${task}_v30

