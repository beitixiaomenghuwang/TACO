
cd ./third_party/lerobot/src/lerobot/datasets/v30

task=adjust_bottle
python convert_dataset_v21_to_v30.py \
    --repo-id=RoboTwin2/demo_clean/$task \
    --push-to-hub=false 

