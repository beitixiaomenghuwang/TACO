
# bash generate.sh ./training_data/demo_randomized/ RoboTwin2/

data_dir=${1}
repo_id=${2}
# uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id
python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id
