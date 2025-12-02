
rt2_root=./third_party/Robotwin
cd "${rt2_root}/policy/pi0"

task_config=demo_clean
expert_data_num=50 # episodes num
task_name=adjust_bottle

echo now task is $task_name !!!!!
hf_repo_id=RoboTwin2/${task_config}/${task_name}

# make sure we have the dir
check_dir_empty_or_create() {
DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "dir $DIR does not exist, creating..."
    mkdir -p "$DIR"
fi

if [ "$(ls -A "$DIR")" ]; then
    echo "dir $DIR is not empty, terminate execution."
    exit 1
fi

echo "dir $DIR is empty, continue."
}

check_dir_empty_or_create "${rt2_root}/policy/pi0/processed_data"
check_dir_empty_or_create "${rt2_root}/policy/pi0/training_data"

# hdf5 to hdf5
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}

mv ${rt2_root}/policy/pi0/processed_data/* ${rt2_root}/policy/pi0/training_data/temp

# hdf5 to lerobot
hdf5_path=./training_data/temp
bash generate.sh ${hdf5_path} ${hf_repo_id}

# rm -r ${rt2_root}/policy/pi0/training_data/temp





