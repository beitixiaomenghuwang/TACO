#!/bin/bash
# JAX版本的PI0.5 + TACO评估脚本

# 请修改以下路径！
export cfn_ckpt_path="your/path/to/cfn_ckpt.pt"  # CFN模型权重路径（保持PyTorch格式，在JAX中使用）

cd ./third_party/Robotwin

policy_name=pi05_jax
policy_path="/home/caslx/Robotics/openpi/checkpoints/pi05_teleavatar_cubestack/my_experiment/19999"  # JAX模型checkpoint路径
train_config_name="pi05_teleavatar"  # 根据你的实际训练配置修改
task_config=demo_clean
seed=1
task_name=adjust_bottle
tag=test_jax
# 输出目录: "./third_party/Robotwin/eval_result/{tag}"

export JAX_PLATFORMS=cuda  # 使用GPU

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_lerobot_jax_pi05_taco.py \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${tag} \
    --policy_path $policy_path \
    --train_config_name $train_config_name

