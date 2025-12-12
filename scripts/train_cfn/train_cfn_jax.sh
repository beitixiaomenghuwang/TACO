#!/bin/bash
# 训练CFN模块（使用从JAX模型提取的特征）

# 特征目录（从JAX版本collect脚本生成的特征）
feature_dir="./representation_collection/pi05_teleavatar_cubestack_jax/"

# 输出目录
output_dir="./outputs/cfn_jax_pi05_teleavatar"

# CFN超参数
input_dim=1024      # 特征维度，应与提取的特征维度一致
cfn_output_dim=20   # CFN输出维度
cfn_hidden_dim=1536 # CFN隐藏层维度

# 训练超参数
batch_size=512
num_workers=16
epochs=16
save_freq=4
lr=1e-4
max_lr=1e-3
grad_accum_steps=2

python scripts/train_cfn/train_cfn.py \
    --output_dir=$output_dir \
    --feature_dir=$feature_dir \
    --input_dim=$input_dim \
    --cfn_output_dim=$cfn_output_dim \
    --cfn_hidden_dim=$cfn_hidden_dim \
    --batch_size=$batch_size \
    --num_workers=$num_workers \
    --epochs=$epochs \
    --save_freq=$save_freq \
    --lr=$lr \
    --max_lr=$max_lr \
    --grad_accum_steps=$grad_accum_steps \
    --multi_feature_file=False

echo "CFN训练完成！模型保存在: $output_dir"

