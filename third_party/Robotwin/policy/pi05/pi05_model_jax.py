#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
JAX版本的PI0.5模型包装器
支持加载JAX训练的模型并结合PyTorch的CFN进行推理
"""

import sys
import os
from pathlib import Path
import numpy as np
import einops

import jax
import jax.numpy as jnp
import torch

# 导入openpi相关模块
openpi_path = str(Path(__file__).parent.parent.parent.parent.parent / "openpi/src")
sys.path.insert(0, openpi_path)
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi import transforms

# 导入PyTorch的CFN模块（保持与原始实现一致）
cfn_path = str(Path(__file__).parent.parent.parent.parent.parent / "cfn")
sys.path.insert(0, cfn_path)
from cfn.cfn_net import CFN


def make_attn_mask(input_mask, mask_ar):
    """Create attention mask"""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class JAX_PI05:
    """纯JAX版本的PI0.5模型包装器（不使用CFN）"""
    
    def __init__(self, task_name, checkpoint_path, train_config_name="pi0_fast_aloha"):
        self.task_name = task_name
        self.checkpoint_path = checkpoint_path
        self.train_config_name = train_config_name
        
        print(f"🔄 加载JAX模型配置: {train_config_name}")
        print(f"📂 Checkpoint路径: {checkpoint_path}")
        
        # 加载配置和模型
        config = _config.get_config(train_config_name)
        checkpoint_dir = Path(checkpoint_path)
        
        # 加载模型参数
        print("📥 恢复模型参数...")
        params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        
        # 创建模型
        print("🧠 创建模型...")
        self.model = config.model.load(params)
        self.config = config
        
        # 创建推理函数（JIT编译以提高性能）
        from openpi.shared import nnx_utils
        self._sample_actions_jit = nnx_utils.module_jit(self.model.sample_actions)
        
        # 初始化RNG
        self.rng = jax.random.key(0)
        
        # 图像尺寸
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        
        # 动作mask（如果需要）
        self.action_mask = np.ones(14, dtype=bool)
        self.action_mask[13] = False
        self.action_mask[6] = False
        
        self.num_result = 1
        
        print("✅ JAX模型加载完成!")
    
    def set_img_size(self, img_size):
        """设置图像尺寸"""
        self.img_size = img_size
    
    def set_language(self, instruction):
        """设置语言指令"""
        self.instruction = instruction
        print(f"📝 设置指令: {instruction}")
    
    def update_observation_window(self, img_arr, state):
        """
        更新观测窗口
        img_arr: [img_front, img_right, img_left, puppet_arm]
        state: 机器人状态
        """
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        
        # 转换图像格式：HWC -> CHW
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))
        
        # 归一化到[0, 1]
        img_front = img_front / 255.0
        img_left = img_left / 255.0
        img_right = img_right / 255.0
        
        # 转换为JAX数组
        img_front_jax = jnp.array(img_front, dtype=jnp.float32)
        img_left_jax = jnp.array(img_left, dtype=jnp.float32)
        img_right_jax = jnp.array(img_right, dtype=jnp.float32)
        state_jax = jnp.array(state, dtype=jnp.float32)
        
        # 创建OpenPI的Observation对象
        self.observation_window = {
            "images": {
                "base_0_rgb": img_front_jax[np.newaxis, ...],  # 添加batch维度
                "left_wrist_0_rgb": img_left_jax[np.newaxis, ...],
                "right_wrist_0_rgb": img_right_jax[np.newaxis, ...],
            },
            "image_masks": {
                "base_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
            },
            "state": state_jax[np.newaxis, ...],
            "tokenized_prompt": None,
            "tokenized_prompt_mask": None,
        }
    
    def get_action(self):
        """执行推理并获取动作"""
        assert self.observation_window is not None, "请先调用update_observation_window!"
        
        # 创建Observation对象
        obs = _model.Observation(**self.observation_window)
        
        # 生成新的随机密钥
        self.rng, sample_rng = jax.random.split(self.rng)
        
        # 执行推理
        actions = self._sample_actions_jit(sample_rng, obs)
        
        # 转换为numpy并返回第一个batch的结果
        actions_np = np.array(actions[0], dtype=np.float32)
        
        return actions_np
    
    def reset_obsrvationwindows(self):
        """重置观测窗口"""
        self.instruction = None
        self.observation_window = None
        print("🔄 已重置观测窗口和指令")


class JAX_PI05_TACO:
    """JAX版本的PI0.5 + TACO模型包装器"""
    
    def __init__(
        self, 
        task_name, 
        checkpoint_path, 
        train_config_name="pi0_fast_aloha",
        cfn_ckpt_path=None,
    ):
        self.task_name = task_name
        self.checkpoint_path = checkpoint_path
        self.train_config_name = train_config_name
        
        print("="*50)
        print("🚀 初始化JAX PI0.5 + TACO模型")
        print("="*50)
        print(f"🔄 加载JAX模型配置: {train_config_name}")
        print(f"📂 Checkpoint路径: {checkpoint_path}")
        
        # 加载配置和模型
        config = _config.get_config(train_config_name)
        checkpoint_dir = Path(checkpoint_path)
        
        # 加载模型参数
        print("📥 恢复模型参数...")
        params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        
        # 创建模型
        print("🧠 创建JAX模型...")
        self.model = config.model.load(params)
        self.config = config
        
        # 创建推理函数（JIT编译）
        from openpi.shared import nnx_utils
        self._sample_actions_jit = nnx_utils.module_jit(self.model.sample_actions)
        
        # 初始化RNG
        self.rng = jax.random.key(0)
        
        # 加载PyTorch的CFN模块
        print("🔧 加载PyTorch CFN模块...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfn = CFN(
            input_dim=1024,  # 特征维度，需要根据实际模型调整
            cfn_output_dim=20,
            cfn_hidden_dim=1536
        ).to(device)
        
        if cfn_ckpt_path:
            print(f"📥 加载CFN权重: {cfn_ckpt_path}")
            self.cfn.cfn.load_state_dict(torch.load(cfn_ckpt_path, map_location=device))
        else:
            print("⚠️  警告: 未提供CFN权重路径，使用随机初始化的CFN")
        
        self.cfn.eval()
        self.device = device
        
        # 图像尺寸
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        
        # 动作mask
        self.action_mask = np.ones(14, dtype=bool)
        self.action_mask[13] = False
        self.action_mask[6] = False
        
        self.num_result = 50  # 采样数量
        
        # 准备噪声（用于多样性采样）
        print(f"🎲 准备噪声样本 (num={self.num_result})...")
        seed = 42
        np.random.seed(seed)
        print(f"   使用随机种子: {seed}")
        
        noise_shape = (self.num_result, config.model.action_horizon, config.model.action_dim)
        self.noise = jnp.array(
            np.random.normal(0.0, 1.0, noise_shape),
            dtype=jnp.bfloat16
        )
        
        print("✅ JAX PI0.5 + TACO模型加载完成!")
        print("="*50)
    
    def set_img_size(self, img_size):
        """设置图像尺寸"""
        self.img_size = img_size
    
    def set_language(self, instruction):
        """设置语言指令"""
        self.instruction = instruction
        print(f"📝 设置指令: {instruction}")
    
    def update_observation_window(self, img_arr, state):
        """
        更新观测窗口
        img_arr: [img_front, img_right, img_left, puppet_arm]
        state: 机器人状态
        """
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        
        # 转换图像格式：HWC -> CHW
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))
        
        # 归一化到[0, 1]
        img_front = img_front / 255.0
        img_left = img_left / 255.0
        img_right = img_right / 255.0
        
        # 转换为JAX数组
        img_front_jax = jnp.array(img_front, dtype=jnp.float32)
        img_left_jax = jnp.array(img_left, dtype=jnp.float32)
        img_right_jax = jnp.array(img_right, dtype=jnp.float32)
        state_jax = jnp.array(state, dtype=jnp.float32)
        
        # 为多样性采样复制num_result份
        self.observation_window = {
            "images": {
                "base_0_rgb": jnp.repeat(img_front_jax[np.newaxis, ...], self.num_result, axis=0),
                "left_wrist_0_rgb": jnp.repeat(img_left_jax[np.newaxis, ...], self.num_result, axis=0),
                "right_wrist_0_rgb": jnp.repeat(img_right_jax[np.newaxis, ...], self.num_result, axis=0),
            },
            "image_masks": {
                "base_0_rgb": jnp.ones(self.num_result, dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones(self.num_result, dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones(self.num_result, dtype=jnp.bool_),
            },
            "state": jnp.repeat(state_jax[np.newaxis, ...], self.num_result, axis=0),
            "tokenized_prompt": None,
            "tokenized_prompt_mask": None,
        }
    
    def get_action(self):
        """执行推理并使用CFN选择最佳动作"""
        assert self.observation_window is not None, "请先调用update_observation_window!"
        
        # 创建Observation对象
        obs = _model.Observation(**self.observation_window)
        
        # 生成新的随机密钥
        self.rng, sample_rng = jax.random.split(self.rng)
        
        # 执行JAX推理并提取特征
        actions, features = self._sample_actions_and_get_feature(sample_rng, obs, self.noise)
        
        # 转换特征为PyTorch张量
        features_torch = torch.from_numpy(np.array(features, dtype=np.float32)).to(self.device)
        
        # 使用CFN计算每个动作的得分
        with torch.no_grad():
            cfn_output = self.cfn.cfn(features_torch)
            norm = cfn_output.norm(dim=1)
            
            # 选择norm最小的动作（最接近先验分布）
            min_val = torch.min(norm)
            indices = torch.nonzero(norm == min_val).squeeze()
            
            if indices.ndim == 0:
                selected_index = indices.item()
            else:
                selected_index = indices[torch.randint(0, len(indices), (1,))].item()
        
        # 返回选中的动作
        actions_np = np.array(actions, dtype=np.float32)
        selected_action = actions_np[selected_index]
        
        return selected_action
    
    def _sample_actions_and_get_feature(self, rng, observation, noise):
        """
        从JAX模型中采样动作并提取特征
        这是内部方法，实现了特征提取逻辑
        """
        # 注意：observation已经包含了所需格式，不需要再preprocess
        
        # 初始化
        num_steps = 10
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # 首先用前缀做一次前向传播填充KV缓存
        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        # 去噪循环
        x_t = noise
        time = 1.0
        feature = None
        
        for step_idx in range(num_steps):
            # 嵌入suffix
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.model.embed_suffix(
                observation, x_t, jnp.broadcast_to(jnp.array(time), (batch_size,))
            )
            
            # 创建attention mask
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)
            
            # 位置编码
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            # 前向传播
            (prefix_out, suffix_out), _ = self.model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            
            # 获取速度预测
            suffix_out_action = suffix_out[:, -self.model.action_horizon:]
            v_t = self.model.action_out_proj(suffix_out_action)
            
            # 如果是最后一步，提取特征
            if step_idx == num_steps - 1:
                feature = suffix_out[:, 0]  # 取第一个token作为特征
            
            # 更新x_t和time
            x_t = x_t + dt * v_t
            time = time + dt
            
            if time < -dt / 2:
                break
        
        return x_t, feature
    
    def reset_obsrvationwindows(self):
        """重置观测窗口"""
        self.instruction = None
        self.observation_window = None
        print("🔄 已重置观测窗口和指令")


# 用于测试的辅助函数
if __name__ == "__main__":
    print("测试JAX PI0.5模型包装器...")
    
    # 示例用法
    model = JAX_PI05_TACO(
        task_name="test_task",
        checkpoint_path="/path/to/checkpoint",
        train_config_name="pi0_fast_aloha",
        cfn_ckpt_path="/path/to/cfn.pt"
    )
    
    print("模型创建成功!")

