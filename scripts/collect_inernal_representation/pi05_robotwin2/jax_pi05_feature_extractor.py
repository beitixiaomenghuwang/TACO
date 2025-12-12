"""
JAX版本的PI0.5模型特征提取辅助模块
扩展openpi模型以支持特征提取
"""

import jax
import jax.numpy as jnp
import einops
from typing import Tuple

from openpi.models import model as _model
from openpi.shared import array_typing as at


def make_attn_mask(input_mask, mask_ar):
    """Create attention mask (from openpi)"""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def sample_actions_and_get_feature(
    model,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    num_steps: int = 10,
) -> Tuple[_model.Actions, at.Float[at.Array, "b feature_dim"]]:
    """
    从JAX模型中采样动作并提取特征
    
    Args:
        model: openpi的Pi0模型实例
        rng: JAX随机密钥
        observation: 观测数据
        noise: 可选的噪声输入
        num_steps: 去噪步骤数
        
    Returns:
        actions: 采样的动作序列
        features: 提取的特征（每个样本一个特征向量）
    """
    # 注意：observation已经通过preprocess_observation处理过
    # 不需要再次调用
    
    # 初始化
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    
    # 首先用前缀做一次前向传播填充KV缓存
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    
    # 去噪循环
    x_t = noise
    time = 1.0
    feature = None  # 将在最后一步提取特征
    
    for step_idx in range(num_steps):
        # 嵌入suffix
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(jnp.array(time), (batch_size,))
        )
        
        # 创建attention mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)
        
        # 位置编码
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        
        # 前向传播
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        
        # 获取速度预测
        suffix_out_action = suffix_out[:, -model.action_horizon:]
        v_t = model.action_out_proj(suffix_out_action)
        
        # 如果是最后一步，提取特征（类似PyTorch版本的suffix_out[:, 0]）
        if step_idx == num_steps - 1:
            feature = suffix_out[:, 0]  # 取第一个token作为特征
        
        # 更新x_t和time
        x_t = x_t + dt * v_t
        time = time + dt
        
        # 提前终止条件
        if time < -dt / 2:
            break
    
    return x_t, feature


def compute_best_noise_index(
    actions: at.Float[at.Array, "n ah ad"],
    features: at.Float[at.Array, "n feature_dim"],
    gt_action: at.Float[at.Array, "ah ad"],
) -> Tuple[int, at.Float[at.Array, "feature_dim"]]:
    """
    从多个噪声样本中选择最佳的一个
    
    Args:
        actions: 所有噪声样本生成的动作 (n, action_horizon, action_dim)
        features: 所有噪声样本的特征 (n, feature_dim)
        gt_action: ground truth动作 (action_horizon, action_dim)
        
    Returns:
        best_index: 最佳噪声样本的索引
        best_feature: 对应的特征
    """
    # 计算L2范数
    gt_action_expanded = jnp.broadcast_to(gt_action[None, ...], actions.shape)
    norms = jnp.linalg.norm(actions - gt_action_expanded, axis=(1, 2))
    
    # 找到最小范数的索引
    best_index = jnp.argmin(norms)
    best_feature = features[best_index]
    
    return int(best_index), best_feature

