"""
JAX版本的PI0.5特征提取脚本
从JAX训练的模型中提取内部特征表示，用于后续CFN训练
"""

import os
import sys
from pathlib import Path

# ========== 关键：在所有导入之前先设置路径优先级 ==========
# 添加项目中的lerobot路径到最前面，优先于pip安装的版本
lerobot_path = str(Path(__file__).parent.parent.parent.parent / "third_party/lerobot/src")
sys.path.insert(0, lerobot_path)

# 添加openpi路径
openpi_path = str(Path(__file__).parent.parent.parent.parent / "openpi/src")
sys.path.insert(0, openpi_path)
# =========================================================

import logging
from pprint import pformat
import argparse
from tqdm import tqdm
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 导入openpi相关模块
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi import transforms

# 导入特征提取辅助模块
from jax_pi05_feature_extractor import sample_actions_and_get_feature, compute_best_noise_index

# 导入lerobot数据加载模块（只用于数据加载，不用于模型）
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def parse_args():
    parser = argparse.ArgumentParser(description="JAX版本PI0.5特征提取")
    
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                       help="LeRobot数据集路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="特征保存路径")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="JAX模型checkpoint路径")
    parser.add_argument("--train_config_name", type=str, default="pi0_fast_aloha",
                       help="openpi训练配置名称")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--noise_num", type=int, default=50,
                       help="噪声样本数量")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--image_keys", type=str, nargs='+',
                       default=["observation.images.cam_high",
                               "observation.images.cam_left_wrist",
                               "observation.images.cam_right_wrist"],
                       help="图像观测的key")
    parser.add_argument("--state_key", type=str, default="observation.state",
                       help="状态观测的key")
    
    return parser.parse_args()


def load_jax_model(train_config_name: str, checkpoint_path: str):
    """加载JAX训练的PI0.5模型"""
    logging.info(f"加载JAX模型配置: {train_config_name}")
    logging.info(f"模型checkpoint路径: {checkpoint_path}")
    
    config = _config.get_config(train_config_name)
    checkpoint_dir = Path(checkpoint_path)
    
    # 加载模型参数
    logging.info("恢复模型参数...")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    
    # 创建模型
    logging.info("创建模型...")
    model = config.model.load(params)
    model.eval()  # 设置为评估模式
    
    return model, config


class SimpleLeRobotDataset(Dataset):
    """简化的LeRobot数据集加载器"""
    def __init__(self, repo_id):
        logging.info(f"加载LeRobot数据集: {repo_id}")
        self.dataset = LeRobotDataset(repo_id)
        logging.info(f"数据集大小: {len(self.dataset)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]




def prepare_observation_single(batch, image_keys, state_key):
    """将LeRobot单个样本数据格式转换为OpenPI格式"""
    # 准备图像（假设数据格式为 CHW 且已归一化）
    images = {}
    image_masks = {}
    
    # 映射LeRobot的相机名到OpenPI的相机名
    camera_mapping = {
        "observation.images.cam_high": "base_0_rgb",
        "observation.images.cam_left_wrist": "left_wrist_0_rgb",
        "observation.images.cam_right_wrist": "right_wrist_0_rgb",
    }
    
    for lerobot_key, openpi_key in camera_mapping.items():
        if lerobot_key in batch:
            img = batch[lerobot_key]
            # 转换图像格式：C,H,W -> H,W,C
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            if len(img.shape) == 3 and img.shape[0] == 3:
                # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            
            # 归一化到[0, 1]
            if img.max() > 1.0:
                img = img / 255.0
            
            # 转换为JAX数组
            images[openpi_key] = jnp.array(img, dtype=jnp.float32)
            image_masks[openpi_key] = jnp.array(True, dtype=jnp.bool_)
        else:
            # 如果某个相机不存在，创建空白图像和mask
            images[openpi_key] = jnp.zeros((224, 224, 3), dtype=jnp.float32)
            image_masks[openpi_key] = jnp.array(False, dtype=jnp.bool_)
    
    # 准备状态
    state_data = batch[state_key]
    if isinstance(state_data, torch.Tensor):
        state_data = state_data.numpy()
    state = jnp.array(state_data, dtype=jnp.float32)
    
    # 创建OpenPI的Observation对象（不包含prompt，会在预处理时添加）
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )
    
    return obs


def main():
    args = parse_args()
    init_logging()
    
    logging.info("="*50)
    logging.info("JAX版本PI0.5特征提取")
    logging.info("="*50)
    logging.info(pformat(vars(args)))
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载JAX模型
    model, config = load_jax_model(args.train_config_name, args.policy_path)
    
    # 创建数据集
    logging.info("创建数据集...")
    dataset = SimpleLeRobotDataset(args.dataset_repo_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    
    # 准备噪声
    logging.info(f"准备噪声样本 (seed={args.seed}, num={args.noise_num})...")
    rng = jax.random.key(args.seed)
    
    # 提取特征
    logging.info("开始提取特征...")
    logging.info(f"注意：每个样本将生成{args.noise_num}个噪声样本，并选择最接近GT的一个")
    features_list = []
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="处理批次")):
        batch_size = batch[args.state_key].shape[0]
        
        for i in range(batch_size):
            total_samples += 1
            
            # 提取单个样本
            single_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    single_batch[key] = value[i]
                elif isinstance(value, list):
                    single_batch[key] = value[i]
                else:
                    single_batch[key] = value
            
            # 转换为OpenPI格式（单个样本）
            obs_single = prepare_observation_single(single_batch, args.image_keys, args.state_key)
            
            # 准备ground truth action
            gt_action = single_batch["action"]
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.numpy()
            gt_action_jax = jnp.array(gt_action, dtype=jnp.float32)
            
            # 生成多个噪声样本
            rng, noise_rng = jax.random.split(rng)
            noise_samples = jax.random.normal(
                noise_rng,
                (args.noise_num, config.model.action_horizon, config.model.action_dim),
                dtype=jnp.float32
            )
            
            # 复制观测为noise_num份
            obs_repeated = _model.Observation(
                images={k: jnp.repeat(v[None, ...], args.noise_num, axis=0) for k, v in obs_single.images.items()},
                image_masks={k: jnp.repeat(v[None, ...], args.noise_num, axis=0) for k, v in obs_single.image_masks.items()},
                state=jnp.repeat(obs_single.state[None, ...], args.noise_num, axis=0),
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
            
            # 使用JAX模型提取特征
            rng, sample_rng = jax.random.split(rng)
            try:
                # 调用特征提取函数
                actions, features = sample_actions_and_get_feature(
                    model, sample_rng, obs_repeated,
                    noise=noise_samples,
                    num_steps=10
                )
                
                # 从多个噪声样本中选择最佳的
                best_idx, best_feature = compute_best_noise_index(
                    actions, features, gt_action_jax
                )
                
                features_list.append(np.array(best_feature))
                
                if total_samples % 100 == 0:
                    logging.info(f"已处理 {total_samples} 个样本, 最佳噪声索引: {best_idx}")
                    
            except Exception as e:
                logging.error(f"处理样本 {total_samples} 时出错: {e}")
                import traceback
                traceback.print_exc()
                # 使用零特征作为后备
                feature_dim = 1024  # 假设特征维度
                features_list.append(np.zeros((feature_dim,), dtype=np.float32))
    
    # 保存特征
    logging.info("保存特征...")
    features_array = np.stack(features_list)
    
    save_path = output_dir / "feature.pt"
    # 转换为PyTorch tensor以便后续训练CFN
    import torch as torch_save
    torch_save.save(torch_save.from_numpy(features_array), save_path)
    
    logging.info(f"特征已保存到: {save_path}")
    logging.info(f"特征形状: {features_array.shape}")
    logging.info(f"总共处理了 {total_samples} 个样本")
    
    # 同时保存为numpy格式
    np_save_path = output_dir / "feature.npy"
    np.save(np_save_path, features_array)
    logging.info(f"也保存为numpy格式: {np_save_path}")


if __name__ == "__main__":
    main()

