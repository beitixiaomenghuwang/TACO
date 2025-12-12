# Teleavatar 推理分析报告

## 问题总结

分析了OpenPI项目中teleavatar机器人的训练配置、推理服务和机器人控制脚本，发现并修复了一个**关键问题**，现在系统应该可以正确工作。

## 修复的问题

### 🔴 问题1：动作维度不匹配 (已修复)

**原始配置：**

```python
# config.py - pi0_teleavatar_low_mem_finetune
model=pi0_config.Pi0Config(
    action_dim=32  # ❌ 错误：模型输出32维动作
)
```

**机器人实际需求：**

```python
# deploy_policy_bridge.py
# 动作格式（16维）：
# - [0:7]: 左臂关节位置
# - [7:8]: 左夹爪位置
# - [8:15]: 右臂关节位置
# - [15:16]: 右夹爪位置
```

**修复：**

```python
# 已修改 config.py 第848行
model=pi0_config.Pi0Config(
    action_dim=16  # ✅ 修复：匹配实际机器人控制需求
)
```

**影响：** 如果不修复，模型会输出32维动作，但机器人只使用前16维，导致训练的后16维信息被丢弃，模型性能严重下降。

## 验证通过的方面

### ✅ 状态维度匹配

**机器人构建的状态 (deploy_policy_bridge.py)：**

```python
state_48d = np.zeros(48, dtype=np.float32)
# 布局：positions[16] + velocities[16] + efforts[16]
state_48d[0:7]   = 左臂关节位置 (7)
state_48d[7]     = 左夹爪位置 (1)
state_48d[8:15]  = 右臂关节位置 (7)
state_48d[15]    = 右夹爪位置 (1)
state_48d[16:23] = 左臂关节速度 (7)
state_48d[23]    = 左夹爪速度 (1)
state_48d[24:31] = 右臂关节速度 (7)
state_48d[31]    = 右夹爪速度 (1)
state_48d[32:39] = 左臂关节力矩 (7)
state_48d[39]    = 左夹爪力矩 (1)
state_48d[40:47] = 右臂关节力矩 (7)
state_48d[47]    = 右夹爪力矩 (1)
```

**模型处理的状态 (teleavatar_policy.py)：**

```python
# TeleavatarInputs从48维提取16维
state_16d = np.concatenate([
    data["observation/state"][0:7],   # 左臂关节位置
    data["observation/state"][39:40], # 左夹爪力矩
    data["observation/state"][8:15],  # 右臂关节位置
    data["observation/state"][47:48], # 右夹爪力矩
], axis=0)
```

**结论：** 布局完全匹配 ✅

### ✅ 增量动作转换

**训练配置 (已修复)：**

```python
data=LeRobotTeleavatarDataConfig(
    use_delta_joint_actions=True,  # ✅ 启用增量动作
)
```

**工作原理：**

1. **训练时：**
   - 输入：`DeltaActions` transform将绝对动作转换为增量动作（相对于当前状态）
   - 模型学习预测增量动作
   - 输出：仅用于验证（不在实际推理中使用）

2. **推理时：**
   - 模型输出：增量动作
   - `AbsoluteActions` transform自动转换：

     ```python
     # transforms.py - AbsoluteActions
     actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
     ```

   - 结果：绝对动作（可直接发送到机器人）

3. **增量动作掩码：**

   ```python
   delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
   # 结果：[True]*7 + [False]*1 + [True]*7 + [False]*1
   # 意义：对左右臂关节应用增量，夹爪保持绝对值
   ```

**结论：** 增量动作会在推理时自动转换为绝对动作，机器人接收的是正确的绝对位置 ✅

### ✅ 观测键格式

**数据流对比：**

| 阶段 | 数据源 | 键格式 | 示例 |
|-----|-------|--------|------|
| 训练数据 | LeRobot数据集 | 使用'.'分隔 | `"observation.images.left_color"` |
| 训练处理 | RepackTransform | 转换为'/'分隔 | `"observation/images/left_color"` |
| 推理输入 | deploy_policy_bridge.py | 直接使用'/'分隔 | `'observation/images/left_color'` |

**验证：**

```python
# deploy_policy_bridge.py构建的观测
obs = {
    'observation/images/left_color': cv_image,    # ✅ 使用'/'
    'observation/images/right_color': cv_image,   # ✅ 使用'/'
    'observation/images/head_camera': cv_image,   # ✅ 使用'/'
    'observation/state': state_48d,               # ✅ 使用'/'
}

# 这与训练时经过repack_transform后的格式完全一致
```

**结论：** 观测键格式匹配 ✅

## 完整数据流

### 训练时

```
LeRobot数据集 (使用'.'分隔的键)
    ↓ RepackTransform
转换为'/'分隔的键
    ↓ TeleavatarInputs
48维状态 → 16维状态, 提取16维动作
    ↓ DeltaActions
绝对动作 → 增量动作
    ↓ Normalize
归一化
    ↓ Model
模型训练
```

### 推理时

```
ROS2机器人 (deploy_policy_bridge.py)
    ↓ 构建观测 (已经是'/'分隔的键)
48维状态 + 3个相机图像
    ↓ TeleavatarInputs
48维状态 → 16维状态
    ↓ Normalize
归一化
    ↓ Model
预测增量动作
    ↓ Unnormalize
反归一化
    ↓ AbsoluteActions
增量动作 → 绝对动作 (使用输入的state)
    ↓ TeleavatarOutputs
提取前16维动作
    ↓ ROS2发布
控制机器人运动
```

## 推理命令

**1. 启动策略服务器：**

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_low_mem_finetune \
    --policy.dir=checkpoints/pi0_teleavatar_low_mem_finetune/my_experiment/20000
```

**2. 运行机器人控制脚本：**

```bash
python examples/teleavatar/deploy_policy_bridge.py \
    --server-url ws://localhost:8000 \
    --control-frequency 30.0
```

## 重要注意事项

### ⚠️ 需要重新训练

由于修改了`action_dim`从32改为16，**已有的检查点将不兼容**。需要使用修改后的配置重新训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi0_teleavatar_low_mem_finetune \
    --exp_name=my_experiment
```

### ⚠️ 检查点路径

确保训练完成后的检查点路径正确，例如：

```
checkpoints/pi0_teleavatar_low_mem_finetune/my_experiment/20000/
├── params/
├── assets/
├── model.safetensors (如果使用PyTorch)
└── ...
```

### ⚠️ ROS2环境

确保机器人控制脚本运行在正确的ROS2环境中，并且以下ROS2主题可用：

- `/left/color/image_raw`
- `/right/color/image_raw`
- `/xr_video_topic/image_raw`
- `/left_arm/joint_states`
- `/right_arm/joint_states`
- `/left_gripper/joint_states`
- `/right_gripper/joint_states`

## 总结

✅ **修复后的系统应该可以正确工作**

主要修复：

1. ✅ 动作维度从32改为16
2. ✅ 启用增量动作转换
3. ✅ 验证状态维度匹配
4. ✅ 验证观测键格式匹配

下一步：

1. 使用修改后的配置重新训练模型
2. 训练完成后，使用上述命令启动推理
3. 监控机器人行为，根据需要调整控制频率和其他参数
