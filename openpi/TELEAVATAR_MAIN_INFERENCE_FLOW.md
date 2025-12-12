# Teleavatar Main.py 推理控制流程详解

## 📋 概览

您使用两个脚本运行 Teleavatar 机器人：

1. **策略服务器**：`uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_teleavatar_low_mem_finetune --policy.dir=pi0_teleavatar_low_mem_finetune/pi0_lora_with_joint_positions_and_gripper_efforts/29999`
2. **机器人客户端**：`python examples/teleavatar/main.py --remote-host 192.168.1.100`

---

## 🔧 配置参数详解

### 默认参数（main.py Args）

```python
control_frequency: float = 20.0      # 控制循环频率 20Hz
action_horizon: int = 8              # 策略返回的动作序列长度（未使用）
open_loop_horizon: int = 4           # 执行4个动作后重新推理
prompt: str = "pick a toy..."        # 语言指令
num_episodes: int = 1                # 运行1个episode
max_episode_steps: int = 600         # 每个episode最多600步
```

### 策略配置（pi0_teleavatar_low_mem_finetune）

```python
# src/openpi/training/config.py:843-870
model = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    action_dim=32,                   # 模型输出32维动作
    action_horizon=50                # 模型默认返回50步的动作序列！
)
```

**关键发现**：`action_horizon` 默认为 **50**（未在配置中显式设置时使用默认值）

---

## 🔄 完整推理控制流程

### 第1步：初始化 (main.py:84-115)

```python
# 1. 创建 WebSocket 客户端连接到策略服务器
ws_client_policy = WebsocketClientPolicy(
    host="192.168.1.100",  # 您的远程主机
    port=8000
)

# 2. 创建 Teleavatar 环境（ROS2接口）
environment = TeleavatarEnvironment(prompt="pick a toy...")

# 3. 创建代理，包装了 ActionChunkBroker
agent = PolicyAgent(
    policy=ActionChunkBroker(
        policy=ws_client_policy,
        action_horizon=4  # open_loop_horizon=4
    )
)

# 4. 创建运行时
runtime = Runtime(
    environment=environment,
    agent=agent,
    max_hz=20.0,         # 20Hz 控制循环
    num_episodes=1,
    max_episode_steps=600
)
```

### 第2步：运行时循环 (runtime.py)

```python
# runtime.run() -> _run_episode() -> _step() 循环

def _step(self):
    # 2.1 获取观测（20Hz）
    observation = environment.get_observation()
    # 返回: {
    #   'observation/state': [48],              # 机器人状态
    #   'observation/images/left_color': [480,848,3],
    #   'observation/images/right_color': [480,848,3],
    #   'observation/images/head_camera': [1080,1920,3],
    #   'prompt': "pick a toy..."
    # }
    
    # 2.2 获取动作（通过 agent）
    action = agent.get_action(observation)
    
    # 2.3 应用动作到机器人
    environment.apply_action(action)
    
    # 2.4 保持 20Hz 频率
    # 每 50ms 执行一次循环
```

### 第3步：动作获取流程（核心！）

#### 3.1 PolicyAgent.get_action()

```python
# policy_agent.py:14
def get_action(self, observation: dict) -> dict:
    return self._policy.infer(observation)
    # 这里的 _policy 是 ActionChunkBroker
```

#### 3.2 ActionChunkBroker.infer()（关键逻辑）

```python
# action_chunk_broker.py:27-44
def infer(self, obs: Dict) -> Dict:
    # 第一次调用或者动作用完了？
    if self._last_results is None:
        # ✅ 发起网络推理！
        self._last_results = self._policy.infer(obs)
        # 服务器返回: {"actions": [50, 16]}  <-- 50个时间步，每个16维
        self._cur_step = 0
    
    # 从动作序列中提取当前步的动作
    def slicer(x):
        if isinstance(x, np.ndarray):
            return x[self._cur_step, ...]  # 取第 cur_step 个动作
        else:
            return x
    
    results = tree.map_structure(slicer, self._last_results)
    # 返回: {"actions": [16]}  <-- 单步动作
    
    self._cur_step += 1
    
    # 已经执行了 action_horizon(4) 个动作？
    if self._cur_step >= self._action_horizon:  # >= 4
        self._last_results = None  # 清空，下次会重新推理
    
    return results
```

#### 3.3 WebsocketClientPolicy.infer()

```python
# websocket_client_policy.py:44-51
def infer(self, obs: Dict) -> Dict:
    # 序列化观测数据
    data = self._packer.pack(obs)
    
    # 发送到服务器
    self._ws.send(data)
    
    # 接收服务器响应
    response = self._ws.recv()
    
    # 解包返回
    return msgpack_numpy.unpackb(response)
    # 返回: {"actions": [50, 16]}
```

### 第4步：服务器端推理 (serve_policy.py)

```python
# websocket_policy_server.py (简化版本)
def handle_client(self, connection):
    while True:
        # 接收观测
        obs_data = connection.recv()
        obs = msgpack_numpy.unpackb(obs_data)
        
        # 调用策略推理
        action = self._policy.infer(obs)
        # 策略返回: {"actions": [50, 16]}
        
        # 发送回客户端
        response = msgpack_numpy.packb(action)
        connection.send(response)
```

---

## 📊 关键数据流分析

### 观测数据 (Environment → Agent)

```
TeleavatarEnvironment.get_observation()
↓
{
    'observation/state': np.ndarray[48],           # 48维状态
        # 布局：
        # [0:7]   左臂关节位置
        # [7:8]   左臂夹爪关节位置
        # [8:15]  右臂关节位置
        # [15:16] 右臂夹爪关节位置
        # [16:23] 左臂关节速度
        # [23:24] 左臂夹爪速度
        # [24:31] 右臂关节速度
        # [31:32] 右臂夹爪速度
        # [32:39] 左臂关节力矩
        # [39:40] 左臂夹爪力矩
        # [40:47] 右臂关节力矩
        # [47:48] 右臂夹爪力矩
    
    'observation/images/left_color': np.ndarray[480, 848, 3],
    'observation/images/right_color': np.ndarray[480, 848, 3],
    'observation/images/head_camera': np.ndarray[1080, 1920, 3],
    'prompt': "pick a toy and put it in the basket using left gripper"
}
```

### 策略输出 (Server → Client)

```
WebsocketPolicyServer.policy.infer(obs)
↓
{
    'actions': np.ndarray[50, 16]  # 50个时间步，每个16维
        # 16维动作布局：
        # [0:7]   左臂关节位置目标
        # [7:8]   左臂夹爪力矩目标
        # [8:15]  右臂关节位置目标
        # [15:16] 右臂夹爪力矩目标
}
```

### ActionChunkBroker 输出 (Agent → Environment)

```
ActionChunkBroker.infer(obs)
↓
{
    'actions': np.ndarray[16]  # 单步16维动作
}
```

---

## ⏱️ 时序分析

### 时间线（以 20Hz 控制为例）

```
时刻    步数    动作                  推理?    网络请求
────────────────────────────────────────────────────
0ms     0      actions[0] from chunk 0   ✅       ✅
50ms    1      actions[1] from chunk 0   ❌       ❌
100ms   2      actions[2] from chunk 0   ❌       ❌
150ms   3      actions[3] from chunk 0   ❌       ❌
200ms   4      actions[0] from chunk 1   ✅       ✅  <-- 重新推理
250ms   5      actions[1] from chunk 1   ❌       ❌
300ms   6      actions[2] from chunk 1   ❌       ❌
350ms   7      actions[3] from chunk 1   ❌       ❌
400ms   8      actions[0] from chunk 2   ✅       ✅  <-- 重新推理
...
```

### 推理频率计算

- **控制频率**：20 Hz
- **开环步数**：4 步
- **推理频率**：20 Hz ÷ 4 = **5 Hz** (每秒推理5次)
- **网络请求间隔**：200 ms

### 动作序列利用率

- **服务器返回**：50 个动作（shape: [50, 16]）
- **实际使用**：4 个动作
- **利用率**：4/50 = **8%**

---

## 🔍 详细代码执行示例

### 完整的4步循环

```python
# ===== 第1步 (0ms) =====
# runtime._step() 调用
observation = environment.get_observation()  # 获取传感器数据

# agent.get_action(observation)
#   → ActionChunkBroker.infer(observation)
#       → _last_results is None, 需要推理!
#       → WebsocketClientPolicy.infer(observation)
#           → 发送观测到服务器
#           → 服务器推理：返回 {"actions": [50, 16]}
#       → _last_results = {"actions": [50, 16]}
#       → _cur_step = 0
#       → 返回 actions[0] → {"actions": [16]}
#       → _cur_step = 1

action = {"actions": actions[0]}  # [16] 维
environment.apply_action(action)  # 发送到机器人
# 等待 50ms


# ===== 第2步 (50ms) =====
observation = environment.get_observation()

# agent.get_action(observation)
#   → ActionChunkBroker.infer(observation)
#       → _last_results 不为空，使用缓存
#       → 返回 actions[1] → {"actions": [16]}
#       → _cur_step = 2

action = {"actions": actions[1]}
environment.apply_action(action)
# 等待 50ms


# ===== 第3步 (100ms) =====
observation = environment.get_observation()

# agent.get_action(observation)
#   → ActionChunkBroker.infer(observation)
#       → 返回 actions[2]
#       → _cur_step = 3

action = {"actions": actions[2]}
environment.apply_action(action)
# 等待 50ms


# ===== 第4步 (150ms) =====
observation = environment.get_observation()

# agent.get_action(observation)
#   → ActionChunkBroker.infer(observation)
#       → 返回 actions[3]
#       → _cur_step = 4
#       → _cur_step (4) >= _action_horizon (4)
#       → _last_results = None  # 清空缓存

action = {"actions": actions[3]}
environment.apply_action(action)
# 等待 50ms


# ===== 第5步 (200ms) =====
# 重复第1步的流程，再次发起网络推理！
```

---

## 🎯 关键问题回答

### Q1: 输出的 action 包含多少条序列？

**分层回答**：

1. **服务器端推理输出**：
   - 返回 `[50, 16]` 的动作张量
   - 50 个时间步，每个 16 维动作

2. **ActionChunkBroker 输出**：
   - 每次返回 `[16]` 的单步动作
   - 从 50 步序列中按顺序提取

3. **实际使用**：
   - 每次推理使用前 4 步
   - 后 46 步被丢弃（利用率 8%）

### Q2: 是只进行一次推理还是执行几步后再次推理？

**答案**：**执行 4 步后再次推理**

- **机制**：ActionChunkBroker 实现了动作缓存
- **缓存大小**：`open_loop_horizon = 4`
- **推理触发条件**：

  ```python
  if self._cur_step >= self._action_horizon:  # >= 4
      self._last_results = None  # 触发下次推理
  ```

### Q3: 推理频率是多少？

- **控制循环**：20 Hz（每 50ms 一步）
- **推理频率**：5 Hz（每 200ms 一次）
- **效率提升**：比每步推理快 **4倍**

---

## 💡 优化建议

### 当前低效问题

```
服务器返回: [50, 16] = 800个浮点数
实际使用:   [4, 16]  = 64个浮点数
浪费比例:   92%
```

### 优化方案1：调整服务器 action_horizon

修改 `config.py`：

```python
TrainConfig(
    name="pi0_teleavatar_low_mem_finetune",
    model=pi0_config.Pi0Config(
        action_horizon=4,  # 改为4，匹配 open_loop_horizon
        action_dim=32
    ),
    ...
)
```

### 优化方案2：增加 open_loop_horizon

修改 `main.py`：

```python
open_loop_horizon: int = 10  # 使用更多缓存动作
```

**权衡**：

- ✅ 减少推理次数，提高效率
- ❌ 开环控制时间更长，可能影响反应速度

### 优化方案3：动态调整（推荐）

```python
# 根据任务复杂度动态调整
if task_requires_precision:
    open_loop_horizon = 2  # 更频繁的闭环反馈
else:
    open_loop_horizon = 8  # 更高效的开环执行
```

---

## 📈 与 DROID 对比

| 指标               | Teleavatar (当前) | DROID          |
|-------------------|------------------|----------------|
| 控制频率           | 20 Hz           | 15 Hz          |
| 开环步数           | 4               | 8              |
| 推理频率           | 5 Hz            | 1.875 Hz       |
| 服务器输出长度     | 50              | 15             |
| 动作利用率         | 8% (4/50)       | 53.3% (8/15)   |
| 动作维度           | 16              | 8              |
| 网络请求间隔       | 200ms           | 533ms          |

**分析**：

- Teleavatar 推理更频繁（更及时的反馈）
- DROID 利用率更高（更高效的资源使用）

---

## 🔬 调试技巧

### 添加日志查看推理时机

在 `action_chunk_broker.py` 中：

```python
def infer(self, obs: Dict) -> Dict:
    if self._last_results is None:
        print(f"🔄 [推理] 发起新的推理请求...")
        self._last_results = self._policy.infer(obs)
        self._cur_step = 0
        print(f"   收到动作序列: {self._last_results['actions'].shape}")
    
    results = tree.map_structure(slicer, self._last_results)
    print(f"📤 [步{self._cur_step}] 使用缓存动作 {self._cur_step}/{self._action_horizon}")
    self._cur_step += 1
    
    if self._cur_step >= self._action_horizon:
        print(f"✅ [完成] 动作序列用完，下次将重新推理")
        self._last_results = None
    
    return results
```

### 验证动作序列长度

在 `main.py` 中：

```python
metadata = ws_client_policy.get_server_metadata()
print(f"服务器配置: {metadata}")
# 应该包含 action_horizon 信息
```

---

## 📝 总结

### 核心流程

1. **20Hz 控制循环**：每 50ms 获取观测并执行一个动作
2. **5Hz 推理频率**：每 4 步（200ms）请求一次新的动作序列
3. **动作缓存机制**：服务器返回 50 步，客户端使用前 4 步
4. **开环执行**：在缓存的 4 步内不考虑新的传感器反馈

### 关键组件职责

- **Runtime**：维护 20Hz 控制循环
- **ActionChunkBroker**：管理动作缓存，决定何时推理
- **WebsocketClientPolicy**：与服务器通信
- **TeleavatarEnvironment**：ROS2 接口，读传感器写动作

### 性能特点

- ✅ 降低网络延迟影响（批量获取动作）
- ✅ 减少推理次数（5Hz vs 20Hz）
- ⚠️ 动作序列利用率低（8%）
- ⚠️ 开环执行可能影响精度

---

*生成时间：2025-10-17*
*基于：openpi @ commit gxy branch*
