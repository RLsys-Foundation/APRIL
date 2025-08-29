# APRIL: Active Partial Rollouts in ReInforcement Learning to tame long-tail generation
## 关于
### 背景：同步 RL 的采样–训练闭环为何被“长尾”拖累

- 在 on‑policy 的 RLHF/GRxO 训练中，系统按“轮”（round）收集 **N** 条 rollout 样本后才能进入一次更新。由于生成长度、拒答/重试、路由排队等随机性，单轮时长由**最慢的少数样本**决定（典型长尾分布），GPU 在等待“尾巴”时空转、有效吞吐被拖慢。
    
- 常见缓解手段（加大超时/截断、提高并发、上更快的解码内核/连续批处理，或改为异步 RL）各有代价：要么牺牲样本质量与策略一致性，要么改变训练假设、调度复杂度陡增，而且仍无法从根因上减少“等待未完样本”的浪费。
    
### 我们做了什么：Active Partial Rollout（APRIL）

**核心思想**：在每轮**有计划地过量采样**（N' > N），一旦达成目标 **N** 条就**主动中断**拖后的请求；把**未完成的响应片段**（含上下文与解码状态）写入**跨轮缓冲**，并在下一轮**优先续写**，从而消灭“等尾巴”的无效时间。
（TODO：加架构图）
![scheduling](./imgs/partial_scheduling.png)
### 亮点特性

- **长尾克星**：过量发起 N' > N 的 rollouts；当达成目标 N 后立刻中断剩余请求，将**未完成响应加入缓冲**，并在下一轮**优先续写**。
- **稳态训练**：对 PPO/GRPO/DAPO/GSPO 等主流 on‑policy 变体友好，实践中保持甚至略有精度提升。
- **工程无侵入**：作用于系统调度层，不修改解码/批处理内核；已与 **slime** 框架打通，NVIDIA/AMD 通用。
- **算法兼容**：继续样本可能跨策略版本产生“轻微 off‑policy”，实践中未见不稳定，且常带来正向正则化效果（更快收敛、略增精度）。

## 三步上手

### 1) 环境准备

**推荐（Docker）**
- AMD：
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch-vim /bin/bash
```
- Nvidia（TODO：找到对应的 docker 版本号）：
### 2) 安装 APRIL

```bash
git clone https://github.com/RLsys-Foundation/APRIL.git
cd APRIL
pip install -e .
```

若使用源码内的示例脚本，确保已安装 `ray` 与可用的推理后端（SGLang/vLLM 二选一，推荐 SGLang）。
### 3) 运行实例

_脚本会：启动后端 → 发起过量 rollouts → 达标即中断 → 将未完成的样本写入缓冲并在下一轮续写 → 打印单轮吞吐/时延对比。_

```bash
bash scripts/partial_rollout/qwen/grpo/run-qwen3-4B-dapo-partial.sh
```
### 4) 参数详解

partial rollout 的核心功能集中在如下参数
```bash
# 开启 partial rollout 功能
# 设置该参数来启用 rollout 时的数量达标中断生成 + 未完成样本回收机制
--partial-rollout

# 采样的 batch size 大小。该参数控制单轮的采样粒度。
# 若该参数 > rollout_batch_size，则进行过采样
# 若改参数 < rollout_batch_size，会持续不断按该粒度进行采样，直到收集到 rollout_batch_size 个样本
--over-sampling-batch-size 16
```
其他参数，可参考 [arguments.py](./slime/utils/arguments.py) 的参数进行配置，更多细节可以参考 [slime](https://github.com/THUDM/slime) 原仓库。
## 结果与对照（精简版）

| Dataset       | Model    | Metric     | APRIL 相对基线          |
| ------------- | -------- | ---------- | ------------------- |
| DAPO‑Math‑17k | Qwen3‑4B | Rollout 吞吐 | **+17%**            |
| DeepScaleR    | Qwen3‑4B | Rollout 吞吐 | **+21%**            |
| DeepMath‑103K | Qwen3‑4B | Rollout 吞吐 | **+35%**            |
| AIME‑2024     | 多设置      | 最终准确率      | **+2–5%**（视数据/算法而定） |

![evaluation](./imgs/eval_dapo_qwen.png)

## 常见问答（FAQ）

- **Q：APRIL 会不会影响策略纯度与收敛？**
    
    - A：从工程与实验看无显著不稳定，建议监控 off‑policy token 比例，并保持 `oversample ≈ 2× roll_batch` 的温和设置。
        
- **Q：需要改动解码内核吗？**
    
    - A：不需要。APRIL 作用在**系统调度层**，与 speculative decoding、continuous batching 等推理加速手段可叠加。
        
- **Q：NVIDIA/AMD 都能用吗？**
    
    - A：可以；我们在 8×H100 与 8×MI300 上均复现收益。
        
## 目录结构

```
APRIL/
├── scripts/
│   └── partial_rollout/
│       ├── deepseek/            # deepseek-r1-distill-1.5B 实验代码
│       └── qwen/                # qwen3-4B 实验代码
├── slime/
│   ├── backends/
│   ├── rollout/
│   │   └── sglang_example.py    # 采样核心代码
│   ├── ray/                     # 核心调度逻辑
│   │   └── buffer.py            # 缓冲区代码实现
│   └── utils/
└── tools/                       # megatron 格式转换

```
## 引用

若本项目对你有帮助，请在论文或项目中引用 APRIL 论文并给仓库加 ⭐。
（TODO：论文 arxiv 链接）