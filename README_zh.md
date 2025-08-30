APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation
About
Background: Why the sampling-training loop of synchronous RL is dragged down by the "long tail"
In on-policy RLHF/GR?O training, the system enters an update phase only after collecting N rollout samples in a "round." Due to the inconsistent lengths of generated samples, the system has to wait for a few long-tail samples to complete before starting the training phase. This leads to decreased GPU utilization and lower throughput in the later stages of the rollout phase.

What We Did: Active Partial Rollout (APRIL)
Core Idea: In each round, we over-sample (N' > N) and actively interrupt the remaining in-progress requests once the target of N completed samples is reached. The unfinished responses are stored in a buffer and are prioritized for continued rollout in the next round, thereby mitigating the efficiency degradation caused by long-tail requests.

Highlights
Over-sampling: Assuming the training phase requires rollout_batch_size=32 complete samples per round, we actually initiate a larger sampling request, i.e., over_sampling_batch_size=64.

Stop upon collection: As soon as the number of collected complete sample groups reaches rollout_batch_size, an abort signal is immediately sent to the sglang router.

Collect and reuse: Upon receiving the abort signal, sglang stops the ongoing generation tasks and returns their partially generated portions (half-completed trajectories). This partial data is not discarded but is stored in a buffer. When the next rollout round begins, they continue generating from where they left off, along with new prompts, thus achieving seamless reuse across iteration steps.

Elegant implementation: Slime's partial rollout provides a more native and lightweight optimization solution that is less intrusive to the original pipeline. You can enable it out-of-the-box simply by setting the --partial-rollout flag and specifying --over-sampling-batch-size.

Three Steps to Get Started
1) Environment Setup (Requires an AMD GPU)
Start docker

Bash

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch-vim /bin/bash
2) Install APRIL
Bash

git clone https://github.com/RLsys-Foundation/APRIL.git
cd APRIL
pip install -e .
3) Run an Example
All scripts are in the scripts/partial_rollout/ directory.

Bash

bash scripts/partial_rollout/qwen/grpo/run-qwen3-4B-dapo-partial.sh
4) Parameter Details
The core functionality of partial rollout is controlled by the following parameters:

Bash

# Enable the partial rollout feature
# Set this parameter to enable the mechanism of stopping generation upon reaching the target count + recycling unfinished samples
--partial-rollout

# The batch size for sampling. This parameter controls the sampling granularity per round.
# If this parameter > rollout_batch_size, over-sampling is performed.
# If this parameter < rollout_batch_size, sampling will continue at this granularity until rollout_batch_size samples are collected.
--over-sampling-batch-size 16
For other parameters, please refer to the arguments in arguments.py. For more details, you can consult the original slime repository.

Results and Comparison (Abridged)
Dataset	Model	Metric	APRIL vs. Baseline
DAPO‑Math‑17k	Qwen3‑4B	Rollout Throughput	+17%
DeepScaleR	Qwen3‑4B	Rollout Throughput	+21%
DeepMath‑103K	Qwen3‑4B	Rollout Throughput	+35%

导出到 Google 表格
Frequently Asked Questions (FAQ)
Q: Will APRIL affect policy purity and convergence?

A: It will definitely have an impact on policy purity; the proportion of off-policy tokens in one round is about 40%. However, from both an engineering and experimental perspective, partial rollout has not introduced significant instability under the current settings. Further verification is needed for tasks with a much larger max_response_length (e.g., agent tasks, multi-turn tasks).

Q: Are changes to the decoding kernel required?

A: No. APRIL operates at the system scheduling layer and does not conflict with inference acceleration techniques like speculative decoding or continuous batching. Instead, they are complementary and can be stacked.

Directory Structure
APRIL/
├── scripts/
│   └── partial_rollout/
│       ├── deepseek/               # Experiment code for deepseek-r1-distill-1.5B
│       └── qwen/                   # Experiment code for qwen3-4B
├── slime/
│   ├── backends/
│   ├── rollout/
│   │   └── sglang_example.py       # Core sampling code
│   ├── ray/                      # Core scheduling logic
│   │   └── buffer.py             # Buffer implementation code
│   └── utils/
└── tools/                        # Megatron format conversion tools

Paper
(TODO: arXiv link for the paper)