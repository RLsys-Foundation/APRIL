## Build Docker

```
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name slime_yuzhen \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -w $PWD \
  rlsys/slime_bkp:latest
  /bin/bash
```

## Clone Repo

```
git clone https://github.com/zyzshishui/slime_.git
```

## Setup Environment

```
vim ~/.bashrc
export PYTHONPATH=/workspace/Megatron-LM-amd_version
export WANDB_API_KEY="cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e"

source ~/.bashrc
```

## Download Models

```
# Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
huggingface-cli download guapisolo/Qwen3-4B-torch --local-dir /root/Qwen3-4B_torch

# DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download zyzshishui0627/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist --local-dir /root/DeepSeek-R1-Distill-Qwen-1.5B_torch
```

## Download Datas

```
# dapo
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# DeepScaler
huggingface-cli download --repo-type dataset  zyzshishui0627/DeepScaleR-openai-format --local-dir /root/DeepScaleR

# DeepMath
huggingface-cli download --repo-type dataset zyzshishui0627/DeepMath-103K-openai-format --local-dir /root/DeepMath-103K

# aime(eval)
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

## Run Script

for example:

```
nohup bash scripts/partial_rollout/qwen/grpo/run-qwen3-4B-dapo.sh > qwen3-4B-dapo.out
```

