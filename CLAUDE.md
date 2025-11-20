# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

EasyR1 is a **reinforcement learning (RL) training framework** for vision-language models (VLMs), forked from [veRL](https://github.com/volcengine/verl). It enables efficient multi-modal RL training using HybridEngine architecture and vLLM's SPMD mode.

**Key capabilities:**
- RL algorithms: GRPO, DAPO, Reinforce++, ReMax, RLOO, GSPO, CISPO
- Models: Qwen2-VL/Qwen2.5-VL/Qwen3-VL, Llama3, Qwen2/2.5/3, DeepSeek-R1
- Features: Padding-free training, multi-node support via Ray, checkpoint resumption

## Environment Setup

### Using Docker (Recommended)

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

### Manual Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

**Python environment:**
- Activate venv: `source .venv/bin/activate`
- Requirements: Python 3.9+, transformers>=4.54.0, flash-attn>=2.4.3, vllm>=0.8.3

### Environment Variables

```bash
export HF_ENDPOINT=https://hf-mirror.com  # If Hugging Face is blocked
export USE_MODELSCOPE_HUB=1              # Use ModelScope hub instead
```

## Running Training

### Basic GRPO Training

```bash
# Example: Qwen2.5-VL-7B on Geometry3K
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

Training scripts follow this pattern:

```bash
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=my_experiment \
    trainer.n_gpus_per_node=8
```

### Multi-Node Training

```bash
# On head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker nodes
ray start --address=<head_node_ip>:6379

# Check Ray cluster
ray status

# Run training (on head node only)
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

### Merge Checkpoints

After training, convert to Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## Configuration System

### Main Config File: `examples/config.yaml`

Key sections:
- **`data`**: Dataset paths, batch sizes, prompt/response lengths, image parameters
- **`algorithm`**: RL algorithm (grpo/dapo/etc), KL penalty, advantage estimator
- **`worker.actor`**: Model path, FSDP settings, optimizer config, gradient checkpointing
- **`worker.rollout`**: Sampling parameters (n, temperature, top_p), vLLM settings (gpu_memory_utilization, tensor_parallel_size)
- **`worker.reward`**: Reward function path (e.g., `./examples/reward_function/math.py:compute_score`)
- **`trainer`**: Epochs, validation/save frequency, logger (wandb/swanlab/tensorboard), GPU counts

### Config Override via CLI

```bash
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=/path/to/model \
    trainer.total_epochs=20 \
    worker.rollout.temperature=0.8
```

### Custom Datasets

Format requirements (see dataset examples on Hugging Face):
- Text: `hiyouga/math12k`
- Image-text: `hiyouga/geometry3k`
- Multi-image: `hiyouga/journeybench-multi-image-vqa`

Dataset structure:
- `prompt_key`: Input field name (default: "problem")
- `answer_key`: Ground truth field (default: "answer")
- `image_key`: Image field (default: "images")
- `format_prompt`: Jinja template path for prompt formatting

## Code Architecture

### Core Components

```
verl/
├── trainer/           # Main training loop and orchestration
├── workers/           # Distributed worker implementations
│   ├── actor/         # Actor model (policy network being trained)
│   ├── rollout/       # Rollout worker (vLLM-based generation)
│   ├── reward/        # Reward computation
│   └── ref/           # Reference model (for KL penalty)
├── models/            # Model implementations and wrappers
├── single_controller/ # Ray-based distributed coordination
└── utils/             # Utilities (FSDP, data loading, etc.)
```

### HybridEngine Architecture

EasyR1 uses **HybridEngine** for efficient RL training:
1. **Rollout Phase**: vLLM generates responses (fast inference)
2. **Training Phase**: FSDP trains actor model (gradient updates)
3. **Reference Model**: Computes KL divergence (optional CPU offload)

Key insight: Rollout and training use different backends optimized for their tasks.

### Worker Parallelism

- **Actor**: FSDP with optional CPU offload (`worker.actor.fsdp`, `worker.actor.offload`)
- **Rollout**: Tensor parallelism via vLLM (`worker.rollout.tensor_parallel_size`)
- **Reference**: FSDP with aggressive CPU offload to save GPU memory

## Validation and Analysis

### Directory: `validation/`

#### 1. Inference with Sampling

```bash
cd validation
bash run_sampling_inference.sh
```

Generates multi-sample predictions for analysis.

#### 2. Analysis Tools: `validation/analysis/`

**Accuracy Analysis:**
```bash
python analyze_sampling_results.py \
    --sampling_file /path/to/sampling_results.jsonl \
    --data_dir /path/to/test_data \
    --agent_type qwen3vl
```

Computes Pass@1, Pass@k, average accuracy.

**InfoGain Calculation (Multi-GPU):**
```bash
# Uses 4 GPUs with DataLoader multi-threading
bash _run.sh

# Or run directly:
python calculate_infogain_multiGPU.py \
    --sampling_file /path/to/sampling_results.jsonl \
    --data_dir /path/to/test_data \
    --model_path /path/to/model \
    --num_gpus 4 \
    --mini_batch_size 8 \
    --num_workers 4
```

**InfoGain Implementation Details:**
- Uses PyTorch Dataset/DataLoader for parallel image loading
- Each GPU process loads complete model (data parallelism, not model parallelism)
- Results saved incrementally to JSON files (avoids multiprocessing queue deadlock)
- Handles `CUDA_VISIBLE_DEVICES` to map logical/physical GPU IDs correctly
- Key fix: Use `device_map=<specific_device>` not `device_map="auto"` to prevent uneven GPU memory allocation

**Architecture:**
1. **Stage 1**: Preprocess all samples (single-threaded)
2. **Stage 2**: Distribute samples to N GPU workers (multiprocessing)
3. Each worker:
   - Creates Dataset with multi-threaded image loading (num_workers=4)
   - Loads model on assigned GPU
   - Runs batch inference with DataLoader
   - Writes results to `gpu_<id>_results.json`
4. Main process collects results from JSON files and computes final InfoGain

#### 3. AgentNetBench Evaluation

```bash
cd validation/agentnetbench
# See validation/agentnetbench/README.md for details
```

## Development Commands

### Code Quality

```bash
make style      # Auto-format with ruff
make quality    # Check code quality
make license    # Check license headers
make test       # Run pytest
make commit     # Run pre-commit hooks
```

### Build

```bash
make build      # Build sdist and wheel
```

## Common Issues

**"Image features and image tokens do not match"**
- Increase `data.max_prompt_length` or reduce `data.max_pixels`

**"CUDA Error: out of memory"**
- Reduce `worker.rollout.gpu_memory_utilization`
- Enable `worker.actor.offload.offload_params=true`

**"0 active drivers"**
- Uninstall `deepspeed` from the environment

**Multi-GPU InfoGain: All models load on one GPU**
- Ensure using `device_map=<specific_device>` not `device_map="auto"`
- Check `CUDA_VISIBLE_DEVICES` is correctly parsed into physical GPU IDs

**Multi-GPU InfoGain: Deadlock during result collection**
- Use file-based result collection, not `mp.Queue()`
- Each worker writes to `temp_dir/gpu_{id}_results.json`

## Architecture Notes for VLM Training

### Vision Tower Handling

- Set `worker.actor.model.freeze_vision_tower=false` to train vision encoder
- For VLMs, `data.min_pixels` and `data.max_pixels` control image resolution
- Use `data.limit_images=0` in rollout to include all images (for multi-image tasks)

### Padding-Free Training

- `worker.actor.padding_free=true` enables efficient variable-length training
- Reduces memory and compute waste from padding tokens
- Especially beneficial for long vision-text sequences

### Memory Optimization

Critical settings for large VLMs:
- `worker.actor.offload.offload_params=true` (CPU offload for params)
- `worker.actor.offload.offload_optimizer=true` (CPU offload for optimizer states)
- `worker.rollout.gpu_memory_utilization=0.6` (limit vLLM memory)
- `worker.actor.model.enable_gradient_checkpointing=true` (trade compute for memory)

### BF16 Training

For memory-constrained scenarios:
```bash
worker.actor.fsdp.torch_dtype=bf16 \
worker.actor.optim.strategy=adamw_bf16
```

## Important Implementation Details

### DataLoader in Multi-GPU Scripts

The `calculate_infogain_multiGPU.py` uses PyTorch DataLoader with:
- **num_workers**: Parallel data loading threads per GPU (default: 4)
- **pin_memory=True**: Faster CPU→GPU transfer
- **prefetch_factor=2**: Pre-load batches to reduce GPU waiting

Benefits:
- 4x faster image I/O with 4 workers
- CPU-GPU pipeline overlap (load next batch while GPU computes)

### GPU Allocation Logic

When using `CUDA_VISIBLE_DEVICES=4,5,6,7`:
1. Main process reads env var, extracts physical GPU IDs: `[4, 5, 6, 7]`
2. Spawns workers with physical IDs: Worker 0 gets GPU 4, Worker 1 gets GPU 5, etc.
3. Each worker:
   - Computes logical GPU ID from `CUDA_VISIBLE_DEVICES` list index
   - Loads model with `device_map=f"cuda:{logical_id}"` (NOT `device_map="auto"`)

Critical: Using `device_map="auto"` causes model parallelism across all visible GPUs, leading to uneven memory usage.

### Avoiding Deadlocks

Multiprocessing with `mp.Queue()` can deadlock if:
- Queue buffer fills up before workers finish
- Workers block on `queue.put()` while main process blocks on `queue.get()`

Solution: File-based result collection:
```python
# Worker writes to file
with open(output_dir / f"gpu_{gpu_id}_results.json", "w") as f:
    json.dump(results, f)

# Main process reads after workers complete
for gpu_id in range(num_gpus):
    with open(output_dir / f"gpu_{gpu_id}_results.json") as f:
        results = json.load(f)
```
