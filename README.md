# GRPO with Length Penalty on Qwen3-1.7B

This repo trains [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) with **Group Relative Policy Optimization (GRPO)** plus a configurable **length penalty**, using a GDPO-style multi-objective reward setup. The goal is to study how length pressure affects **chain-of-thought (CoT) legibility** in reasoning models — whether penalizing verbosity degrades the quality or interpretability of `<think>...</think>` traces.

Training uses **LoRA** (rank-16 by default) so everything fits on a single GPU, and **vLLM** in colocate mode for fast generation.

## Quick start

See [SETUP.md](SETUP.md) for full environment setup (pip installs, version pinning, model download).

```bash
# 1. Run preflight checks (verifies tokenizer, vLLM, GPU, wandb, etc.)
python preflight.py

# 2. Trial run (fast sanity check, ~3 steps)
python grpo_train.py --lambda_length 0.0 --max_steps 3 --num_generations 4 --max_completion_length 512

# 3. Full training run with length penalty
python grpo_train.py --lambda_length 0.3

# 4. Evaluate base model or a checkpoint
python baseline_eval.py --min_solved_pct 50 --max_solved_pct 95
python baseline_eval.py --checkpoint_path ./outputs/<run>/checkpoint-100
```

## Files

| File | Purpose |
|---|---|
| `grpo_train.py` | Main entry point. Loads model/tokenizer/dataset, builds the `GRPOTrainer`, runs training, saves the final LoRA adapter. Contains monkey-patches for vLLM/transformers compatibility. |
| `config.py` | `ExperimentConfig` dataclass with all hyperparameters. Builds `GRPOConfig` and `LoraConfig` from CLI args. |
| `rewards.py` | Two reward functions (`CorrectnessRewardFunction`, `LengthPenaltyRewardFunction`) plus `RewardLogger` for JSONL/W&B logging. Correctness uses `math-verify` for symbolic equivalence. |
| `data.py` | Loads `lime-nlp/deepscaleR_difficulty`, filters by solved-percentage difficulty band, formats prompts in Qwen3 chat format (`<\|im_start\|>user ... <\|im_end\|> <\|im_start\|>assistant`). |
| `preflight.py` | 9-phase pre-flight check suite. Tests environment, logic, math-verify, tokenizer, vLLM/HF parity, model config, wandb, GPU generation, and TRL config. Run this first on every new instance. |
| `baseline_eval.py` | Generates completions for a fixed problem sample and scores them. Supports vLLM batched inference or HF generate, with optional LoRA checkpoint loading. |
| `logging_utils.py` | `StepSyncCallback` (per-step timing, GPU memory, ETA) and JSONL helpers. |

## Architecture

### Multi-objective reward (normalize-then-sum)

The single most important design choice. TRL's `GRPOTrainer` receives **two separate reward functions**:

1. **Correctness** — binary 0/1 per completion (does `\boxed{...}` match the ground truth?)
2. **Length penalty** — `-(num_tokens / max_completion_length)`, always in `[-1, 0]`

TRL normalizes each signal independently within each group (z-scores), then combines them:

```
combined = 1.0 * z(correctness) + lambda * z(length_penalty)
```

This is set via `multi_objective_aggregation="normalize_then_sum"` and `reward_weights=[1.0, lambda_length]` in the `GRPOConfig`.

**Why not a single entangled reward?** In the naive setup `r = correctness + lambda * length_penalty`, GRPO's group normalization z-scores the combined reward. Because correctness variance dominates, changes to lambda in the range 0.1–2.0 produce nearly identical training dynamics — the length signal gets washed out. Separating the signals and normalizing them independently (the GDPO approach) ensures lambda actually controls the strength of the length pressure.

### Training flow

1. `config.py` builds a `GRPOConfig` with all hyperparameters
2. `grpo_train.py` loads tokenizer + dataset, creates both reward functions and a `RewardLogger` they share
3. `GRPOTrainer` runs in vLLM colocate mode: vLLM handles generation on-GPU, then LoRA adapter weights are trained
4. At each step, `CorrectnessRewardFunction` runs first (buffers data on the logger), then `LengthPenaltyRewardFunction` runs and triggers the full log flush
5. `StepSyncCallback` tracks timing and GPU memory
6. Final LoRA adapter is saved to `<output_dir>/<run_name>/final_adapter/`

### Logging

Three JSONL files are written to `<run_dir>/logs/`:
- `rollouts.jsonl` — per-completion: text, correctness, length penalty, token count
- `prompt_stats.jsonl` — per-prompt: accuracy, reward stats, advantages
- `step_stats.jsonl` — per-step: accuracy, mean length, timing, GPU memory

All metrics are also logged to W&B under the `custom/` and `perf/` namespaces.

## Key design decisions

### Why LoRA excludes embeddings
Qwen3-1.7B has `tie_word_embeddings=True` — the input embedding matrix and the output `lm_head` are the same tensor. Targeting `embed_tokens` or `lm_head` in LoRA would break this tying, so LoRA targets only attention projections (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`).

### Why no BOS token
Qwen3's tokenizer has `bos_token=null` and `add_bos_token=false`. Prompts start directly with `<|im_start|>user`. The preflight checks verify this — a spurious BOS would shift all token positions and break generation.

### Stop tokens
Generation stops at `<|im_end|>` (151645) or the true EOS token (151643). These are passed as `stop_token_ids` in both training and eval configs.

### Monkey-patches
Two monkey-patches in `grpo_train.py` fix compatibility issues:

1. **`all_special_tokens_extended`** — vLLM 0.10.x accesses `tokenizer.all_special_tokens_extended`, which was removed in transformers 5.x. The patch falls back to `all_special_tokens`.
2. **`max_num_batched_tokens`** — TRL hardcodes `max_num_batched_tokens=4096` when initializing vLLM, but our `max_model_len` can be larger (e.g. 1024 + 8192 = 9216). The patch bumps `max_num_batched_tokens` to match `max_model_len`.

### vLLM V0 engine
`VLLM_USE_V1=0` is set before importing vLLM. The V1 engine has `torch.compile` bugs when using `model_impl="transformers"`.

### Loss type
`dr_grpo` (DeepSeek's variant of GRPO) is the default loss type.

### Difficulty filtering
Problems are filtered from `lime-nlp/deepscaleR_difficulty` by `solved_percentage` (default 50–95%). This keeps problems in the "sweet spot" — hard enough to provide learning signal but not so hard the model never gets them right.

## Configuration

### Training (`grpo_train.py`)

| Argument | Default | Description |
|---|---|---|
| `--lambda_length` | (required) | Length penalty weight (0.0 = no penalty, 0.3 = moderate, 1.0+ = aggressive) |
| `--max_steps` | 150 | Total training steps |
| `--num_generations` | 8 | Rollouts per prompt per step |
| `--max_completion_length` | 8192 | Max tokens per completion |
| `--max_answer_tokens` | 0 | Max tokens after `</think>` for correctness check (0 = unlimited) |
| `--learning_rate` | 5e-6 | LoRA learning rate (10x typical full fine-tuning) |
| `--lora_rank` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--loss_type` | `dr_grpo` | GRPO loss variant |
| `--temperature` | 0.6 | Sampling temperature |
| `--model_name` | `Qwen/Qwen3-1.7B` | Base model (HF name or local path) |
| `--min_solved_pct` | 50.0 | Difficulty filter lower bound |
| `--max_solved_pct` | 95.0 | Difficulty filter upper bound |
| `--init_checkpoint` | `""` | LoRA checkpoint to merge into base before training |
| `--skip_first_n_prompts` | 0 | Skip N prompts from prior run (same seed order) |
| `--beta` | 0.0 | KL penalty coefficient (0 = disabled) |
| `--no_vllm` | flag | Disable vLLM, use HF generate (for debugging) |

### Evaluation (`baseline_eval.py`)

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_path` | `None` | Path to LoRA adapter directory |
| `--num_problems` | 100 | Number of problems to evaluate |
| `--num_generations` | 8 | Completions per problem |
| `--max_new_tokens` | 8192 | Max tokens per completion |
| `--no_vllm` | flag | Use HF generate instead of vLLM |
| `--min_solved_pct` | 50.0 | Difficulty filter lower bound |
| `--max_solved_pct` | 95.0 | Difficulty filter upper bound |
