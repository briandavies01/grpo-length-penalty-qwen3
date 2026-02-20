# Environment Setup

Tested on RunPod with an **NVIDIA H100 80GB** and **Python 3.12**. Other GPUs with 40GB+ VRAM and CUDA 12.x should work.

## 1. Install PyTorch (CUDA 12.8)

Install PyTorch first, pinned to the version vLLM expects:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## 2. Install vLLM (pinned by TRL compatibility)

**vLLM 0.10.2 is required.** This is constrained by TRL, not our code — TRL 0.28.0 only supports vLLM 0.10.2, 0.11.0–0.11.2, and 0.12.0. Our code additionally sets `VLLM_USE_V1=0` to force the V0 engine (which avoids `torch.compile` bugs with the `model_impl="transformers"` backend on 0.10.x), but the primary version constraint comes from TRL's internal vLLM integration. Upgrading to vLLM 0.15+ would require a newer TRL version and likely changes to `config.py`'s `GRPOConfig` kwargs.

Install vLLM **before** other HF packages to avoid dependency conflicts:

```bash
pip install vllm==0.10.2
```

This pulls in many CUDA/nvidia dependencies automatically. You may see warnings about `setuptools` or `cuda-bindings` version mismatches — these are harmless.

## 3. Install training dependencies

```bash
pip install transformers==5.2.0 trl==0.28.0 peft==0.18.1 accelerate==1.12.0 datasets==4.5.0
pip install wandb==0.25.0 math-verify==0.9.0
```

## 4. Install flash-attn (optional but recommended)

```bash
pip install flash-attn==2.8.3
```

If this fails to build (common on some CUDA setups), training will still work — the code falls back to default attention automatically.

## 5. Download the model locally

**This is a critical step.** The training scripts and preflight checks expect the model to be available locally. If you skip this, `preflight.py` will fail during the vLLM tokenization and GPU generation phases, and `grpo_train.py` will be much slower (or fail on machines without reliable internet).

```bash
# Download Qwen3-1.7B to the HuggingFace cache
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
print('Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype='auto')
print('Done — model is cached locally.')
"
```

Alternatively, download to a specific directory and pass `--model_name /path/to/local/model` to all scripts:

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./qwen3-1.7b
```

## 6. Log in to Weights & Biases

```bash
wandb login
```

Or set the environment variable:

```bash
export WANDB_API_KEY=your_key_here
```

## 7. Run preflight checks

```bash
python preflight.py
```

This runs 9 phases of verification: environment, pure logic, math-verify, tokenizer, vLLM/HF parity, model config, wandb, GPU generation, and TRL config. Fix any `[FAIL]` items before training.

If you want to skip slow GPU tests while debugging setup issues:

```bash
python preflight.py --skip-gpu --skip-vllm
```

## Version summary

These are the exact versions tested together on RunPod (H100, CUDA 12.8, Python 3.12.3):

| Package | Version |
|---|---|
| torch | 2.8.0+cu128 |
| transformers | 5.2.0 |
| trl | 0.28.0 |
| peft | 0.18.1 |
| accelerate | 1.12.0 |
| datasets | 4.5.0 |
| vllm | 0.10.2 |
| flash-attn | 2.8.3 |
| wandb | 0.25.0 |
| math-verify | 0.9.0 |
| tokenizers | 0.22.2 |
| safetensors | 0.7.0 |
| triton | 3.5.1 |

## Common issues

**`transformers` corruption after vLLM install** — vLLM sometimes leaves transformers in a broken state (`Cannot uninstall transformers None`). Fix with:
```bash
pip install --force-reinstall --no-deps transformers==5.2.0
```

**torch version mismatch warnings** — vLLM 0.10.2 pins `torch==2.8.0`. If pip resolves a different version, force it:
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**vLLM version too new** — TRL 0.28.0 only supports vLLM 0.10.2–0.12.0. If you install a newer version, TRL's vLLM integration will break. Downgrade: `pip install vllm==0.10.2`.

**`all_special_tokens_extended` AttributeError** — This is a vLLM 0.10.x + transformers 5.x incompatibility. The training scripts monkey-patch it automatically; for standalone vLLM usage, either downgrade transformers or apply the same patch.

**RunPod `--break-system-packages`** — On RunPod instances where pip is system-managed, you may need to add `--break-system-packages` to all pip commands.
