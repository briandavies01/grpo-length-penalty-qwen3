"""Main entry point for GRPO training of Qwen3 with GDPO-style length penalty and LoRA.

Uses normalize_then_sum multi-objective aggregation: correctness and length
penalty are separate reward functions, each z-scored independently within
groups, then combined with reward_weights=[1.0, lambda_length]. This prevents
group normalization from canceling out lambda (the root cause of lambda 0.1-2.0
producing identical training dynamics in the entangled single-reward setup).

Uses LoRA (Low-Rank Adaptation) to fit training on a single GPU.

Usage:
    # Full training run
    python grpo_train.py --lambda_length 0.3

    # Quick trial run (sanity check)
    python grpo_train.py --lambda_length 0.0 --max_steps 3 --num_generations 4 --max_completion_length 512

    # Without vLLM (debugging)
    python grpo_train.py --lambda_length 0.0 --no_vllm --max_steps 3
"""

import json
import os

# Force vLLM V0 engine — V1 has torch.compile bugs with model_impl="transformers"
os.environ["VLLM_USE_V1"] = "0"

import torch
import wandb
from peft import PeftModel

# Monkey-patch for vLLM 0.10.x + transformers 5.x compatibility:
# vLLM accesses tokenizer.all_special_tokens_extended which was removed in
# transformers 5.x TokenizersBackend. Patch it to fall back gracefully.
import transformers.tokenization_utils_base as _tub
_orig_getattr = _tub.PreTrainedTokenizerBase.__getattr__
def _patched_getattr(self, key):
    if key == "all_special_tokens_extended":
        return self.all_special_tokens
    return _orig_getattr(self, key)
_tub.PreTrainedTokenizerBase.__getattr__ = _patched_getattr

# Monkey-patch TRL's hardcoded max_num_batched_tokens=4096 in vLLM init.
# When max_model_len > 4096 (e.g. 1024 prompt + 5000 completion = 6024),
# vLLM rejects it. Patch vLLM's LLM.__init__ to fix the kwarg.
from vllm.entrypoints.llm import LLM as _VllmLLM
_orig_llm_init = _VllmLLM.__init__
def _patched_llm_init(self, *args, **kwargs):
    mml = kwargs.get('max_model_len')
    mnbt = kwargs.get('max_num_batched_tokens')
    if mml and mnbt and mnbt < mml:
        kwargs['max_num_batched_tokens'] = mml
    return _orig_llm_init(self, *args, **kwargs)
_VllmLLM.__init__ = _patched_llm_init

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer

from config import ExperimentConfig, parse_args
from data import load_and_format_dataset, verify_prompt_tokenization
from logging_utils import StepSyncCallback
from rewards import RewardLogger, CorrectnessRewardFunction, LengthPenaltyRewardFunction


def merge_checkpoint_into_base(model_name: str, checkpoint_path: str, save_dir: str) -> str:
    """Load base model + LoRA checkpoint, merge, save to disk.

    Returns the path to the saved merged model. This merged model can then
    be used as the base for a new round of LoRA training.
    """
    print(f"Merging LoRA checkpoint into base model...")
    print(f"  Base model: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Save to: {save_dir}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to avoid GPU memory issues
    )

    # Load LoRA adapter on top
    model_with_lora = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Merge LoRA weights into base model weights and discard adapter
    merged_model = model_with_lora.merge_and_unload()

    # Save merged model + tokenizer so vLLM/TRL can load it from disk
    os.makedirs(save_dir, exist_ok=True)
    merged_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # Free memory
    del merged_model, model_with_lora, base_model
    torch.cuda.empty_cache()

    print(f"  Merged model saved ({save_dir})")
    return save_dir


def main():
    config = parse_args()

    # --- Build LoRA config ---
    lora_config = config.build_lora_config()

    # --- Build run directory ---
    run_name = config.get_run_name()
    run_dir = os.path.join(config.output_dir, run_name)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # --- Save config ---
    config_path = os.path.join(run_dir, "config.json")
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith("_")
    }
    # Also save LoRA config details
    config_dict["lora_config"] = {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": list(lora_config.target_modules),
        "bias": lora_config.bias,
        "task_type": str(lora_config.task_type),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    # --- Initialize W&B ---
    wandb.init(
        project=getattr(config, "wandb_project", "grpo-length-penalty"),
        name=run_name,
        config=config_dict,
    )

    # --- Merge checkpoint if continuing from a previous run ---
    if config.init_checkpoint:
        merged_dir = os.path.join(run_dir, "merged_init")
        model_path = merge_checkpoint_into_base(
            config.model_name, config.init_checkpoint, merged_dir
        )
    else:
        model_path = config.model_name

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Qwen3 has no BOS token (add_bos_token=false by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Load dataset ---
    print(f"Loading dataset: {config.dataset_name}")
    train_dataset = load_and_format_dataset(
        dataset_name=config.dataset_name,
        min_solved_pct=config.min_solved_pct,
        max_solved_pct=config.max_solved_pct,
    )
    print(f"Dataset size: {len(train_dataset)} problems")

    # --- Skip prompts seen in prior training ---
    if config.skip_first_n_prompts > 0:
        n_skip = config.skip_first_n_prompts
        n_total = len(train_dataset)

        if n_skip >= n_total:
            raise ValueError(
                f"skip_first_n_prompts ({n_skip}) >= dataset size ({n_total}). "
                f"Cannot skip more prompts than exist in the dataset."
            )

        # Replicate TRL's RepeatRandomSampler ordering: it uses
        # torch.randperm(N, generator=Generator(seed)) to determine
        # the order of unique prompts. We identify the first N unique
        # indices and remove them from the dataset.
        g = torch.Generator()
        g.manual_seed(config.seed)
        perm = torch.randperm(n_total, generator=g)
        seen_indices = set(perm[:n_skip].tolist())
        keep_indices = [i for i in range(n_total) if i not in seen_indices]
        train_dataset = train_dataset.select(keep_indices)
        print(f"Skipped {len(seen_indices)} prompts from prior training, "
              f"{len(train_dataset)} remain (was {n_total})")

    # --- Verify prompt tokenization (fail-fast on BOS/special token issues) ---
    verify_prompt_tokenization(tokenizer, train_dataset[0]["prompt"])

    # --- Build GRPOConfig ---
    grpo_config = config.build_grpo_config()
    # Override run_name to match what we computed
    grpo_config.run_name = run_name
    grpo_config.output_dir = run_dir

    # --- Create reward functions (GDPO: two separate signals) ---
    reward_logger = RewardLogger(
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        lambda_length=config.lambda_length,  # For logging only; TRL uses reward_weights
        log_dir=log_dir,
    )
    correctness_fn = CorrectnessRewardFunction(
        logger=reward_logger,
        max_answer_tokens=config.max_answer_tokens,
    )
    length_fn = LengthPenaltyRewardFunction(logger=reward_logger)

    # --- Create callback for cleanup ---
    step_sync_callback = StepSyncCallback(reward_logger)

    # --- Create GRPOTrainer ---
    print(f"Initializing GRPOTrainer...")
    print(f"  Lambda: {config.lambda_length}")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Multi-objective: {grpo_config.multi_objective_aggregation}")
    print(f"  Reward weights: {grpo_config.reward_weights}")
    print(f"  Num generations: {config.num_generations}")
    print(f"  Max completion length: {config.max_completion_length}")
    print(f"  Max answer tokens: {config.max_answer_tokens} {'(unlimited)' if config.max_answer_tokens == 0 else ''}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA: rank={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"targets={list(lora_config.target_modules)}")
    print(f"  Difficulty filter: solved_pct in [{config.min_solved_pct}, {config.max_solved_pct}]")
    if config.init_checkpoint:
        print(f"  Init checkpoint: {config.init_checkpoint} (merged into base)")
    print(f"  vLLM: {config.use_vllm} (mode: {config.vllm_mode})")
    print(f"  Output dir: {run_dir}")

    if config.use_vllm:
        # vLLM colocate mode: pass model path as string, TRL loads internally
        # model_path is either the HF model name or path to merged checkpoint
        trainer = GRPOTrainer(
            model=model_path,
            reward_funcs=[correctness_fn, length_fn],
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            callbacks=[step_sync_callback],
        )
    else:
        # HF generate fallback: load model ourselves, let TRL wrap with LoRA
        print(f"Loading model (no vLLM): {model_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            print("  flash-attn not available, falling back to default attention")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[correctness_fn, length_fn],
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            callbacks=[step_sync_callback],
        )

    # --- Load explicit reference model for KL penalty ---
    if config.beta > 0 and config.ref_model_name:
        from trl.trainer.utils import disable_dropout_in_model
        print(f"Loading explicit reference model for KL: {config.ref_model_name}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        ref_model.eval()
        disable_dropout_in_model(ref_model)
        trainer.ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
        print(f"  Reference model loaded on {next(trainer.ref_model.parameters()).device}")
    elif config.beta > 0:
        print(f"  KL beta={config.beta} with PEFT adapter-disable reference (merged_init as base)")

    # --- Log trainable parameters ---
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # --- Train ---
    print(f"\nStarting GRPO training...")
    trainer.train()

    # --- Save final adapter ---
    # With LoRA, save_model saves only the adapter weights (small, ~50MB)
    # To get the full model later: load base model + load adapter + merge
    final_model_dir = os.path.join(run_dir, "final_adapter")
    trainer.save_model(final_model_dir)
    print(f"\nFinal LoRA adapter saved to {final_model_dir}")
    print(f"  Base model for this adapter: {model_path}")
    if config.init_checkpoint:
        print(f"  (merged base = {config.model_name} + {config.init_checkpoint})")
        print(f"  For eval, use --model_name with the merged_init dir in this run")

    # --- Cleanup ---
    reward_logger.close()
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
