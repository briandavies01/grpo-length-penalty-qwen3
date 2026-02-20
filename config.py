"""Hyperparameters and GRPOConfig construction for Qwen3 GRPO length penalty experiments."""

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime

from peft import LoraConfig, TaskType
from trl import GRPOConfig


@dataclass
class ExperimentConfig:
    """All experiment-level hyperparameters."""

    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    init_checkpoint: str = ""  # Optional: LoRA checkpoint to merge into base before training
    ref_model_name: str = ""  # Optional: explicit reference model for KL (overrides PEFT adapter-disable)
    skip_first_n_prompts: int = 0  # Skip prompts seen in prior training (from same seed)

    # Dataset
    dataset_name: str = "lime-nlp/deepscaleR_difficulty"

    # Length penalty
    lambda_length: float = 0.0

    # Training
    max_steps: int = 150
    per_device_train_batch_size: int = 0  # 0 = auto-set to num_generations
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6  # 10x higher than full fine-tuning (standard for LoRA)
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "constant_with_warmup"
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    # GRPO-specific
    num_generations: int = 8
    max_completion_length: int = 8192
    max_prompt_length: int = 1024
    max_answer_tokens: int = 0  # 0 = no limit; >0 = only check correctness within N tokens after </think>
    temperature: float = 0.6
    top_p: float = 0.95
    loss_type: str = "dr_grpo"
    beta: float = 0.0
    num_iterations: int = 1
    epsilon: float = 0.2
    scale_rewards: str = "group"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # All attention projections
        "gate_proj", "up_proj", "down_proj",       # All MLP layers
    ])

    # vLLM
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.2
    vllm_model_impl: str = "transformers"  # Use HF kernels inside vLLM (fixes generation quality)

    # Logging
    logging_steps: int = 1
    log_completions: bool = False
    report_to: str = "wandb"

    # Checkpoints
    save_steps: int = 50
    save_total_limit: int = 3

    # Output
    output_dir: str = "./outputs"
    run_name: str = ""

    # Reproducibility
    seed: int = 42

    # Difficulty filtering
    min_solved_pct: float = 50.0
    max_solved_pct: float = 95.0

    # Baseline eval
    baseline_num_problems: int = 100
    baseline_generations_per_problem: int = 8

    def get_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"lambda_{self.lambda_length}_{timestamp}"

    def build_lora_config(self) -> LoraConfig:
        """Construct a LoraConfig for PEFT."""
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_target_modules,
        )

    def build_grpo_config(self) -> GRPOConfig:
        """Construct a GRPOConfig from these experiment params."""
        run_name = self.get_run_name()
        run_output_dir = os.path.join(self.output_dir, run_name)

        # batch_size = num_generations so all rollouts for a prompt are
        # generated in one shot, then one gradient step.
        batch = self.per_device_train_batch_size if self.per_device_train_batch_size > 0 else self.num_generations
        grad_accum = self.gradient_accumulation_steps
        gen_batch = batch * grad_accum
        G = self.num_generations
        if gen_batch % G != 0:
            print(f"[config] WARNING: generation_batch_size ({gen_batch}) not divisible by "
                  f"num_generations ({G}). Setting per_device_train_batch_size={G}.")
            batch = G

        return GRPOConfig(
            output_dir=run_output_dir,
            run_name=run_name,
            # Training schedule
            max_steps=self.max_steps,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            # GRPO-specific
            num_generations=self.num_generations,
            max_completion_length=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            loss_type=self.loss_type,
            beta=self.beta,
            num_iterations=self.num_iterations,
            epsilon=self.epsilon,
            scale_rewards=self.scale_rewards,
            # GDPO: normalize each reward signal independently, then combine
            # This prevents group normalization from canceling out lambda
            multi_objective_aggregation="normalize_then_sum",
            reward_weights=[1.0, self.lambda_length],
            # vLLM
            use_vllm=self.use_vllm,
            vllm_mode=self.vllm_mode,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_max_model_length=self.max_prompt_length + self.max_completion_length,
            vllm_model_impl=self.vllm_model_impl,
            # vLLM importance sampling: use per-token mode to avoid sequence-level
            # ratio compounding (exp of sum over ~1700 tokens kills gradients)
            vllm_importance_sampling_mode="token_truncate",
            # Pass EOS as stop token for vLLM
            generation_kwargs={"stop_token_ids": [151645, 151643]},
            # Logging
            logging_steps=self.logging_steps,
            log_completions=self.log_completions,
            report_to=self.report_to,
            # Checkpoints
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            # Important: keep dataset columns (e.g. 'solution') for reward function
            remove_unused_columns=False,
            # Reproducibility
            seed=self.seed,
        )


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments into an ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="GRPO training with length penalty for CoT monitorability"
    )

    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-1.7B",
                        help="Base model name or path")
    parser.add_argument(
        "--lambda_length", type=float, required=True,
        help="Length penalty coefficient (e.g., 0.0, 0.1, 0.3, 1.0, 3.0)"
    )
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=8192)
    parser.add_argument("--max_answer_tokens", type=int, default=0,
                        help="Max tokens after </think> for correctness check (0=unlimited, 150 recommended)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=0,
                        help="0 = auto-set to num_generations (recommended)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--loss_type", type=str, default="dr_grpo")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--min_solved_pct", type=float, default=50.0,
                        help="Min solved_percentage for difficulty filter (default 50)")
    parser.add_argument("--max_solved_pct", type=float, default=95.0,
                        help="Max solved_percentage for difficulty filter (default 95)")
    parser.add_argument("--skip_first_n_prompts", type=int, default=0,
                        help="Skip the first N unique prompts from the dataset (matching "
                             "the order the prior run would have seen them with the same seed). "
                             "Use to avoid re-training on prompts from a prior run.")
    parser.add_argument("--init_checkpoint", type=str, default="",
                        help="Path to LoRA checkpoint to merge into base model before training. "
                             "Creates a fresh LoRA adapter on top of the merged model.")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0=disabled)")
    parser.add_argument("--ref_model_name", type=str, default="",
                        help="Explicit reference model for KL (e.g. original base model). "
                             "If empty, TRL uses adapter-disable for PEFT models.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max number of checkpoints to keep")
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM, use HF generate")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb_project", type=str, default="grpo-length-penalty",
        help="Weights & Biases project name"
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model_name,
        init_checkpoint=args.init_checkpoint,
        skip_first_n_prompts=args.skip_first_n_prompts,
        lambda_length=args.lambda_length,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_answer_tokens=args.max_answer_tokens,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        beta=args.beta,
        ref_model_name=args.ref_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        min_solved_pct=args.min_solved_pct,
        max_solved_pct=args.max_solved_pct,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_vllm=not args.no_vllm,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
    )

    # Store wandb_project as an extra attribute for grpo_train.py to use
    config.wandb_project = args.wandb_project

    return config
