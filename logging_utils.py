"""Logging utilities: JSONL helpers, advantage computation, trainer callbacks."""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import TrainingArguments

from rewards import check_correctness


def write_jsonl_line(fh, record: dict) -> None:
    """Write a single JSON record as one line to a file handle."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def compute_advantages(rewards: list[float]) -> list[float]:
    """Compute normalized advantages for a group of rewards.

    advantage_i = (reward_i - mean(rewards)) / std(rewards)

    If std is near zero (all rewards identical), all advantages are 0.
    Uses population std (not sample std), matching GRPO's implementation.
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)

    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards)

    if std_r < 1e-8:
        return [0.0] * len(rewards)

    return [(r - mean_r) / std_r for r in rewards]


class StepSyncCallback(TrainerCallback):
    """Callback that tracks per-step timing and GPU memory.

    Logs:
    - Wall-clock time per step (and cumulative)
    - GPU memory allocated/reserved
    - ETA estimate for remaining steps

    Also handles cleanup (closing file handles) on training end.
    """

    def __init__(self, reward_logger, beta: float = 0.0):
        """
        Args:
            reward_logger: The RewardLogger instance (shared by both reward functions).
            beta: KL penalty coefficient, used to compute KL loss contribution metrics.
        """
        self.reward_logger = reward_logger
        self.beta = beta
        self._step_start_time = None
        self._train_start_time = None
        self._step_times = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._train_start_time = time.time()
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_res = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        print(f"\n[Timing] Training started. GPU memory: {gpu_mem:.1f}GB allocated, {gpu_res:.1f}GB reserved")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._step_start_time is None:
            return

        step_time = time.time() - self._step_start_time
        self._step_times.append(step_time)

        # GPU memory
        gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # ETA
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        steps_done = state.global_step
        steps_remaining = state.max_steps - steps_done
        avg_step_time = statistics.mean(self._step_times)
        eta_sec = avg_step_time * steps_remaining

        # Format times
        def fmt_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds / 60:.1f}m"
            else:
                return f"{seconds / 3600:.1f}h"

        print(
            f"[Timing] Step {steps_done}/{state.max_steps}: "
            f"{step_time:.1f}s (avg {avg_step_time:.1f}s) | "
            f"Elapsed: {fmt_time(elapsed)} | ETA: {fmt_time(eta_sec)} | "
            f"GPU: {gpu_alloc:.1f}/{gpu_reserved:.1f}GB (peak {gpu_peak:.1f}GB)"
        )

        # Store on reward_logger so it can include in step_stats JSONL
        self.reward_logger._last_step_time = step_time
        self.reward_logger._avg_step_time = avg_step_time
        self.reward_logger._gpu_allocated_gb = gpu_alloc
        self.reward_logger._gpu_reserved_gb = gpu_reserved
        self.reward_logger._gpu_peak_gb = gpu_peak
        self.reward_logger._elapsed_sec = elapsed
        self.reward_logger._eta_sec = eta_sec

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Log KL contribution metrics when beta > 0."""
        if logs is None or self.beta <= 0:
            return

        kl = logs.get("kl")
        loss = logs.get("loss")
        if kl is not None and loss is not None:
            kl_contribution = self.beta * kl
            kl_fraction = kl_contribution / abs(loss) if abs(loss) > 1e-10 else 0.0
            try:
                wandb.log(
                    {
                        "kl/loss_contribution": kl_contribution,
                        "kl/loss_fraction": kl_fraction,
                    },
                    commit=False,
                )
            except Exception:
                pass

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Print training summary and close log file handles."""
        total_time = time.time() - self._train_start_time if self._train_start_time else 0
        avg_step = statistics.mean(self._step_times) if self._step_times else 0

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Steps: {state.global_step}")
        print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")
        print(f"  Avg step time: {avg_step:.1f}s")
        if torch.cuda.is_available():
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        print(f"{'=' * 60}\n")

        self.reward_logger.close()
        print(f"Log files closed.")


class CheckpointEvalCallback(TrainerCallback):
    """Runs lightweight eval on a fixed problem set every checkpoint save.

    Uses the trainer's vLLM instance (if available) for fast generation with
    continuous batching. Falls back to HF generate if vLLM is not available.
    Logs accuracy and token length stats to W&B and JSONL.
    """

    def __init__(self, eval_dataset, tokenizer, max_completion_length: int, log_dir: str):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.log_dir = Path(log_dir)
        self.trainer = None  # Set after trainer creation
        self._eval_fh = open(self.log_dir / "eval_stats.jsonl", "a", encoding="utf-8")

    def _generate_vllm(self, prompts, answers):
        """Generate completions using the trainer's vLLM instance."""
        from vllm import SamplingParams

        vllm_gen = self.trainer.vllm_generation
        # Sync current LoRA weights to vLLM before generating
        vllm_gen.sync_weights()

        sampling_params = SamplingParams(
            temperature=0,  # Greedy
            max_tokens=self.max_completion_length,
            stop_token_ids=[151645, 151643],  # im_end, eos
        )

        # vLLM handles all 100 prompts at once with continuous batching
        outputs = vllm_gen.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        all_correct = []
        all_num_tokens = []
        for i, output in enumerate(outputs):
            completion = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            all_correct.append(check_correctness(completion, answers[i]))
            all_num_tokens.append(num_tokens)

        return all_correct, all_num_tokens

    def _generate_hf(self, model, prompts, answers):
        """Fallback: generate with HF generate in batches with OOM retry."""
        all_correct = []
        all_num_tokens = []
        batch_size = 50

        with torch.no_grad():
            i = 0
            while i < len(prompts):
                batch_prompts = prompts[i : i + batch_size]
                batch_answers = answers[i : i + batch_size]

                try:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(model.device)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_completion_length,
                        do_sample=False,
                    )

                    prompt_len = inputs["input_ids"].shape[1]
                    for j, output in enumerate(outputs):
                        gen_ids = output[prompt_len:]
                        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        all_correct.append(check_correctness(completion, batch_answers[j]))
                        all_num_tokens.append(len(gen_ids))
                    i += batch_size
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch_size = max(batch_size // 2, 1)
                    print(f"[Eval] OOM, reducing batch size to {batch_size}")

        return all_correct, all_num_tokens

    def _run_eval(self, step: int):
        """Run eval on the fixed problem set and log results.

        Tries vLLM first (fast, ~1-2 min). Falls back to HF generate if
        vLLM is unavailable or fails.
        """
        if self.trainer is None:
            return

        print(f"\n[Eval] Running checkpoint eval at step {step} on {len(self.eval_dataset)} problems...")
        t0 = time.time()

        prompts = self.eval_dataset["prompt"]
        answers = self.eval_dataset["answer"]

        # Try vLLM first (fast), fall back to HF generate (slow)
        use_vllm = hasattr(self.trainer, "vllm_generation") and self.trainer.vllm_generation is not None
        if use_vllm:
            try:
                print("[Eval] Using vLLM for generation...")
                all_correct, all_num_tokens = self._generate_vllm(prompts, answers)
            except Exception as e:
                print(f"[Eval] vLLM failed ({e}), falling back to HF generate...")
                use_vllm = False

        if not use_vllm:
            print("[Eval] Using HF generate (slow fallback)...")
            model = self.trainer.model
            was_training = model.training
            model.eval()
            all_correct, all_num_tokens = self._generate_hf(model, prompts, answers)
            if was_training:
                model.train()

        # Compute stats
        accuracy = statistics.mean(all_correct)
        mean_tokens = statistics.mean(all_num_tokens)
        sorted_tokens = sorted(all_num_tokens)
        median_tokens = sorted_tokens[len(sorted_tokens) // 2]
        elapsed = time.time() - t0

        # Console
        print(
            f"[Eval] Step {step}: acc={accuracy:.3f} "
            f"mean_tok={mean_tokens:.0f} med_tok={median_tokens} "
            f"({elapsed:.1f}s)"
        )

        # JSONL
        record = {
            "step": step,
            "accuracy": accuracy,
            "mean_tokens": mean_tokens,
            "median_tokens": median_tokens,
            "min_tokens": min(all_num_tokens),
            "max_tokens": max(all_num_tokens),
            "num_problems": len(all_correct),
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        self._eval_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._eval_fh.flush()

        # W&B
        try:
            wandb.log(
                {
                    "eval/accuracy": accuracy,
                    "eval/mean_tokens": mean_tokens,
                    "eval/median_tokens": median_tokens,
                },
                commit=False,
            )
        except Exception:
            pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._run_eval(step=0)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._run_eval(step=state.global_step)

    def close(self):
        try:
            self._eval_fh.close()
        except Exception:
            pass
