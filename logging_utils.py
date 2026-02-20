"""Logging utilities: JSONL helpers, advantage computation, trainer callback."""

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import TrainingArguments


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

    def __init__(self, reward_logger):
        """
        Args:
            reward_logger: The RewardLogger instance (shared by both reward functions).
        """
        self.reward_logger = reward_logger
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
