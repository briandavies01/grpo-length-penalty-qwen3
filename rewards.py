"""Reward functions for GRPO training with GDPO-style normalize_then_sum.

Two separate reward functions (correctness + length penalty) that TRL normalizes
independently per group, then combines with reward_weights=[1.0, lambda_length].

This prevents group normalization from canceling out lambda (the root cause of
lambda 0.1-2.0 producing identical training dynamics in the entangled setup).

Architecture:
- RewardLogger: shared state + file handles, called by both reward functions
- CorrectnessRewardFunction: returns 0/1 correctness, buffers data on logger
- LengthPenaltyRewardFunction: returns length penalty, triggers full log flush
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import wandb
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify


def extract_answer_text(
    completion_text: str,
    max_answer_tokens: int = 0,
) -> str:
    """Extract the portion of completion text that should contain the final answer.

    Strips <think>...</think> block to avoid matching intermediate \\boxed{}
    that may appear inside the reasoning trace.

    Three cases:
    1. </think> present: return text after the last </think>
    2. <think> present but no </think> (truncated): return "" (no valid answer)
    3. Neither tag present: return full text

    If max_answer_tokens > 0, truncate the answer to approximately that many
    tokens (using whitespace splitting as a proxy). This prevents the model
    from being rewarded for reasoning done outside of <think> tags.
    """
    if "</think>" in completion_text:
        answer = completion_text.split("</think>")[-1]
    elif "<think>" in completion_text:
        # Think block started but never closed — completion was truncated
        return ""
    else:
        answer = completion_text

    if max_answer_tokens > 0:
        # Approximate token count via whitespace split.
        # Math tokens average ~4-5 chars; whitespace split slightly undercounts
        # vs real tokenizer, so this is a conservative (generous) limit.
        words = answer.split()
        if len(words) > max_answer_tokens:
            answer = " ".join(words[:max_answer_tokens])

    return answer


def check_correctness(
    completion_text: str,
    ground_truth: str,
    max_answer_tokens: int = 0,
) -> float:
    """Check if a completion's answer matches the ground truth.

    Uses math-verify for symbolic equivalence checking (handles fractions,
    decimals, equivalent expressions, etc.).

    Returns 1.0 if correct, 0.0 if incorrect or on any error.

    If max_answer_tokens > 0, only looks at that many tokens after </think>.
    """
    answer_text = extract_answer_text(completion_text, max_answer_tokens)
    if not answer_text.strip():
        return 0.0

    try:
        # Parse ground truth — wrap in \boxed{} so math-verify's LaTeX
        # extractor handles all expressions (sqrt, pi, etc.) correctly.
        # Without the wrapper, parse() fails on e.g. "3\sqrt{3}" or "2\pi".
        gold_parsed = parse(
            r"\boxed{" + ground_truth + "}",
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
            ],
        )
        if not gold_parsed:
            return 0.0

        # Parse model answer — prioritize \boxed{} matches
        answer_parsed = parse(
            answer_text,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )

        return 1.0 if verify(gold_parsed, answer_parsed) else 0.0
    except Exception:
        return 0.0


def compute_length_penalty(num_tokens: int, max_completion_length: int) -> float:
    """Compute length penalty: -(num_tokens / max_completion_length).

    Always negative or zero. A completion using half the max length gets -0.5.
    """
    return -(num_tokens / max_completion_length)


class RewardLogger:
    """Shared logging state between the two reward functions.

    Holds file handles, timing attributes (set by StepSyncCallback), and a
    buffer so CorrectnessRewardFunction can pass data to LengthPenaltyRewardFunction
    for the combined log flush.

    Also computes a synthetic "total_reward" (correctness + lambda * lp) for log
    compatibility with analysis scripts. This is NOT used by TRL for training —
    TRL uses normalize_then_sum with reward_weights instead.
    """

    def __init__(
        self,
        num_generations: int,
        max_completion_length: int,
        lambda_length: float,
        log_dir: str,
    ):
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.lambda_length = lambda_length  # For logging only

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._rollout_fh = open(self.log_dir / "rollouts.jsonl", "a", encoding="utf-8")
        self._prompt_fh = open(self.log_dir / "prompt_stats.jsonl", "a", encoding="utf-8")
        self._step_fh = open(self.log_dir / "step_stats.jsonl", "a", encoding="utf-8")

        # Buffer: set by CorrectnessRewardFunction, consumed by LengthPenaltyRewardFunction
        self._correctness_buffer = None

    def buffer_correctness_data(self, data: dict):
        """Store correctness data for the length penalty function to use."""
        self._correctness_buffer = data

    def flush_logs(self, length_penalties: list[float], num_tokens_list: list[int]):
        """Write all three log files using buffered correctness + fresh length data.

        Called by LengthPenaltyRewardFunction after computing length penalties.
        """
        buf = self._correctness_buffer
        self._correctness_buffer = None

        step = buf["step"]
        timestamp = buf["timestamp"]
        prompts = buf["prompts"]
        completions = buf["completions"]
        solutions = buf["solutions"]
        all_correctness = buf["all_correctness"]
        all_has_think = buf["all_has_think"]
        all_has_boxed = buf["all_has_boxed"]

        G = self.num_generations
        num_completions = len(completions)
        num_prompts = num_completions // G

        # Compute synthetic total_reward for log compatibility
        all_rewards = [
            c + self.lambda_length * lp
            for c, lp in zip(all_correctness, length_penalties)
        ]

        # --- Per-rollout logging ---
        for i in range(num_completions):
            prompt_idx = i // G
            rollout_idx = i % G

            rollout_record = {
                "step": step,
                "prompt_index": prompt_idx,
                "rollout_index": rollout_idx,
                "prompt_text": prompts[i][:300] if i < len(prompts) else "",
                "completion_text": completions[i],
                "ground_truth": solutions[i],
                "correctness_reward": all_correctness[i],
                "length_penalty": length_penalties[i],
                "total_reward": all_rewards[i],
                "num_tokens": num_tokens_list[i],
                "has_think_tags": all_has_think[i],
                "has_boxed": all_has_boxed[i],
                "timestamp": timestamp,
            }
            self._rollout_fh.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")

        self._rollout_fh.flush()

        # --- Per-prompt aggregation ---
        for p in range(num_prompts):
            start = p * G
            end = start + G

            group_rewards = all_rewards[start:end]
            group_correctness = all_correctness[start:end]
            group_length_pens = length_penalties[start:end]
            group_tokens = num_tokens_list[start:end]
            group_think = all_has_think[start:end]
            group_boxed = all_has_boxed[start:end]

            mean_r = statistics.mean(group_rewards)
            std_r = statistics.pstdev(group_rewards)
            if std_r > 1e-8:
                advantages = [(r - mean_r) / std_r for r in group_rewards]
            else:
                advantages = [0.0] * G

            prompt_record = {
                "step": step,
                "prompt_index": p,
                "prompt_text_truncated": prompts[start][:300] if start < len(prompts) else "",
                "ground_truth": solutions[start] if start < len(solutions) else "",
                "num_rollouts": G,
                "num_correct": int(sum(group_correctness)),
                "accuracy": statistics.mean(group_correctness),
                "reward_mean": mean_r,
                "reward_std": std_r,
                "correctness_mean": statistics.mean(group_correctness),
                "length_penalty_mean": statistics.mean(group_length_pens),
                "length_mean": statistics.mean(group_tokens),
                "length_std": statistics.pstdev(group_tokens),
                "length_min": min(group_tokens),
                "length_max": max(group_tokens),
                "normalized_advantages": [round(a, 4) for a in advantages],
                "frac_has_think": sum(group_think) / G,
                "frac_has_boxed": sum(group_boxed) / G,
                "timestamp": timestamp,
            }
            self._prompt_fh.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")

        self._prompt_fh.flush()

        # --- Per-step aggregation ---
        accuracy = statistics.mean(all_correctness)
        mean_tokens = statistics.mean(num_tokens_list)
        sorted_tokens = sorted(num_tokens_list)
        median_tokens = sorted_tokens[len(sorted_tokens) // 2]

        per_prompt_stds = []
        frac_all_correct = 0
        frac_all_incorrect = 0
        for p in range(num_prompts):
            start = p * G
            end = start + G
            group_r = all_rewards[start:end]
            per_prompt_stds.append(statistics.pstdev(group_r))
            group_c = all_correctness[start:end]
            if all(c == 1.0 for c in group_c):
                frac_all_correct += 1
            if all(c == 0.0 for c in group_c):
                frac_all_incorrect += 1

        total_tokens = sum(num_tokens_list)

        step_record = {
            "step": step,
            "num_prompts": num_prompts,
            "num_rollouts": num_completions,
            "accuracy": accuracy,
            "mean_correctness_reward": statistics.mean(all_correctness),
            "mean_length_penalty": statistics.mean(length_penalties),
            "mean_total_reward": statistics.mean(all_rewards),
            "std_total_reward": statistics.pstdev(all_rewards),
            "total_tokens": total_tokens,
            "mean_completion_length": mean_tokens,
            "median_completion_length": median_tokens,
            "min_completion_length": min(num_tokens_list),
            "max_completion_length": max(num_tokens_list),
            "frac_has_think": sum(all_has_think) / num_completions,
            "frac_has_boxed": sum(all_has_boxed) / num_completions,
            "mean_reward_std_per_prompt": statistics.mean(per_prompt_stds) if per_prompt_stds else 0.0,
            "frac_prompts_all_correct": frac_all_correct / num_prompts if num_prompts > 0 else 0.0,
            "frac_prompts_all_incorrect": frac_all_incorrect / num_prompts if num_prompts > 0 else 0.0,
            "step_time_sec": getattr(self, "_last_step_time", None),
            "avg_step_time_sec": getattr(self, "_avg_step_time", None),
            "gpu_allocated_gb": getattr(self, "_gpu_allocated_gb", None),
            "gpu_reserved_gb": getattr(self, "_gpu_reserved_gb", None),
            "gpu_peak_gb": getattr(self, "_gpu_peak_gb", None),
            "elapsed_sec": getattr(self, "_elapsed_sec", None),
            "eta_sec": getattr(self, "_eta_sec", None),
            "timestamp": timestamp,
        }
        self._step_fh.write(json.dumps(step_record, ensure_ascii=False) + "\n")
        self._step_fh.flush()

        # --- Console output ---
        print(
            f"[Step {step}] "
            f"acc={accuracy:.3f} "
            f"len={mean_tokens:.0f} "
            f"(med={median_tokens}, min={min(num_tokens_list)}, max={max(num_tokens_list)}) "
            f"total_tok={total_tokens} "
            f"think={step_record['frac_has_think']:.2f} "
            f"boxed={step_record['frac_has_boxed']:.2f} "
            f"r_total={step_record['mean_total_reward']:.3f} "
            f"r_std={step_record['mean_reward_std_per_prompt']:.3f}"
        )

        # --- W&B logging ---
        try:
            wb_data = {
                "custom/accuracy": accuracy,
                "custom/mean_completion_length": mean_tokens,
                "custom/median_completion_length": median_tokens,
                "custom/total_tokens": total_tokens,
                "custom/mean_correctness_reward": step_record["mean_correctness_reward"],
                "custom/mean_length_penalty": step_record["mean_length_penalty"],
                "custom/mean_total_reward": step_record["mean_total_reward"],
                "custom/frac_has_think": step_record["frac_has_think"],
                "custom/frac_has_boxed": step_record["frac_has_boxed"],
                "custom/mean_reward_std_per_prompt": step_record["mean_reward_std_per_prompt"],
                "custom/frac_prompts_all_correct": step_record["frac_prompts_all_correct"],
                "custom/frac_prompts_all_incorrect": step_record["frac_prompts_all_incorrect"],
            }
            if step_record.get("step_time_sec") is not None:
                wb_data["perf/step_time_sec"] = step_record["step_time_sec"]
                wb_data["perf/gpu_allocated_gb"] = step_record["gpu_allocated_gb"]
                wb_data["perf/gpu_reserved_gb"] = step_record["gpu_reserved_gb"]
                wb_data["perf/gpu_peak_gb"] = step_record["gpu_peak_gb"]
            wandb.log(wb_data)
        except Exception:
            pass

    def close(self):
        """Close file handles. Call at end of training."""
        for fh in (self._rollout_fh, self._prompt_fh, self._step_fh):
            try:
                fh.close()
            except Exception:
                pass


class CorrectnessRewardFunction:
    """Returns binary correctness reward (0 or 1) for each completion.

    Buffers per-rollout metadata on the shared RewardLogger so the
    LengthPenaltyRewardFunction can include it in the combined log flush.
    """

    __name__ = "correctness"

    def __init__(self, logger: RewardLogger, max_answer_tokens: int = 0):
        self.logger = logger
        self.max_answer_tokens = max_answer_tokens

    def __call__(self, prompts, completions, completion_ids, **kwargs):
        solutions = kwargs.get("solution", [""] * len(completions))
        trainer_state = kwargs.get("trainer_state", None)
        step = trainer_state.global_step if trainer_state else 0
        timestamp = datetime.now().isoformat()

        all_correctness = []
        all_has_think = []
        all_has_boxed = []

        for i in range(len(completions)):
            comp_text = completions[i]
            sol = solutions[i]

            correctness = check_correctness(comp_text, sol, self.max_answer_tokens)
            has_think = "<think>" in comp_text and "</think>" in comp_text
            has_boxed = "\\boxed" in comp_text

            all_correctness.append(correctness)
            all_has_think.append(has_think)
            all_has_boxed.append(has_boxed)

        # Buffer data for the length penalty function to use during log flush
        self.logger.buffer_correctness_data({
            "step": step,
            "timestamp": timestamp,
            "prompts": prompts,
            "completions": completions,
            "solutions": solutions,
            "all_correctness": all_correctness,
            "all_has_think": all_has_think,
            "all_has_boxed": all_has_boxed,
        })

        return all_correctness


class LengthPenaltyRewardFunction:
    """Returns raw length penalty for each completion (no lambda scaling).

    Lambda scaling is handled by TRL's reward_weights parameter.
    After computing length penalties, triggers the full log flush on the
    shared RewardLogger.
    """

    __name__ = "length_penalty"

    def __init__(self, logger: RewardLogger):
        self.logger = logger

    def __call__(self, prompts, completions, completion_ids, **kwargs):
        max_len = self.logger.max_completion_length

        all_length_penalties = []
        all_num_tokens = []

        for i in range(len(completions)):
            num_tokens = len(completion_ids[i])
            lp = compute_length_penalty(num_tokens, max_len)
            all_length_penalties.append(lp)
            all_num_tokens.append(num_tokens)

        # Trigger combined log flush
        self.logger.flush_logs(all_length_penalties, all_num_tokens)

        return all_length_penalties
