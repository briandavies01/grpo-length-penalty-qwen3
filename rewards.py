"""Reward function for GRPO training with combined correctness + length penalty.

Single reward function: reward = correctness - lambda * (tokens / max_tokens)
where correctness = +1 if correct, -1 if incorrect.

TRL normalizes this combined reward within each group (scale_rewards="group"),
giving standard GRPO advantage estimation on the total signal.

Architecture:
- RewardLogger: logging state + file handles
- CombinedRewardFunction: computes combined reward, triggers log flush
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


def _normalize_latex(text: str) -> str:
    """Normalize LaTeX formatting differences that don't change mathematical meaning."""
    # \dfrac → \frac (display vs inline — mathematically identical)
    text = text.replace("\\dfrac", "\\frac")
    # Strip trailing percentage signs — we compare the numeric value
    text = text.rstrip().removesuffix("\\%").removesuffix("%").rstrip()
    return text


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

    # Normalize formatting differences (\dfrac→\frac, strip %)
    ground_truth_norm = _normalize_latex(ground_truth)
    answer_text_norm = answer_text.replace("\\dfrac", "\\frac")

    try:
        # Parse ground truth — wrap in \boxed{} so math-verify's LaTeX
        # extractor handles all expressions (sqrt, pi, etc.) correctly.
        # Without the wrapper, parse() fails on e.g. "3\sqrt{3}" or "2\pi".
        gold_parsed = parse(
            r"\boxed{" + ground_truth_norm + "}",
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
            ],
        )
        if not gold_parsed:
            return 0.0

        # Parse model answer — prioritize \boxed{} matches
        answer_parsed = parse(
            answer_text_norm,
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


class RewardLogger:
    """Logging state for the combined reward function.

    Holds file handles and timing attributes (set by StepSyncCallback).
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
        self.lambda_length = lambda_length

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._rollout_fh = open(self.log_dir / "rollouts.jsonl", "a", encoding="utf-8")
        self._prompt_fh = open(self.log_dir / "prompt_stats.jsonl", "a", encoding="utf-8")
        self._step_fh = open(self.log_dir / "step_stats.jsonl", "a", encoding="utf-8")

    def flush_logs(
        self,
        step: int,
        timestamp: str,
        prompts: list[str],
        completions: list[str],
        solutions: list[str],
        all_correctness_binary: list[float],
        all_rewards: list[float],
        length_ratios: list[float],
        num_tokens_list: list[int],
        all_has_think: list[bool],
        all_has_boxed: list[bool],
    ):
        """Write all three log files with combined reward data."""
        G = self.num_generations
        num_completions = len(completions)
        num_prompts = num_completions // G

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
                "correctness_binary": all_correctness_binary[i],
                "length_ratio": length_ratios[i],
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
            group_correctness = all_correctness_binary[start:end]
            group_length_ratios = length_ratios[start:end]
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
                "length_ratio_mean": statistics.mean(group_length_ratios),
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
        accuracy = statistics.mean(all_correctness_binary)
        mean_tokens = statistics.mean(num_tokens_list)
        sorted_tokens = sorted(num_tokens_list)
        median_tokens = sorted_tokens[len(sorted_tokens) // 2]

        per_prompt_stds = []
        per_prompt_correctness_stds = []
        per_prompt_length_stds = []
        per_prompt_length_var_shares = []
        frac_all_correct = 0
        frac_all_incorrect = 0
        prompts_with_max_length = 0
        for p in range(num_prompts):
            start = p * G
            end = start + G
            group_r = all_rewards[start:end]
            per_prompt_stds.append(statistics.pstdev(group_r))
            group_c = all_correctness_binary[start:end]
            if all(c == 1.0 for c in group_c):
                frac_all_correct += 1
            if all(c == 0.0 for c in group_c):
                frac_all_incorrect += 1
            group_tokens = num_tokens_list[start:end]
            if any(t >= self.max_completion_length for t in group_tokens):
                prompts_with_max_length += 1

            # Variance decomposition: correctness component vs length component
            group_correctness_scores = [1.0 if c == 1.0 else -1.0 for c in group_c]
            group_length_terms = [self.lambda_length * lr for lr in length_ratios[start:end]]
            std_c = statistics.pstdev(group_correctness_scores)
            std_l = statistics.pstdev(group_length_terms)
            per_prompt_correctness_stds.append(std_c)
            per_prompt_length_stds.append(std_l)

            # Length variance share: var(length) / var(total), guarded against zero
            var_total = statistics.pvariance(group_r)
            if var_total > 1e-12:
                per_prompt_length_var_shares.append(statistics.pvariance(group_length_terms) / var_total)
            else:
                per_prompt_length_var_shares.append(0.5)  # No signal either way

        total_tokens = sum(num_tokens_list)

        step_record = {
            "step": step,
            "num_prompts": num_prompts,
            "num_rollouts": num_completions,
            "accuracy": accuracy,
            "mean_length_ratio": statistics.mean(length_ratios),
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
            "frac_prompts_hit_max_length": prompts_with_max_length / num_prompts if num_prompts > 0 else 0.0,
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
                "custom/max_completion_length": max(num_tokens_list),
                "custom/total_tokens": total_tokens,
                "custom/mean_length_ratio": step_record["mean_length_ratio"],
                "custom/mean_total_reward": step_record["mean_total_reward"],
                "custom/frac_has_think": step_record["frac_has_think"],
                "custom/frac_has_boxed": step_record["frac_has_boxed"],
                "custom/mean_reward_std_per_prompt": step_record["mean_reward_std_per_prompt"],
                "custom/frac_prompts_all_correct": step_record["frac_prompts_all_correct"],
                "custom/frac_prompts_all_incorrect": step_record["frac_prompts_all_incorrect"],
                "custom/frac_prompts_hit_max_length": step_record["frac_prompts_hit_max_length"],
                "signal/mean_group_std_correctness": statistics.mean(per_prompt_correctness_stds),
                "signal/mean_group_std_length": statistics.mean(per_prompt_length_stds),
                "signal/length_variance_share": statistics.mean(per_prompt_length_var_shares),
            }
            if step_record.get("step_time_sec") is not None:
                wb_data["perf/step_time_sec"] = step_record["step_time_sec"]
                wb_data["perf/gpu_allocated_gb"] = step_record["gpu_allocated_gb"]
                wb_data["perf/gpu_reserved_gb"] = step_record["gpu_reserved_gb"]
                wb_data["perf/gpu_peak_gb"] = step_record["gpu_peak_gb"]
            wandb.log(wb_data, commit=False)
        except Exception:
            pass

    def close(self):
        """Close file handles. Call at end of training."""
        for fh in (self._rollout_fh, self._prompt_fh, self._step_fh):
            try:
                fh.close()
            except Exception:
                pass


class CombinedRewardFunction:
    """Returns combined reward: correctness_score - lambda * (tokens / max_tokens).

    correctness_score = +1 if correct, -1 if incorrect.
    TRL normalizes this single reward within each group (scale_rewards="group").
    """

    __name__ = "combined_reward"

    def __init__(self, logger: RewardLogger, lambda_length: float = 2.0, max_answer_tokens: int = 0,
                 length_penalty_on_correct_only: bool = False):
        self.logger = logger
        self.lambda_length = lambda_length
        self.max_answer_tokens = max_answer_tokens
        self.length_penalty_on_correct_only = length_penalty_on_correct_only

    def __call__(self, prompts, completions, completion_ids, **kwargs):
        solutions = kwargs.get("solution", [""] * len(completions))
        trainer_state = kwargs.get("trainer_state", None)
        step = trainer_state.global_step if trainer_state else 0
        timestamp = datetime.now().isoformat()

        max_len = self.logger.max_completion_length

        all_rewards = []
        all_correctness_binary = []
        all_length_ratios = []
        all_num_tokens = []
        all_has_think = []
        all_has_boxed = []

        for i in range(len(completions)):
            comp_text = completions[i]
            sol = solutions[i]

            # Binary correctness (0/1) for accuracy logging
            correct = check_correctness(comp_text, sol, self.max_answer_tokens)

            # Correctness score: +1 or -1
            correctness_score = 1.0 if correct == 1.0 else -1.0

            # Length ratio (0 to 1)
            num_tokens = len(completion_ids[i])
            length_ratio = num_tokens / max_len

            # Combined reward
            if self.length_penalty_on_correct_only and correct != 1.0:
                reward = -1.0  # Flat penalty for wrong answers, no length component
            else:
                reward = correctness_score - self.lambda_length * length_ratio

            all_rewards.append(reward)
            all_correctness_binary.append(correct)
            all_length_ratios.append(length_ratio)
            all_num_tokens.append(num_tokens)

            has_think = "<think>" in comp_text and "</think>" in comp_text
            has_boxed = "\\boxed" in comp_text
            all_has_think.append(has_think)
            all_has_boxed.append(has_boxed)

        # Flush all logs
        self.logger.flush_logs(
            step=step,
            timestamp=timestamp,
            prompts=prompts,
            completions=completions,
            solutions=solutions,
            all_correctness_binary=all_correctness_binary,
            all_rewards=all_rewards,
            length_ratios=all_length_ratios,
            num_tokens_list=all_num_tokens,
            all_has_think=all_has_think,
            all_has_boxed=all_has_boxed,
        )

        return all_rewards
