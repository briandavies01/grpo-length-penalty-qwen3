"""TruncationDroppingTrainer: drops truncated completions to save memory.

Replaces truncated completions (those hitting max_completion_length) with
a minimal 1-token stub. This shrinks the padded batch tensor from
[batch, max_completion_length] to [batch, max_non_truncated_length],
preventing OOM on long-sequence batches.

mask_truncated_completions=True zeros their loss so they don't affect training.
The stub keeps the first token (non-EOS) so TRL still detects it as truncated.
"""

import torch
import wandb
from trl import GRPOTrainer


class ResamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that drops truncated completions to save memory.

    Overrides _generate() to detect completions that hit max_completion_length
    (no EOS token at end) and replace them with 1-token stubs. This shrinks
    the padded batch tensor, preventing OOM. mask_truncated_completions=True
    zeros loss for these stubs.
    """

    def __init__(self, *args, max_resample_retries=None, **kwargs):
        # max_resample_retries accepted for config compat but unused
        kwargs.pop("max_resample_retries", None)
        super().__init__(*args, **kwargs)
        # Zero loss for truncated (now stubbed) completions
        self.mask_truncated_completions = True

        # Build EOS token set from tokenizer + config (not hardcoded)
        self._eos_tokens = set()
        if self.processing_class.eos_token_id is not None:
            self._eos_tokens.add(self.processing_class.eos_token_id)
        stop_ids = self.args.generation_kwargs.get("stop_token_ids", [])
        self._eos_tokens.update(stop_ids)
        if self.processing_class.pad_token_id is not None:
            self._eos_tokens.add(self.processing_class.pad_token_id)

    def _is_truncated(self, ids):
        """Check if a completion was truncated (didn't end with EOS/pad)."""
        return len(ids) > 0 and ids[-1] not in self._eos_tokens

    def _generate(self, prompts):
        # Get initial generations from parent (includes weight sync)
        result = super()._generate(prompts)
        (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        ) = result

        # Detect truncated completions
        truncated_indices = [
            i for i, ids in enumerate(completion_ids) if self._is_truncated(ids)
        ]

        if not truncated_indices:
            return result

        num_truncated = len(truncated_indices)
        total_completions = len(completion_ids)
        max_before = max(len(ids) for ids in completion_ids)

        # Replace truncated completions with 1-token stub
        # Keep first token (non-EOS) so TRL still detects as truncated → mask zeros loss
        for idx in truncated_indices:
            completion_ids[idx] = [completion_ids[idx][0]]
            completions[idx] = ""
            if logprobs is not None and logprobs[idx]:
                logprobs[idx] = [logprobs[idx][0]]
            if tool_mask is not None and tool_mask[idx]:
                tool_mask[idx] = [tool_mask[idx][0]]

        max_after = max(len(ids) for ids in completion_ids)

        # Recompute total_completion_tokens
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids],
            device=self.accelerator.device,
        )
        total_completion_tokens = self.accelerator.gather(completion_lengths).sum()

        print(
            f"[Drop truncated] {num_truncated}/{total_completions} dropped "
            f"(max seq: {max_before} -> {max_after})"
        )

        try:
            wandb.log(
                {
                    "truncation/num_dropped": num_truncated,
                    "truncation/frac_dropped": num_truncated / total_completions,
                    "truncation/max_seq_before": max_before,
                    "truncation/max_seq_after": max_after,
                },
                commit=False,
            )
        except Exception:
            pass

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )
