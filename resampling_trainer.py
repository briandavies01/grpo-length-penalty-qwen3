"""ResamplingGRPOTrainer: resamples completions that hit max_completion_length.

When a completion is truncated (doesn't end with an EOS token), it gets
resampled up to `max_resample_retries` times via vLLM. Any still-truncated
after retries have their loss zeroed via mask_truncated_completions.
"""

import math

import torch
import wandb
from trl import GRPOTrainer
from vllm import SamplingParams


def _sanitize_logprob(logprob):
    """Extract logprob value, returning None for NaN."""
    value = logprob.logprob
    if math.isnan(value):
        return None
    return value


class ResamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that resamples truncated completions before scoring.

    Overrides _generate() to detect completions that hit max_completion_length
    (no EOS token at end), resample them via vLLM, and splice replacements in.
    Falls back to mask_truncated_completions=True for any still-truncated.
    """

    def __init__(self, *args, max_resample_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_resample_retries = max_resample_retries
        # Fallback: zero loss for completions still truncated after retries
        self.mask_truncated_completions = True

        # Resampling requires direct access to vLLM LLM instance (colocate only)
        if not hasattr(self, "vllm_generation") or not hasattr(self.vllm_generation, "llm"):
            raise ValueError(
                "ResamplingGRPOTrainer requires vllm_mode='colocate'. "
                "Server mode is not supported."
            )

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

        initial_truncated = len(truncated_indices)
        total_completions = len(completion_ids)

        # Build SamplingParams matching training config
        stop_ids = self.args.generation_kwargs.get("stop_token_ids", [])
        sampling_params = SamplingParams(
            n=1,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_completion_length,
            stop_token_ids=stop_ids,
            logprobs=0,
        )

        # Resample up to K times
        for retry in range(self.max_resample_retries):
            if not truncated_indices:
                break

            # Build token-based prompts for truncated completions
            # Use int() to handle both plain lists and tensor elements
            resample_prompts = [
                {"prompt_token_ids": [int(t) for t in prompt_ids[i]]}
                for i in truncated_indices
            ]

            # Call vLLM directly (weights already synced by super()._generate)
            outputs = self.vllm_generation.llm.generate(
                resample_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Splice results back in
            still_truncated = []
            for j, idx in enumerate(truncated_indices):
                output = outputs[j].outputs[0]
                new_ids = list(output.token_ids)

                # Decode with same method as TRL's _generate
                new_text = self.processing_class.decode(new_ids, skip_special_tokens=True)

                # Extract logprobs
                new_lps = None
                if output.logprobs:
                    new_lps = [
                        _sanitize_logprob(next(iter(lp.values())))
                        for lp in output.logprobs
                    ]

                # Update in-place
                completion_ids[idx] = new_ids
                completions[idx] = new_text
                if logprobs is not None and new_lps is not None:
                    logprobs[idx] = new_lps

                if self._is_truncated(new_ids):
                    still_truncated.append(idx)

            truncated_indices = still_truncated

            print(
                f"[Resample retry {retry + 1}/{self.max_resample_retries}] "
                f"{len(still_truncated)} still truncated"
            )

        # Recompute total_completion_tokens (distributed-aware)
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids],
            device=self.accelerator.device,
        )
        total_completion_tokens = self.accelerator.gather(completion_lengths).sum()

        # Log resampling stats
        final_truncated = len(truncated_indices)
        resampled_ok = initial_truncated - final_truncated

        print(
            f"[Resample] {initial_truncated}/{total_completions} truncated -> "
            f"{resampled_ok} resampled OK, {final_truncated} still truncated"
        )

        try:
            wandb.log(
                {
                    "resample/initial_truncated": initial_truncated,
                    "resample/initial_truncated_frac": initial_truncated / total_completions,
                    "resample/resampled_ok": resampled_ok,
                    "resample/still_truncated": final_truncated,
                    "resample/still_truncated_frac": final_truncated / total_completions,
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
