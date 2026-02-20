"""Evaluation of Qwen3 base model or LoRA checkpoints on math problems.

Generates multiple completions per problem and logs per-rollout data +
summary statistics. Uses vLLM by default for fast batched inference.

Usage:
    # Eval base model with vLLM (default)
    python baseline_eval.py --min_solved_pct 98 --max_solved_pct 99 --max_new_tokens 4096

    # Eval a LoRA checkpoint
    python baseline_eval.py --checkpoint_path ./outputs/some_run/checkpoint-100

    # Without vLLM (debugging)
    python baseline_eval.py --no_vllm --num_problems 10 --num_generations 2
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Force vLLM V0 engine — V1 has torch.compile bugs with model_impl="transformers"
os.environ["VLLM_USE_V1"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

from data import sample_baseline_problems, verify_prompt_tokenization
from rewards import check_correctness, compute_length_penalty

# Qwen3 stop tokens: <|im_end|> (151645) and EOS (151643)
STOP_TOKEN_IDS = [151645, 151643]


# ── HF generate (slow, one completion at a time) ──

def generate_completions_hf(
    model,
    tokenizer,
    prompt_text: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """Generate multiple completions for a single prompt using HF generate."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    results = []
    for _ in range(num_generations):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=STOP_TOKEN_IDS,
            )

        completion_ids = output_ids[0][prompt_len:].tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        results.append({
            "completion_text": completion_text,
            "num_tokens": len(completion_ids),
            "has_think_tags": "<think>" in completion_text and "</think>" in completion_text,
            "has_boxed": "\\boxed" in completion_text,
        })

    return results


# ── vLLM generate (fast, all completions in one batch) ──

def generate_all_vllm(
    llm,
    prompts: list[str],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    lora_request=None,
) -> list[list[dict]]:
    """Generate completions for all prompts in a single vLLM batch.

    Returns list of lists: results[problem_idx][rollout_idx] = dict.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=STOP_TOKEN_IDS,
    )

    kwargs = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    outputs = llm.generate(prompts, sampling_params, **kwargs)

    all_results = []
    for output in outputs:
        problem_results = []
        for completion in output.outputs:
            text = completion.text
            problem_results.append({
                "completion_text": text,
                "num_tokens": len(completion.token_ids),
                "has_think_tags": "<think>" in text and "</think>" in text,
                "has_boxed": "\\boxed" in text,
            })
        all_results.append(problem_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 base model or LoRA checkpoint")
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-1.7B",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to LoRA adapter directory (checkpoint or final_adapter)")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--min_solved_pct", type=float, default=50.0,
                        help="Min solved_percentage for difficulty filter")
    parser.add_argument("--max_solved_pct", type=float, default=95.0,
                        help="Max solved_percentage for difficulty filter")
    parser.add_argument("--seed", type=int, default=42)
    # vLLM options
    parser.add_argument("--no_vllm", action="store_true", help="Use HF generate instead of vLLM")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Max prompt length (for vLLM max_model_len calculation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vllm = not args.no_vllm

    # --- Load evaluation problems ---
    print(f"Sampling {args.num_problems} problems (seed={args.seed}, "
          f"difficulty={args.min_solved_pct}-{args.max_solved_pct}%)...")
    dataset = sample_baseline_problems(
        num_problems=args.num_problems,
        seed=args.seed,
        min_solved_pct=args.min_solved_pct,
        max_solved_pct=args.max_solved_pct,
    )
    print(f"Loaded {len(dataset)} problems")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Qwen3 has no BOS token (add_bos_token=false by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    verify_prompt_tokenization(tokenizer, dataset[0]["prompt"])

    # --- Initialize model ---
    lora_request = None

    if use_vllm:
        from vllm import LLM

        max_model_len = args.max_prompt_length + args.max_new_tokens

        llm_kwargs = dict(
            model=args.model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            model_impl="transformers",
        )

        if args.checkpoint_path:
            from vllm.lora.request import LoRARequest
            print(f"Loading base model with LoRA support: {args.model_name}")
            print(f"  Checkpoint: {args.checkpoint_path}")
            adapter_config_path = os.path.join(args.checkpoint_path, "adapter_config.json")
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            max_lora_rank = adapter_cfg.get("r", 16)

            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
            llm = LLM(**llm_kwargs)
            lora_request = LoRARequest("checkpoint", 1, args.checkpoint_path)
        else:
            print(f"Loading base model with vLLM: {args.model_name}")
            llm = LLM(**llm_kwargs)

        model = None  # not used in vLLM path
    else:
        print(f"Loading model with HF: {args.model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception:
            print("  flash-attn not available, falling back to default attention")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        if args.checkpoint_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter: {args.checkpoint_path}")
            model = PeftModel.from_pretrained(model, args.checkpoint_path)
            model = model.merge_and_unload()

        model.eval()
        llm = None  # not used in HF path

    # --- Generate completions ---
    print(f"\nGenerating {args.num_generations} completions per problem "
          f"({'vLLM' if use_vllm else 'HF generate'})...")

    if use_vllm:
        all_prompts = [example["prompt"] for example in dataset]
        all_completions = generate_all_vllm(
            llm=llm,
            prompts=all_prompts,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            lora_request=lora_request,
        )

    # --- Score and log ---
    rollout_fh = open(output_dir / "rollouts.jsonl", "w", encoding="utf-8")

    total_correct = 0
    total_completions = 0
    all_token_counts = []
    all_has_think = []
    all_has_boxed = []
    per_problem_results = []

    for idx, example in enumerate(dataset):
        problem = example["problem"]
        answer = example["answer"]
        prompt = example["prompt"]

        if use_vllm:
            completions = all_completions[idx]
        else:
            print(f"  [{idx + 1}/{len(dataset)}] {problem[:80]}...")
            completions = generate_completions_hf(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        problem_correct = 0
        for rollout_idx, comp in enumerate(completions):
            correctness = check_correctness(comp["completion_text"], answer)
            length_pen = compute_length_penalty(comp["num_tokens"], args.max_new_tokens)

            is_correct = correctness == 1.0
            problem_correct += int(is_correct)
            total_correct += int(is_correct)
            total_completions += 1
            all_token_counts.append(comp["num_tokens"])
            all_has_think.append(comp["has_think_tags"])
            all_has_boxed.append(comp["has_boxed"])

            rollout_record = {
                "problem_index": idx,
                "rollout_index": rollout_idx,
                "problem_text": problem[:300],
                "completion_text": comp["completion_text"],
                "ground_truth": answer,
                "correctness_reward": correctness,
                "length_penalty": length_pen,
                "num_tokens": comp["num_tokens"],
                "has_think_tags": comp["has_think_tags"],
                "has_boxed": comp["has_boxed"],
                "checkpoint": args.checkpoint_path or "base",
                "timestamp": datetime.now().isoformat(),
            }
            rollout_fh.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")

        problem_accuracy = problem_correct / len(completions)
        mean_tokens = sum(c["num_tokens"] for c in completions) / len(completions)
        per_problem_results.append({
            "index": idx,
            "problem": problem[:200],
            "answer": answer,
            "accuracy": problem_accuracy,
            "mean_tokens": mean_tokens,
            "num_correct": problem_correct,
            "num_completions": len(completions),
        })

        if use_vllm and (idx + 1) % 10 == 0:
            running_acc = total_correct / total_completions
            print(f"  [{idx + 1}/{len(dataset)}] running_acc={running_acc:.3f} "
                  f"mean_tokens={sum(all_token_counts) / len(all_token_counts):.0f}")

    rollout_fh.close()

    # --- Summary ---
    overall_accuracy = total_correct / total_completions if total_completions > 0 else 0
    sorted_tokens = sorted(all_token_counts)
    mean_tokens = sum(all_token_counts) / len(all_token_counts)
    median_tokens = sorted_tokens[len(sorted_tokens) // 2]

    summary = {
        "model": args.model_name,
        "checkpoint": args.checkpoint_path or "base",
        "num_problems": len(dataset),
        "num_generations_per_problem": args.num_generations,
        "total_completions": total_completions,
        "overall_accuracy": overall_accuracy,
        "mean_tokens": mean_tokens,
        "median_tokens": median_tokens,
        "min_tokens": min(all_token_counts),
        "max_tokens": max(all_token_counts),
        "std_tokens": (
            sum((t - mean_tokens) ** 2 for t in all_token_counts) / len(all_token_counts)
        ) ** 0.5,
        "frac_has_think": sum(all_has_think) / len(all_has_think),
        "frac_has_boxed": sum(all_has_boxed) / len(all_has_boxed),
        "difficulty_range": [args.min_solved_pct, args.max_solved_pct],
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "backend": "vllm" if use_vllm else "hf",
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "per_problem.jsonl", "w", encoding="utf-8") as f:
        for r in per_problem_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS (Qwen3)")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_name}")
    print(f"  Checkpoint: {args.checkpoint_path or 'base (no adapter)'}")
    print(f"  Backend: {'vLLM' if use_vllm else 'HF generate'}")
    print(f"  Difficulty: {args.min_solved_pct}-{args.max_solved_pct}%")
    print(f"  Problems: {len(dataset)}")
    print(f"  Completions: {total_completions}")
    print(f"  Overall accuracy: {overall_accuracy:.4f}")
    print(f"  Mean tokens: {mean_tokens:.0f}")
    print(f"  Median tokens: {median_tokens}")
    print(f"  Min/Max tokens: {min(all_token_counts)}/{max(all_token_counts)}")
    print(f"  Frac with <think> tags: {summary['frac_has_think']:.3f}")
    print(f"  Frac with \\boxed{{}}: {summary['frac_has_boxed']:.3f}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
