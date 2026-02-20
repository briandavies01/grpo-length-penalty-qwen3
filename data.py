"""Dataset loading and formatting for GRPO training with Qwen3.

Uses lime-nlp/deepscaleR_difficulty dataset (DeepScaleR with difficulty scores).
Filters by solved_percentage range (configurable, default 50-95%).

Qwen3 chat format uses <|im_start|>/<|im_end|> delimiters.
No BOS token (Qwen3 tokenizer has bos_token=null, add_bos_token=false).
"""

from datasets import Dataset, load_dataset

MIN_SOLVED_PERCENTAGE = 50.0
MAX_SOLVED_PERCENTAGE = 95.0

# Qwen3 special tokens
IM_START = "<|im_start|>"  # ID 151644
IM_END = "<|im_end|>"      # ID 151645

INSTRUCTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def format_prompt(problem: str) -> str:
    """Format a math problem as a Qwen3 chat prompt.
    Result: <|im_start|>user\n{problem}{instruction}<|im_end|>\n<|im_start|>assistant\n
    """
    return f"{IM_START}user\n{problem}{INSTRUCTION_SUFFIX}{IM_END}\n{IM_START}assistant\n"


def load_and_format_dataset(
    dataset_name: str = "lime-nlp/deepscaleR_difficulty",
    dataset_config: str = "Difficulty Score",
    split: str = "train",
    min_solved_pct: float = MIN_SOLVED_PERCENTAGE,
    max_solved_pct: float = MAX_SOLVED_PERCENTAGE,
) -> Dataset:
    """Load DeepScaleR with difficulty scores, filter to sweet-spot problems, format for GRPOTrainer."""
    ds = load_dataset(dataset_name, dataset_config, split=split)
    total_before = len(ds)
    ds = ds.filter(lambda x: min_solved_pct <= x["solved_percentage"] <= max_solved_pct)
    print(f"  Difficulty filter: {len(ds)}/{total_before} problems with solved_pct in [{min_solved_pct}, {max_solved_pct}]")

    def _format(example):
        return {
            "prompt": format_prompt(example["problem"]),
            "solution": str(example["ground_truth"]),
        }

    columns_to_remove = [c for c in ds.column_names if c not in ("prompt", "solution")]
    ds = ds.map(_format, remove_columns=columns_to_remove)
    return ds


def sample_baseline_problems(
    dataset_name: str = "lime-nlp/deepscaleR_difficulty",
    dataset_config: str = "Difficulty Score",
    split: str = "train",
    num_problems: int = 100,
    seed: int = 42,
    min_solved_pct: float = MIN_SOLVED_PERCENTAGE,
    max_solved_pct: float = MAX_SOLVED_PERCENTAGE,
) -> Dataset:
    """Sample a fixed subset of sweet-spot problems for baseline evaluation."""
    ds = load_dataset(dataset_name, dataset_config, split=split)
    total_before = len(ds)
    ds = ds.filter(lambda x: min_solved_pct <= x["solved_percentage"] <= max_solved_pct)
    print(f"  Difficulty filter: {len(ds)}/{total_before} problems with solved_pct in [{min_solved_pct}, {max_solved_pct}]")
    ds = ds.shuffle(seed=seed).select(range(min(num_problems, len(ds))))

    def _format(example):
        return {
            "prompt": format_prompt(example["problem"]),
            "answer": str(example["ground_truth"]),
        }

    columns_to_remove = [c for c in ds.column_names if c not in ("problem", "answer", "prompt")]
    ds = ds.map(_format, remove_columns=columns_to_remove)
    return ds


def verify_prompt_tokenization(tokenizer, sample_prompt: str) -> None:
    """Verify that a formatted prompt tokenizes correctly for Qwen3.
    Checks special token encoding and prompt structure."""
    token_ids = tokenizer.encode(sample_prompt)
    decoded_tokens = [tokenizer.decode([t]) for t in token_ids]

    print(f"\n  Prompt tokenization check ({len(token_ids)} tokens):")
    print(f"    First 10 tokens: {decoded_tokens[:10]}")
    print(f"    Last 5 tokens:   {decoded_tokens[-5:]}")

    # Check 1: <|im_start|> is a single token (ID 151644)
    im_start_id = tokenizer.encode("<|im_start|>")
    assert len(im_start_id) == 1 and im_start_id[0] == 151644, (
        f"<|im_start|> should be single token 151644, got {im_start_id}"
    )
    print("    [OK] <|im_start|> is single token 151644")

    # Check 2: <|im_end|> is a single token (ID 151645)
    im_end_id = tokenizer.encode("<|im_end|>")
    assert len(im_end_id) == 1 and im_end_id[0] == 151645, (
        f"<|im_end|> should be single token 151645, got {im_end_id}"
    )
    print("    [OK] <|im_end|> is single token 151645")

    # Check 3: No <think> token in prompt (ID 151667 for Qwen3)
    think_id = 151667
    assert think_id not in token_ids, (
        f"Prompt should NOT contain <think> token (ID {think_id})"
    )
    print("    [OK] No <think> token in prompt")

    # Check 4: First token is <|im_start|> (no BOS -- Qwen3 has none)
    assert token_ids[0] == 151644, (
        f"First token should be <|im_start|> (151644), got {token_ids[0]} = '{decoded_tokens[0]}'"
    )
    print("    [OK] First token is <|im_start|> (no BOS)")
