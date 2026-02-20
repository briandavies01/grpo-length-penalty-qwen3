"""Pre-flight checks for Qwen3 GRPO training.

Run this FIRST after SSH'ing into the RunPod instance and copying files over.
Tests everything in layers, from pure logic (no deps) up to GPU inference.

Special focus on tokenization: verifies that our raw-string prompt format
produces identical tokenization across HF tokenizer and vLLM, checks special
token IDs, BOS handling, stop tokens, and thinking mode activation.

Usage:
    python preflight.py              # Run all checks
    python preflight.py --skip-gpu   # Skip GPU/model checks (faster)
    python preflight.py --skip-wandb # Skip wandb login check
    python preflight.py --skip-vllm  # Skip vLLM tokenization comparison
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# ============================================================================
# Helpers
# ============================================================================

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================================
# Phase 1: Environment
# ============================================================================

def check_environment():
    section("Phase 1: Environment")

    # Python version
    v = sys.version_info
    check("Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")

    # CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        check("PyTorch installed", True, torch.__version__)
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            check("CUDA available", True, f"{gpu_count}x {gpu_name} ({gpu_mem_gb:.1f} GB each)")
            check("bf16 supported", torch.cuda.is_bf16_supported(), "")
        else:
            check("CUDA available", False, "No GPU detected")
    except ImportError:
        check("PyTorch installed", False, "pip install torch")
        return

    # Key packages
    packages = {
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "wandb": "wandb",
        "math_verify": "math-verify",
    }
    for import_name, pip_name in packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            check(f"{pip_name} installed", True, ver)
        except ImportError:
            check(f"{pip_name} installed", False, f"pip install {pip_name}")

    # Optional packages (warn but don't fail)
    try:
        import vllm
        check("vllm installed", True, vllm.__version__)
    except ImportError:
        print(f"  [{WARN}] vllm not installed — use --no_vllm flag or pip install vllm")

    try:
        import flash_attn
        check("flash-attn installed", True, flash_attn.__version__)
    except ImportError:
        print(f"  [{WARN}] flash-attn not installed — model will use default attention (slower)")


# ============================================================================
# Phase 2: Pure Python logic (no network, no GPU)
# ============================================================================

def check_logic():
    section("Phase 2: Pure Python Logic (offline)")

    # --- Prompt formatting ---
    from data import format_prompt, IM_START, IM_END, INSTRUCTION_SUFFIX

    # Verify Qwen3 special token constants
    check("IM_START is '<|im_start|>'",
          IM_START == "<|im_start|>",
          f"got {IM_START!r}")
    check("IM_END is '<|im_end|>'",
          IM_END == "<|im_end|>",
          f"got {IM_END!r}")

    # Verify INSTRUCTION_SUFFIX has literal backslash (not backspace)
    check("INSTRUCTION_SUFFIX has literal \\\\boxed (not \\x08)",
          "\\boxed{}" in INSTRUCTION_SUFFIX and "\x08" not in INSTRUCTION_SUFFIX,
          f"repr={INSTRUCTION_SUFFIX!r}")

    # Verify format_prompt output
    test_problem = "What is 2 + 2?"
    prompt = format_prompt(test_problem)

    check("format_prompt starts with <|im_start|>user",
          prompt.startswith("<|im_start|>user\n"),
          f"first 30: {prompt[:30]!r}")
    check("format_prompt ends with <|im_start|>assistant\\n",
          prompt.endswith("<|im_start|>assistant\n"),
          f"last 30: {prompt[-30:]!r}")
    check("format_prompt contains <|im_end|>",
          "<|im_end|>" in prompt, "")
    check("format_prompt contains problem text",
          test_problem in prompt, "")
    check("format_prompt contains \\boxed instruction",
          "\\boxed{}" in prompt, "")
    check("format_prompt does NOT contain BOS",
          "<|begin" not in prompt and "begin_of_sentence" not in prompt,
          "Qwen3 has no BOS token")
    check("format_prompt does NOT contain <think>",
          "<think>" not in prompt,
          "<think> should be generated by the model, not in the prompt")

    # Verify prompt structure: user msg then assistant prefix
    # Expected: <|im_start|>user\n{problem}{suffix}<|im_end|>\n<|im_start|>assistant\n
    expected = f"<|im_start|>user\n{test_problem}{INSTRUCTION_SUFFIX}<|im_end|>\n<|im_start|>assistant\n"
    check("format_prompt matches expected structure",
          prompt == expected,
          f"len={len(prompt)}, expected_len={len(expected)}")

    # --- Answer extraction ---
    from rewards import extract_answer_text

    text1 = "<think>\nLet me work...\n\\boxed{wrong}\n</think>\n\nThe answer is \\boxed{42}."
    out1 = extract_answer_text(text1)
    check("extract_answer_text: strips think block",
          "\\boxed{wrong}" not in out1 and "\\boxed{42}" in out1,
          f"got: {out1[:80]!r}")

    text2 = "<think>\nLet me work on this problem..."
    out2 = extract_answer_text(text2)
    check("extract_answer_text: truncated think -> empty",
          out2 == "", f"got: {out2!r}")

    text3 = "The answer is \\boxed{7}."
    out3 = extract_answer_text(text3)
    check("extract_answer_text: no tags -> full text",
          out3 == text3, f"got: {out3[:80]!r}")

    # --- Length penalty ---
    from rewards import compute_length_penalty
    check("length_penalty(4096, 8192) == -0.5",
          compute_length_penalty(4096, 8192) == -0.5, "")
    check("length_penalty(0, 8192) == 0.0",
          compute_length_penalty(0, 8192) == 0.0, "")
    check("length_penalty(8192, 8192) == -1.0",
          compute_length_penalty(8192, 8192) == -1.0, "")

    # --- Config ---
    from config import ExperimentConfig
    cfg = ExperimentConfig(lambda_length=0.3, max_steps=3)

    check("Config model_name is Qwen3",
          "Qwen3" in cfg.model_name or "qwen3" in cfg.model_name.lower(),
          f"got {cfg.model_name}")

    grpo_cfg = cfg.build_grpo_config()
    check("GRPOConfig builds without error", True,
          f"loss_type={grpo_cfg.loss_type}, scale_rewards={grpo_cfg.scale_rewards}")
    check("GRPOConfig.loss_type is dr_grpo",
          grpo_cfg.loss_type == "dr_grpo", "")
    check("GRPOConfig.scale_rewards is 'group'",
          grpo_cfg.scale_rewards == "group" and isinstance(grpo_cfg.scale_rewards, str), "")
    check("GRPOConfig.beta is 0.0 (no KL)",
          grpo_cfg.beta == 0.0, "")
    check("GRPOConfig.remove_unused_columns is False",
          grpo_cfg.remove_unused_columns is False,
          "Critical: preserves 'solution' column for reward fn")
    check("GRPOConfig.max_completion_length > 256",
          grpo_cfg.max_completion_length > 256,
          f"got {grpo_cfg.max_completion_length}")

    # Check stop tokens
    gen_kwargs = grpo_cfg.generation_kwargs or {}
    stop_ids = gen_kwargs.get("stop_token_ids", [])
    check("stop_token_ids contains 151645 (<|im_end|>)",
          151645 in stop_ids,
          f"got {stop_ids}")
    check("stop_token_ids contains 151643 (EOS)",
          151643 in stop_ids,
          f"got {stop_ids}")

    # --- LoRA Config ---
    lora_cfg = cfg.build_lora_config()
    check("LoraConfig builds without error", True,
          f"r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, targets={list(lora_cfg.target_modules)}")
    check("LoRA rank >= 8",
          lora_cfg.r >= 8,
          f"got {lora_cfg.r}")
    check("LoRA targets include attention",
          "q_proj" in lora_cfg.target_modules and "v_proj" in lora_cfg.target_modules,
          f"targets: {list(lora_cfg.target_modules)}")
    check("LoRA targets include MLP",
          "gate_proj" in lora_cfg.target_modules,
          f"targets: {list(lora_cfg.target_modules)}")
    check("LoRA does NOT target embed_tokens (tie_word_embeddings)",
          "embed_tokens" not in (lora_cfg.target_modules or []),
          "Qwen3 has tie_word_embeddings=True, targeting embeddings would break tying")
    check("LoRA does NOT target lm_head (tie_word_embeddings)",
          "lm_head" not in (lora_cfg.target_modules or []),
          "Qwen3 has tie_word_embeddings=True")


# ============================================================================
# Phase 3: Math-verify
# ============================================================================

def check_math_verify():
    section("Phase 3: math-verify answer checking")

    from rewards import check_correctness

    check("math-verify: boxed integer match",
          check_correctness("The answer is \\boxed{42}.", "42") == 1.0, "")
    check("math-verify: boxed fraction match",
          check_correctness("\\boxed{\\frac{1}{2}}", "0.5") == 1.0, "")
    check("math-verify: boxed decimal match",
          check_correctness("\\boxed{0.5}", "\\frac{1}{2}") == 1.0, "")
    check("math-verify: wrong answer -> 0.0",
          check_correctness("\\boxed{43}", "42") == 0.0, "")
    check("math-verify: empty text -> 0.0",
          check_correctness("", "42") == 0.0, "")
    check("math-verify: exception safety (malformed)",
          check_correctness("\\boxed{\\invalid{", "42") == 0.0,
          "should return 0.0, not raise")

    # Think-tag stripping
    think_comp = "<think>Let me try \\boxed{999}</think>\n\nThe answer is \\boxed{42}."
    check("math-verify: ignores \\boxed inside <think>",
          check_correctness(think_comp, "42") == 1.0, "")
    check("math-verify: think \\boxed{999} not matched against 999",
          check_correctness(think_comp, "999") == 0.0,
          "Should match \\boxed{42} after </think>, not \\boxed{999} inside <think>")


# ============================================================================
# Phase 4: HuggingFace Tokenizer — Deep Verification
# ============================================================================

def check_tokenizer():
    section("Phase 4: Qwen3 Tokenizer — Deep Verification")

    print(f"  [{INFO}] Downloading Qwen3 tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        check("Tokenizer downloaded", True, f"vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        check("Tokenizer downloaded", False, str(e)[:100])
        return None

    # --- Tokenizer config checks ---
    native_bos = getattr(tokenizer, "add_bos_token", None)
    check("add_bos_token is False (Qwen3 has no BOS)",
          native_bos is False or native_bos is None,
          f"got {native_bos}")

    bos_token = getattr(tokenizer, "bos_token", "MISSING")
    check("bos_token value",
          True,  # Informational
          f"bos_token={bos_token!r} (may be None or set but unused)")

    eos_token = getattr(tokenizer, "eos_token", None)
    check("eos_token exists",
          eos_token is not None,
          f"eos_token={eos_token!r}")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    check("eos_token_id is 151645 (<|im_end|>)",
          eos_id == 151645,
          f"got {eos_id} — Qwen3 uses <|im_end|> as its EOS token")

    # Set up tokenizer as grpo_train.py does
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Special token ID verification ---
    print(f"\n  [{INFO}] Verifying special token IDs...")

    # <|im_start|> = 151644
    im_start_ids = tokenizer.encode("<|im_start|>")
    check("<|im_start|> is single token ID 151644",
          len(im_start_ids) == 1 and im_start_ids[0] == 151644,
          f"got {im_start_ids}")

    # <|im_end|> = 151645
    im_end_ids = tokenizer.encode("<|im_end|>")
    check("<|im_end|> is single token ID 151645",
          len(im_end_ids) == 1 and im_end_ids[0] == 151645,
          f"got {im_end_ids}")

    # <think> = 151667
    think_ids = tokenizer.encode("<think>")
    check("<think> token ID",
          True,  # Informational
          f"encode('<think>') = {think_ids}")
    if len(think_ids) == 1:
        check("<think> is single token",
              True, f"ID={think_ids[0]}")
    else:
        check("<think> is single token",
              False, f"got {len(think_ids)} tokens: {think_ids}")

    # </think>
    think_close_ids = tokenizer.encode("</think>")
    check("</think> token ID",
          True,
          f"encode('</think>') = {think_close_ids}")

    # --- Full prompt tokenization ---
    print(f"\n  [{INFO}] Testing full prompt tokenization...")
    from data import format_prompt

    test_prompt = format_prompt("Let $n$ be a positive integer such that $n > 5$. Find $n$.")
    token_ids = tokenizer.encode(test_prompt)
    decoded_tokens = [tokenizer.decode([t]) for t in token_ids]

    print(f"  [{INFO}] Prompt length: {len(token_ids)} tokens")
    print(f"  [{INFO}] First 15 tokens (IDs): {token_ids[:15]}")
    print(f"  [{INFO}] First 15 tokens (text): {decoded_tokens[:15]}")
    print(f"  [{INFO}] Last 10 tokens (IDs):  {token_ids[-10:]}")
    print(f"  [{INFO}] Last 10 tokens (text):  {decoded_tokens[-10:]}")

    # Check 1: First token is <|im_start|> (NOT a BOS token)
    check("First token is <|im_start|> (151644), not BOS",
          token_ids[0] == 151644,
          f"got {token_ids[0]} = {decoded_tokens[0]!r}")

    # Check 2: No spurious BOS anywhere
    # Common BOS IDs: 1 (Llama), 151643 (some Qwen configs)
    # We check that position 0 is im_start and no extra BOS was prepended
    if len(token_ids) > 1:
        check("Second token is NOT a duplicate <|im_start|>",
              token_ids[1] != 151644,
              f"got {token_ids[1]} = {decoded_tokens[1]!r}")

    # Check 3: <|im_end|> appears exactly once
    im_end_count = token_ids.count(151645)
    check("<|im_end|> appears exactly once in prompt",
          im_end_count == 1,
          f"found {im_end_count} occurrences")

    # Check 4: <|im_start|> appears exactly twice (user + assistant)
    im_start_count = token_ids.count(151644)
    check("<|im_start|> appears exactly twice (user + assistant)",
          im_start_count == 2,
          f"found {im_start_count} occurrences")

    # Check 5: No <think> token in prompt
    think_id = 151667
    check("No <think> token in prompt",
          think_id not in token_ids,
          f"<think> (ID {think_id}) should NOT be in the prompt")

    # Check 6: Prompt ends with expected sequence
    # Should be: ... <|im_start|>(151644) "assistant"(...) "\n"(...)
    # The last token should be the newline after "assistant"
    last_few = decoded_tokens[-5:]
    check("Prompt ends near 'assistant'",
          any("assistant" in t for t in last_few),
          f"last 5 decoded: {last_few}")

    # Check 7: \\boxed is in the prompt (not eaten by tokenizer)
    full_decoded = tokenizer.decode(token_ids)
    check("Round-trip decode contains \\boxed",
          "\\boxed" in full_decoded or "boxed" in full_decoded,
          f"'boxed' in decoded: {'boxed' in full_decoded}")

    # Check 8: Round-trip fidelity — encode then decode should preserve content
    # (Qwen tokenizers sometimes mangle special chars)
    check("Round-trip preserves <|im_start|>",
          "<|im_start|>" in full_decoded,
          "")
    check("Round-trip preserves <|im_end|>",
          "<|im_end|>" in full_decoded,
          "")

    # --- Full tokenization verification from data.py ---
    print(f"\n  [{INFO}] Running data.verify_prompt_tokenization()...")
    from data import verify_prompt_tokenization
    try:
        verify_prompt_tokenization(tokenizer, test_prompt)
        check("data.verify_prompt_tokenization passed", True, "all checks OK")
    except AssertionError as e:
        check("data.verify_prompt_tokenization passed", False, str(e)[:150])
    except Exception as e:
        check("data.verify_prompt_tokenization passed", False, f"Unexpected: {e}")

    # --- Compare with chat template ---
    print(f"\n  [{INFO}] Comparing raw format vs chat template...")
    try:
        messages = [{"role": "user", "content": "Let $n$ be a positive integer such that $n > 5$. Find $n$.\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        template_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        template_ids = tokenizer.encode(template_prompt)

        print(f"  [{INFO}] Chat template prompt ({len(template_ids)} tokens):")
        print(f"  [{INFO}]   {template_prompt[:200]!r}")
        print(f"  [{INFO}] Our raw prompt ({len(token_ids)} tokens):")
        print(f"  [{INFO}]   {test_prompt[:200]!r}")

        # Check if they're identical or explain differences
        if template_ids == token_ids:
            check("Raw format == chat template", True, "identical tokenization!")
        else:
            # Find first difference
            for i, (a, b) in enumerate(zip(template_ids, token_ids)):
                if a != b:
                    check("Raw format == chat template", False,
                          f"first diff at pos {i}: template={a}({tokenizer.decode([a])!r}) vs raw={b}({tokenizer.decode([b])!r})")
                    break
            else:
                len_diff = len(template_ids) - len(token_ids)
                check("Raw format == chat template", False,
                      f"same prefix but length differs by {len_diff}: template={len(template_ids)}, raw={len(token_ids)}")

            # Show the template for manual inspection
            print(f"  [{INFO}] Template first 15 IDs: {template_ids[:15]}")
            print(f"  [{INFO}] Raw      first 15 IDs: {token_ids[:15]}")
            print(f"  [{INFO}] Template last  10 IDs: {template_ids[-10:]}")
            print(f"  [{INFO}] Raw      last  10 IDs: {token_ids[-10:]}")

            # Even if they differ, we should check if the key structure matches
            check("Both start with <|im_start|> (151644)",
                  template_ids[0] == 151644 and token_ids[0] == 151644,
                  "")

    except Exception as e:
        print(f"  [{WARN}] Chat template comparison failed: {e}")
        print(f"  [{INFO}] This may be OK if the model doesn't support enable_thinking")

    # --- Dataset access ---
    print(f"\n  [{INFO}] Checking dataset access...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "lime-nlp/deepscaleR_difficulty",
            "Difficulty Score",
            split="train",
            streaming=True,
        )
        first = next(iter(ds))
        has_problem = "problem" in first
        has_ground_truth = "ground_truth" in first
        has_solved_pct = "solved_percentage" in first
        check("Dataset accessible", True, f"columns={list(first.keys())}")
        check("Dataset has 'problem' column", has_problem, "")
        check("Dataset has 'ground_truth' column", has_ground_truth, "")
        check("Dataset has 'solved_percentage' column", has_solved_pct,
              f"value={first.get('solved_percentage', 'MISSING')}")
    except Exception as e:
        check("Dataset accessible", False, str(e)[:100])

    return tokenizer


# ============================================================================
# Phase 5: vLLM Tokenization Comparison
# ============================================================================

def check_vllm_tokenization(tokenizer):
    section("Phase 5: vLLM vs HF Tokenization")

    if tokenizer is None:
        print(f"  [{WARN}] Skipping — tokenizer not available")
        return

    try:
        import vllm
    except ImportError:
        print(f"  [{WARN}] vllm not installed — skipping")
        return

    from data import format_prompt

    # vLLM loads its own tokenizer internally. We need to verify that
    # vLLM's tokenizer produces the same token IDs as our HF tokenizer.
    print(f"  [{INFO}] Loading vLLM tokenizer for comparison...")

    try:
        from vllm import LLM, SamplingParams

        # Force V0 engine like our training does
        os.environ["VLLM_USE_V1"] = "0"

        llm = LLM(
            model="Qwen/Qwen3-1.7B",
            max_model_len=2048,  # Small for preflight
            gpu_memory_utilization=0.3,
            dtype="bfloat16",
            trust_remote_code=True,
            model_impl="transformers",  # Same as our training config
        )

        # Get vLLM's tokenizer
        vllm_tokenizer = llm.get_tokenizer()

        test_prompt = format_prompt("What is 2 + 2?")

        # Tokenize with both
        hf_ids = tokenizer.encode(test_prompt)
        vllm_ids = vllm_tokenizer.encode(test_prompt)

        check("vLLM tokenizer loaded", True,
              f"type={type(vllm_tokenizer).__name__}")

        if hf_ids == vllm_ids:
            check("HF == vLLM tokenization", True,
                  f"both produce {len(hf_ids)} tokens")
        else:
            check("HF == vLLM tokenization", False,
                  f"HF={len(hf_ids)} tokens, vLLM={len(vllm_ids)} tokens")
            # Find first diff
            for i, (a, b) in enumerate(zip(hf_ids, vllm_ids)):
                if a != b:
                    print(f"  [{INFO}] First diff at pos {i}: HF={a} vs vLLM={b}")
                    break
            print(f"  [{INFO}] HF   first 10: {hf_ids[:10]}")
            print(f"  [{INFO}] vLLM first 10: {vllm_ids[:10]}")

        # Check first token specifically — BOS handling is the #1 risk
        check("vLLM first token is <|im_start|> (151644)",
              vllm_ids[0] == 151644,
              f"got {vllm_ids[0]}")
        check("vLLM does NOT prepend BOS",
              len(vllm_ids) == len(hf_ids) or vllm_ids[0] == 151644,
              f"vLLM={len(vllm_ids)} tokens, HF={len(hf_ids)} tokens")

        # Test that vLLM generation actually works and starts with <think>
        print(f"\n  [{INFO}] Testing vLLM generation (short, 256 tokens)...")
        sampling_params = SamplingParams(
            n=1,
            temperature=0.6,
            top_p=0.95,
            max_tokens=256,
            stop_token_ids=[151645, 151643],
        )
        outputs = llm.generate([test_prompt], sampling_params)
        if outputs and outputs[0].outputs:
            gen_text = outputs[0].outputs[0].text
            gen_ids = list(outputs[0].outputs[0].token_ids)
            check("vLLM generation produced output", True,
                  f"{len(gen_ids)} tokens")
            check("vLLM output starts with <think>",
                  gen_text.strip().startswith("<think>"),
                  f"first 60: {gen_text[:60]!r}")

            # Check stop token handling
            if gen_ids:
                last_id = gen_ids[-1]
                stopped_at_eos = last_id in (151643, 151645)
                check("vLLM stopped at EOS/im_end or hit max",
                      stopped_at_eos or len(gen_ids) == 256,
                      f"last token ID={last_id}, len={len(gen_ids)}")
        else:
            check("vLLM generation produced output", False, "no outputs")

        # Clean up vLLM
        del llm
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    except Exception as e:
        check("vLLM tokenization test", False, f"{type(e).__name__}: {str(e)[:150]}")
        print(f"  [{INFO}] Full error: {e}")


# ============================================================================
# Phase 6: Model Config Verification
# ============================================================================

def check_model_config():
    section("Phase 6: Qwen3 Model Config Verification")

    print(f"  [{INFO}] Downloading model config (no weights)...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
        check("Model config downloaded", True,
              f"arch={config.architectures}")
    except Exception as e:
        check("Model config downloaded", False, str(e)[:100])
        return

    # Architecture
    arch = getattr(config, "architectures", [])
    check("Architecture is Qwen3-family",
          any("Qwen" in a for a in arch),
          f"{arch}")

    # tie_word_embeddings — critical for LoRA targeting
    tie_embeddings = getattr(config, "tie_word_embeddings", None)
    check("tie_word_embeddings is True",
          tie_embeddings is True,
          f"got {tie_embeddings} — must NOT target embed_tokens/lm_head in LoRA")

    # Vocab size
    vocab_size = getattr(config, "vocab_size", None)
    check("vocab_size", True,  # Informational
          f"{vocab_size}")

    # Hidden size / num layers
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    print(f"  [{INFO}] hidden_size={hidden_size}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}")

    # Max position embeddings
    max_pos = getattr(config, "max_position_embeddings", None)
    check("max_position_embeddings >= 9216 (prompt + completion)",
          max_pos is not None and max_pos >= 9216,
          f"got {max_pos} (need 1024 prompt + 8192 completion = 9216)")

    # Check expected layer names exist
    # Qwen3 should have q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    print(f"  [{INFO}] Verifying LoRA target module names exist in architecture...")
    expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    try:
        from transformers import AutoModelForCausalLM
        import torch
        # Load with meta device to avoid downloading weights
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        for target in expected_modules:
            found = target in module_names
            check(f"Module '{target}' exists in model",
                  found,
                  "" if found else "LoRA target won't work!")
        del model
    except Exception as e:
        print(f"  [{WARN}] Could not verify module names: {e}")


# ============================================================================
# Phase 7: wandb
# ============================================================================

def check_wandb():
    section("Phase 7: Weights & Biases")

    try:
        import wandb
        check("wandb installed", True, wandb.__version__)

        if wandb.api.api_key:
            check("wandb logged in", True, "API key found")
        else:
            check("wandb logged in", False,
                  "Run: wandb login (or set WANDB_API_KEY env var)")
    except Exception as e:
        check("wandb check", False, str(e)[:100])


# ============================================================================
# Phase 8: GPU Model Loading & Generation
# ============================================================================

def check_gpu_generation(tokenizer):
    section("Phase 8: GPU Model Loading & HF Generation")

    import torch
    if not torch.cuda.is_available():
        print(f"  [{WARN}] No GPU available — skipping")
        return

    from transformers import AutoModelForCausalLM
    from data import format_prompt

    model_name = "Qwen/Qwen3-1.7B"

    print(f"  [{INFO}] Loading model (this may take 1-2 minutes)...")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        load_time = time.time() - t0
        check("Model loaded", True, f"{load_time:.1f}s, flash_attention_2")
    except Exception as e:
        if "flash" in str(e).lower():
            print(f"  [{WARN}] Flash attention failed, trying without it...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            check("Model loaded (no flash-attn)", True, "")
        else:
            check("Model loaded", False, str(e)[:150])
            return

    model.eval()

    # GPU memory
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  [{INFO}] GPU memory: {mem_allocated:.1f} GB allocated, {mem_reserved:.1f} GB reserved")

    # Generate a completion
    prompt = format_prompt("What is 2 + 2?")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Verify input token IDs match our expectations
    input_ids = inputs["input_ids"][0].tolist()
    check("HF generate input starts with <|im_start|>",
          input_ids[0] == 151644,
          f"first ID={input_ids[0]}")
    check("HF generate input has no BOS prepended",
          input_ids[0] == 151644 and (len(input_ids) < 2 or input_ids[1] != 151644),
          f"first 3 IDs: {input_ids[:3]}")

    max_gen_tokens = 2048  # Shorter than full 8192 for preflight speed
    print(f"  [{INFO}] Generating completion (max {max_gen_tokens} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            eos_token_id=[151645, 151643],  # Stop at <|im_end|> or EOS
        )
    gen_time = time.time() - t0

    completion_ids = output[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    num_tokens = len(completion_ids)
    tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0

    check("HF generation completed", True,
          f"{num_tokens} tokens in {gen_time:.1f}s ({tokens_per_sec:.0f} tok/s)")

    # Check completion format — critical for GRPO training
    has_think_open = "<think>" in completion
    has_think_close = "</think>" in completion
    has_boxed = "\\boxed" in completion
    starts_with_think = completion.strip().startswith("<think>")

    check("Completion starts with <think>",
          starts_with_think,
          f"first 80: {completion[:80]!r}")

    if not starts_with_think:
        print(f"  [{WARN}] *** CRITICAL: Model did not enter thinking mode! ***")
        print(f"  [{WARN}] This means GRPO training will not produce CoT reasoning.")
        print(f"  [{WARN}] You may need to prefill '<think>\\n' in the prompt.")

    if has_think_open and has_think_close:
        check("Completion has matching </think>", True, "")
    elif has_think_open:
        print(f"  [{WARN}] <think> found but no </think> (may be truncated at {num_tokens} tokens)")

    check("Completion has \\boxed{}",
          has_boxed,
          f"{'found' if has_boxed else 'NOT found'}")

    # End-to-end correctness check
    from rewards import check_correctness
    correctness = check_correctness(completion, "4")
    check("check_correctness(completion, '4') works",
          correctness in (0.0, 1.0),
          f"returned {correctness} ({'correct' if correctness == 1.0 else 'incorrect'})")

    print(f"\n  [{INFO}] === SAMPLE COMPLETION (first 500 chars) ===")
    print(f"  {completion[:500]}")
    if len(completion) > 500:
        print(f"  [...truncated, {len(completion)} chars total]")
    print(f"  [{INFO}] === END COMPLETION ===")

    # Clean up
    del model
    torch.cuda.empty_cache()


# ============================================================================
# Phase 9: TRL GRPOTrainer smoke test
# ============================================================================

def check_trl():
    section("Phase 9: TRL Configuration")

    try:
        from trl import GRPOTrainer, GRPOConfig
        check("GRPOTrainer importable", True, "")
        check("GRPOConfig importable", True, "")
    except ImportError as e:
        check("TRL import", False, str(e)[:100])
        return

    # Check our config params are valid
    try:
        cfg = GRPOConfig(
            output_dir="/tmp/test_grpo_qwen3",
            loss_type="dr_grpo",
            scale_rewards="group",
            beta=0.0,
            num_generations=4,
            max_completion_length=512,
            num_iterations=1,
            epsilon=0.2,
            use_vllm=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1,
            remove_unused_columns=False,
            report_to="none",
            # Qwen3-specific stop tokens
            generation_kwargs={"stop_token_ids": [151645, 151643]},
        )
        check("GRPOConfig with Qwen3 params", True,
              f"loss={cfg.loss_type}, stop_ids={cfg.generation_kwargs}")
    except TypeError as e:
        check("GRPOConfig with Qwen3 params", False, f"Bad param: {e}")
    except Exception as e:
        check("GRPOConfig with Qwen3 params", False, str(e)[:150])

    # Check vLLM colocate mode
    try:
        cfg_vllm = GRPOConfig(
            output_dir="/tmp/test_grpo_vllm_qwen3",
            use_vllm=True,
            vllm_mode="colocate",
            vllm_model_impl="transformers",
            max_steps=1,
            report_to="none",
        )
        check("GRPOConfig vllm_mode='colocate' + model_impl='transformers'", True, "")
    except Exception as e:
        check("GRPOConfig vllm colocate", False, str(e)[:100])

    # Check multi-objective config
    try:
        cfg_mo = GRPOConfig(
            output_dir="/tmp/test_grpo_mo_qwen3",
            multi_objective_aggregation="normalize_then_sum",
            reward_weights=[1.0, 0.3],
            max_steps=1,
            report_to="none",
        )
        check("GRPOConfig normalize_then_sum + reward_weights", True,
              f"weights={cfg_mo.reward_weights}")
    except Exception as e:
        check("GRPOConfig multi-objective", False, str(e)[:100])


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    section("SUMMARY")

    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)
    total = len(results)

    print(f"\n  {passed}/{total} checks passed, {failed} failed\n")

    if failed > 0:
        print(f"  Failed checks:")
        for name, p in results:
            if not p:
                print(f"    - {name}")
        print()

    if failed == 0:
        print(f"  All checks passed! Ready to run Qwen3 GRPO training:")
        print(f"    1. python grpo_train.py --lambda_length 0.0 --max_steps 3 --num_generations 4 --max_completion_length 512  # trial")
        print(f"    2. python grpo_train.py --lambda_length 0.05 --max_steps 200 --num_generations 10 --gradient_accumulation_steps 6  # real run")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3 GRPO pre-flight checks")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip GPU model loading/generation tests")
    parser.add_argument("--skip-wandb", action="store_true",
                        help="Skip wandb login check")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM tokenization comparison (saves GPU memory)")
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  QWEN3 GRPO — PRE-FLIGHT CHECKS")
    print("#" * 60)

    # Phase 1: Environment
    check_environment()

    # Phase 2: Pure logic
    check_logic()

    # Phase 3: Math-verify
    check_math_verify()

    # Phase 4: Tokenizer deep verification
    tokenizer = check_tokenizer()

    # Phase 5: vLLM tokenization comparison
    if not args.skip_vllm and not args.skip_gpu:
        check_vllm_tokenization(tokenizer)
    else:
        print(f"\n  [{INFO}] Skipping vLLM tokenization (--skip-vllm or --skip-gpu)")

    # Phase 6: Model config
    check_model_config()

    # Phase 7: wandb
    if not args.skip_wandb:
        check_wandb()
    else:
        print(f"\n  [{INFO}] Skipping wandb check (--skip-wandb)")

    # Phase 8: GPU generation
    if not args.skip_gpu:
        if tokenizer is not None:
            check_gpu_generation(tokenizer)
        else:
            print(f"\n  [{WARN}] Skipping GPU check — tokenizer not available")
    else:
        print(f"\n  [{INFO}] Skipping GPU check (--skip-gpu)")

    # Phase 9: TRL config
    check_trl()

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
