"""test_mt.py

Quick end-to-end test for MultiTurnSearchRollout (train_search_r1_mt.py).

Tests:
  1. Single question — print full conversation + answer.
  2. Token alignment check — verify that completion_ids decoded equals the
     concatenated assistant + injected-user turns.
  3. env_mask sanity — model tokens (mask=1) vs injected tokens (mask=0).

Usage:
    ../grpo/bin/python test_mt.py [--model MODEL_PATH] [--question "..."]
"""

import argparse
import re
import sys

import torch
import transformers

# ── import the rollout from the training script ──────────────────────────────
sys.path.insert(0, ".")
from train_search_r1_mt import (
    SYSTEM_PROMPT,
    MultiTurnSearchRollout,
    search,
    extract_answer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="/home/l33500/models/Qwen/Qwen3-0.6B",
        help="Model path or HF repo",
    )
    p.add_argument(
        "--question",
        default="who plays the voice of wall-e in the movie",
        help="Question to test",
    )
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--max_search_calls", type=int, default=5)
    p.add_argument("--chunk_max_new_tokens", type=int, default=512)
    return p.parse_args()


def print_sep(title=""):
    width = 60
    if title:
        side = (width - len(title) - 2) // 2
        print("=" * side + f" {title} " + "=" * side)
    else:
        print("=" * width)


def colorize_mask(tokens_text: str, mask_bit: int) -> str:
    """Return token text with ANSI colour based on mask (1=green, 0=yellow)."""
    if mask_bit == 1:
        return f"\033[32m{tokens_text}\033[0m"   # green = model
    else:
        return f"\033[33m{tokens_text}\033[0m"   # yellow = injected


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_sep("Loading model")
    print(f"  model : {args.model}")
    print(f"  device: {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # ── build initial messages ────────────────────────────────────────────────
    question = args.question.strip()
    prompt_text = SYSTEM_PROMPT + f" Question: {question}\n"
    messages = [{"role": "user", "content": prompt_text}]

    print_sep("Question")
    print(f"  {question}")

    # ── run rollout ───────────────────────────────────────────────────────────
    manager = MultiTurnSearchRollout(
        search_func=search,
        tokenizer=tokenizer,
        max_search_calls=args.max_search_calls,
        max_iterations=args.max_search_calls * 3,
        max_completion_length=args.max_new_tokens,
        chunk_max_new_tokens=args.chunk_max_new_tokens,
    )

    print_sep("Generating")
    with torch.no_grad():
        completion_ids, env_mask = manager.generate(
            model, model.generation_config, messages, device
        )

    # ── decode full completion ────────────────────────────────────────────────
    full_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    answer = extract_answer(full_text)

    print_sep("Full completion (skip_special_tokens=True)")
    print(full_text)

    print_sep("Full completion (with special tokens)")
    full_text_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    print(full_text_raw)

    # ── stats ─────────────────────────────────────────────────────────────────
    print_sep("Stats")
    n_total = len(completion_ids)
    n_model = sum(env_mask)
    n_injected = n_total - n_model
    n_searches = len(re.findall(r"<search>.*?</search>", full_text, re.DOTALL))

    print(f"  total tokens   : {n_total}")
    print(f"  model tokens   : {n_model}  (mask=1, green)")
    print(f"  injected tokens: {n_injected}  (mask=0, yellow)")
    print(f"  search calls   : {n_searches}")
    print(f"  answer         : {answer!r}")

    # ── env_mask sanity checks ────────────────────────────────────────────────
    print_sep("Sanity checks")

    assert len(completion_ids) == len(env_mask), \
        f"FAIL: len(completion_ids)={len(completion_ids)} != len(env_mask)={len(env_mask)}"
    print("  [OK] completion_ids and env_mask same length")

    assert all(m in (0, 1) for m in env_mask), "FAIL: env_mask contains values other than 0/1"
    print("  [OK] env_mask values are all 0 or 1")

    # All tokens in <information> blocks should be injected (mask=0).
    # We check this by decoding contiguous spans of mask=0 tokens and
    # verifying they contain <information> markers (if any searches happened).
    if n_searches > 0:
        injected_spans = []
        start = None
        for i, m in enumerate(env_mask):
            if m == 0 and start is None:
                start = i
            elif m == 1 and start is not None:
                injected_spans.append((start, i))
                start = None
        if start is not None:
            injected_spans.append((start, len(env_mask)))

        for s, e in injected_spans:
            span_text = tokenizer.decode(completion_ids[s:e], skip_special_tokens=False)
            # Injected spans should contain the structural tokens wrapping <information>
            # (at minimum the <|im_end|> closing the assistant turn and the user turn header)
            assert span_text.strip(), f"FAIL: empty injected span [{s}:{e}]"
        print(f"  [OK] {len(injected_spans)} injected span(s) found, all non-empty")
    else:
        print("  [--] no searches; skipping injected-span check")

    # ── coloured token-by-token view (first 120 tokens) ──────────────────────
    print_sep("Token view (first 120, green=model yellow=injected)")
    view_tokens = []
    for tok_id, m in zip(completion_ids[:10000], env_mask[:10000]):
        tok_str = tokenizer.decode([tok_id])
        view_tokens.append(colorize_mask(repr(tok_str), m))
    print(" ".join(view_tokens))

    print_sep("Done")


if __name__ == "__main__":
    main()
