"""batch_eval_mt.py

Batch evaluation script for models trained with train_search_r1_mt.py.

Uses the SAME multi-turn rollout (MultiTurnSearchRollout) and SAME prompt
format as training, so evaluation exactly replicates the training distribution.

Conversation pattern:
  user:      <system_prompt> Question: ...
  assistant: <thought>...</thought><search>query</search>
  user:      <information>\\nresults\\n</information>
  assistant: <thought>...</thought><answer>answer</answer>

Metrics:
  EM      — exact match (substring, normalised, same as training reward)
  F1      — token-level F1
  has_answer_rate — fraction where <answer> tag was present
  mean_search_calls
  mean_completion_length (model-generated tokens only, mask=1)

Usage:
  # single model, 200 NQ samples
  ../grpo/bin/python batch_eval_mt.py \\
      --model_name_or_path outputs/search-r1-mt \\
      --num_samples 200

  # compare checkpoints
  ../grpo/bin/python batch_eval_mt.py \\
      --model_name_or_path \\
          /home/l33500/models/Qwen/Qwen3-0.6B \\
          outputs/search-r1-mt/checkpoint-500 \\
          outputs/search-r1-mt \\
      --num_samples 200 --output_file eval_mt.jsonl

Requires the retrieval server running on http://127.0.0.1:8000/retrieve.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import requests
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from train_search_r1_mt import (
    SYSTEM_PROMPT,
    MultiTurnSearchRollout,
    search,
)


# ============================================================================
# Metrics  (plain EM, no length penalty — standard eval convention)
# ============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    return int(any(normalize_answer(g) in norm_pred for g in golden_answers))


def f1_check(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_toks = normalize_answer(prediction).split()
    if not pred_toks:
        return 0.0
    best = 0.0
    for g in golden_answers:
        gold_toks = normalize_answer(g).split()
        if not gold_toks:
            continue
        num_same = sum((Counter(pred_toks) & Counter(gold_toks)).values())
        if num_same == 0:
            continue
        p = num_same / len(pred_toks)
        r = num_same / len(gold_toks)
        best = max(best, 2 * p * r / (p + r))
    return best


# ============================================================================
# Evaluate one model
# ============================================================================

def evaluate_model(model_path: str, dataset, args) -> tuple[dict, list[dict]]:
    print(f"\n{'=' * 70}\nLoading: {model_path}\n{'=' * 70}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    manager = MultiTurnSearchRollout(
        search_func=search,
        tokenizer=tokenizer,
        max_search_calls=args.max_search_calls,
        max_iterations=args.max_search_calls * 3,
        max_completion_length=args.max_completion_length,
        chunk_max_new_tokens=args.chunk_max_new_tokens,
    )

    if args.temperature > 0:
        gen_cfg = transformers.GenerationConfig(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        gen_cfg = transformers.GenerationConfig(
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    em_total = 0
    f1_total = 0.0
    has_ans_total = 0
    search_counts: list[int] = []
    model_tok_lens: list[int] = []   # mask=1 tokens only
    total_tok_lens: list[int] = []   # all completion tokens
    records: list[dict] = []

    for ex in tqdm(dataset, desc=Path(model_path).name):
        question = ex["question"]
        golden = ex["golden_answers"]

        # Build initial messages (same as train_search_r1_mt.format_example)
        prompt_text = SYSTEM_PROMPT + f" Question: {question}\n"
        messages = [{"role": "user", "content": prompt_text}]

        with torch.no_grad():
            completion_ids, env_mask = manager.generate(
                unwrapped_model=model,
                generation_config=gen_cfg,
                messages=messages,
                device=device,
            )

        # Decode flat completion text for metric extraction
        flat_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        n_search = len(re.findall(r"<search>.*?</search>", flat_text, re.DOTALL))
        prediction = extract_answer(flat_text)
        em = em_check(prediction, golden) if prediction else 0
        f1 = f1_check(prediction, golden) if prediction else 0.0
        n_model_toks = sum(env_mask)      # tokens the model generated
        n_total_toks = len(completion_ids)

        em_total += em
        f1_total += f1
        has_ans_total += int(prediction is not None)
        search_counts.append(n_search)
        model_tok_lens.append(n_model_toks)
        total_tok_lens.append(n_total_toks)

        records.append({
            "question": question,
            "golden": golden,
            "prediction": prediction,
            "em": em,
            "f1": round(f1, 4),
            "search_count": n_search,
            "model_tokens": n_model_toks,
            "total_tokens": n_total_toks,
            "completion": flat_text,
        })

    n = len(records)
    summary = {
        "model": model_path,
        "n_samples": n,
        "em": em_total / n,
        "f1": f1_total / n,
        "has_answer_rate": has_ans_total / n,
        "mean_search_calls": sum(search_counts) / n,
        "max_search_calls": max(search_counts, default=0),
        "mean_model_tokens": sum(model_tok_lens) / n,
        "mean_total_tokens": sum(total_tok_lens) / n,
    }

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    del model, tokenizer, manager
    torch.cuda.empty_cache()

    return summary, records


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch eval for MT-trained search-r1 models"
    )
    p.add_argument("--model_name_or_path", nargs="+", required=True)
    p.add_argument("--dataset_repo", default="RUC-NLPIR/FlashRAG_datasets")
    p.add_argument("--dataset_name", default="nq")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")

    # generation
    p.add_argument("--max_completion_length", type=int, default=4096)
    p.add_argument("--chunk_max_new_tokens", type=int, default=512)
    p.add_argument("--max_search_calls", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (recommended for eval)")
    p.add_argument("--top_p", type=float, default=1.0)

    # output
    p.add_argument("--output_file", default=None,
                   help="Per-sample JSONL (model name appended as suffix).")
    p.add_argument("--summary_file", default=None,
                   help="Aggregate JSON for all models.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset_repo}/{args.dataset_name} [{args.split}]")
    ds = load_dataset(args.dataset_repo, args.dataset_name)[args.split]
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.num_samples and args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))
    print(f"Evaluating on {len(ds)} samples.\n")

    all_summaries = []
    for model_path in args.model_name_or_path:
        summary, records = evaluate_model(model_path, ds, args)
        all_summaries.append(summary)

        if args.output_file:
            out = Path(args.output_file)
            tag = Path(model_path).name.replace("/", "_")
            target = out.with_name(f"{out.stem}.{tag}.jsonl")
            with target.open("w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote per-sample records → {target}")

    # ── comparison table ──────────────────────────────────────────────────
    keys = ["em", "f1", "has_answer_rate", "mean_search_calls",
            "mean_model_tokens", "mean_total_tokens"]
    print("\n\n" + "=" * 110)
    print("FINAL COMPARISON")
    print("=" * 110)
    col_w = 16
    print(f"{'model':<55}" + "".join(f"{k:>{col_w}}" for k in keys))
    print("-" * 110)
    for s in all_summaries:
        name = s["model"]
        # Show last two path components for readability
        parts = Path(name).parts
        name_short = str(Path(*parts[-2:])) if len(parts) >= 2 else name
        print(f"{name_short:<55}" + "".join(f"{s[k]:>{col_w}.4f}" for k in keys))
    print("=" * 110)

    if args.summary_file:
        with open(args.summary_file, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nWrote summary → {args.summary_file}")


if __name__ == "__main__":
    main()
