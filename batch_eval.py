"""
Batch evaluation script.

Reuses the EXACT same generation loop as training (`SearchRolloutManager`
from `search_rollout.py`) and the EXACT same prompt format as
`train_search_r1.py`.  This guarantees evaluation matches what the model
saw during GRPO training.

Metrics:
    - EM (exact match, substring after normalization, same as training reward)
    - F1 (token-level, SQuAD-style)
    - search_count distribution
    - has_answer rate
    - mean completion length (tokens)

Usage:
    # Evaluate a single checkpoint on 200 NQ test samples
    python batch_eval.py \
        --model_name_or_path outputs/search-r1/checkpoint-500 \
        --num_samples 200

    # Compare multiple checkpoints (space-separated)
    python batch_eval.py \
        --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
                              outputs/search-r1 \
                              outputs/search-r1/checkpoint-500 \
        --num_samples 200 \
        --output_file eval_results.jsonl

    # Different dataset / split
    python batch_eval.py \
        --model_name_or_path outputs/search-r1 \
        --dataset_name hotpotqa --split dev --num_samples 500

Requires the retrieval server running on http://127.0.0.1:8000/retrieve
(see `retrieval_launch.sh`).
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

from search_rollout import SearchRolloutManager


# ============================================================================
# Prompt — MUST match train_search_r1.py exactly
# ============================================================================

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <thought> and </thought> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""


def format_prompt(question: str, tokenizer) -> list[int]:
    """Same as train_search_r1.format_example + trainer's chat templating."""
    user_text = SYSTEM_PROMPT + f" Question: {question}\n"
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,  # matches training_args.chat_template_kwargs
    )
    enc = tokenizer(text, add_special_tokens=False)
    return enc["input_ids"]
# ============================================================================
# Search backend — same as train_search_r1.py
# ============================================================================

def search(query: str) -> str:
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve", json=payload, timeout=10
        )
        results = response.json()["result"]
    except Exception as e:
        return f"Search error: {str(e)}"

    formatted = []
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        lines = content.split("\n")
        title = lines[0] if lines else "Unknown"
        text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        formatted.append(f"Doc {idx+1}(Title: {title}) {text}")
    return "\n".join(formatted)


# ============================================================================
# Metrics — same normalization / EM as training, plus F1
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


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    for g in golden_answers:
        if normalize_answer(g) in norm_pred:
            return 1
    return 0


def f1_check(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for g in golden_answers:
        gold_tokens = normalize_answer(g).split()
        if not gold_tokens:
            continue
        num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if num_same == 0:
            continue
        p = num_same / len(pred_tokens)
        r = num_same / len(gold_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


# ============================================================================
# Lightweight trainer-like shim for SearchRolloutManager.generate()
# ============================================================================

class _DummyTrainer:
    """SearchRolloutManager only reads tokenizer + a few attrs; no trainer
    is needed at eval time, so we just instantiate the manager directly."""


# ============================================================================
# Evaluate one model
# ============================================================================

def evaluate_model(model_path: str, dataset, args) -> dict:
    print(f"\n{'=' * 70}\nLoading: {model_path}\n{'=' * 70}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    manager = SearchRolloutManager(
        search_func=search,
        tokenizer=tokenizer,
        max_search_calls=args.max_search_calls,
        max_iterations=args.max_iterations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_max_new_tokens=args.chunk_max_new_tokens,
    )

    # Build a clean GenerationConfig: when greedy, omit sampling-only kwargs
    # (temperature/top_p/top_k) so transformers doesn't warn about them.
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

    em_total, f1_total, has_ans_total = 0, 0.0, 0
    search_counts: list[int] = []
    comp_lens: list[int] = []
    records = []

    for ex in tqdm(dataset, desc=Path(model_path).name):
        question = ex["question"]
        golden = ex["golden_answers"]

        prompt_ids = format_prompt(question, tokenizer)

        # Re-implement minimal generation with manager — manager.generate
        # itself drives the loop.  We give it a simple unwrapped_model.
        completion_ids, env_mask = manager.generate(
            unwrapped_model=model,
            generation_config=gen_cfg,
            prompt_ids=prompt_ids,
            device=device,
        )
        # search_count is internal; recompute from the decoded text.
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        search_count = len(re.findall(r"<search>.*?</search>", text, re.DOTALL))

        prediction = extract_answer(text)
        em = em_check(prediction, golden) if prediction else 0
        f1 = f1_check(prediction, golden) if prediction else 0.0
        has_ans = prediction is not None

        em_total += em
        f1_total += f1
        has_ans_total += int(has_ans)
        search_counts.append(search_count)
        comp_lens.append(len(completion_ids))

        records.append({
            "question": question,
            "golden": golden,
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "search_count": search_count,
            "completion_length": len(completion_ids),
            "completion": text,
        })

    n = len(dataset)
    summary = {
        "model": model_path,
        "n_samples": n,
        "em": em_total / n,
        "f1": f1_total / n,
        "has_answer_rate": has_ans_total / n,
        "mean_search_calls": sum(search_counts) / n,
        "max_search_calls": max(search_counts) if search_counts else 0,
        "mean_completion_length": sum(comp_lens) / n,
    }

    print("\n--- Summary ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Free GPU memory before next checkpoint.
    del model, tokenizer, manager
    torch.cuda.empty_cache()

    return summary, records


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", nargs="+", required=True,
                   help="One or more model paths to evaluate (compared sequentially).")
    p.add_argument("--dataset_repo", default="RUC-NLPIR/FlashRAG_datasets")
    p.add_argument("--dataset_name", default="nq")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle before sub-sampling (default: take first N).")

    # generation
    p.add_argument("--max_completion_length", type=int, default=4096)
    p.add_argument("--chunk_max_new_tokens", type=int, default=512)
    p.add_argument("--max_search_calls", type=int, default=10)
    p.add_argument("--max_iterations", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 => greedy decoding (recommended for eval).")
    p.add_argument("--top_p", type=float, default=1.0)

    # output
    p.add_argument("--output_file", default=None,
                   help="If set, write per-sample JSONL records (one file per model, with model name suffix).")
    p.add_argument("--summary_file", default=None,
                   help="If set, write summary JSON for all models.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset_repo} / {args.dataset_name} [{args.split}]")
    ds = load_dataset(args.dataset_repo, args.dataset_name)[args.split]
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.num_samples and args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))
    print(f"Evaluating on {len(ds)} samples.")

    all_summaries = []
    for model_path in args.model_name_or_path:
        summary, records = evaluate_model(model_path, ds, args)
        all_summaries.append(summary)

        if args.output_file:
            out_path = Path(args.output_file)
            stem = out_path.stem
            tag = Path(model_path).name.replace("/", "_")
            target = out_path.with_name(f"{stem}.{tag}.jsonl")
            with target.open("w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote per-sample records → {target}")

    print("\n\n================ FINAL COMPARISON ================")
    keys = ["em", "f1", "has_answer_rate", "mean_search_calls", "mean_completion_length"]
    print(f"{'model':<60} " + " ".join(f"{k:>14}" for k in keys))
    for s in all_summaries:
        name = s["model"][-58:]
        print(f"{name:<60} " + " ".join(f"{s[k]:>14.4f}" for k in keys))

    if args.summary_file:
        with open(args.summary_file, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nWrote summary → {args.summary_file}")


if __name__ == "__main__":
    main()
