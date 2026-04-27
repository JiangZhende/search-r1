"""
Evaluation script for GRPO-trained search models.

Self-contained: no dependency on search_rollout_v2.py or other project scripts.
Uses the same rollout logic (token ID concatenation + chat template diff for
tool suffix) as training, and the same prompt format as train_grpo_v2.py.

Usage:
    # Evaluate a single model
    python eval_search.py \
        --model_name_or_path outputs/search-r1-v2/checkpoint-500

    # Compare original vs trained model
    python eval_search.py \
        /home/l33500/models/Qwen/Qwen3-0.6B \
        outputs/search-r1-v2/checkpoint-500

    # With more samples and custom parameters
    python eval_search.py \
        --model_name_or_path outputs/search-r1-v2/checkpoint-500 \
        --num_samples 200 \
        --max_new_tokens 512 \
        --temperature 0.0
"""

import argparse
import random
import re
import string
import requests

import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# System Prompt (same as train_grpo_v2.py)
# ============================================================================

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <thought> and </thought> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n"""

# ============================================================================
# Search Function
# ============================================================================

def search(query: str) -> str:
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True,
    }
    try:
        response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=10)
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
# Scoring Functions
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


def extract_answer(solution_str: str) -> str | None:
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def em_check(prediction: str, golden_answers: str | list) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1
    return 0


# ============================================================================
# Search Rollout Logic (same as search_rollout_v2.py, self-contained)
# ============================================================================

def _apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
    """Apply chat template to messages and return token IDs."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            think=False,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            think=False
        )
        return tokenizer.encode(text, add_special_tokens=False)


def compute_tool_suffix_ids(tokenizer, assistant_text: str, info_content: str) -> list[int]:
    """
    Compute tool suffix token IDs using the same diff approach as trl's
    _get_tool_suffix_ids and search_rollout_v2's _compute_tool_suffix.
    """
    user_msg = {"role": "user", "content": "dummy"}
    assistant_msg = {"role": "assistant", "content": assistant_text}
    tool_msg = {"role": "user", "content": info_content}

    # Prefix: user + assistant (no generation prompt)
    prefix_ids = _apply_chat_template(
        tokenizer, [user_msg, assistant_msg], add_generation_prompt=True
    )["input_ids"]
    print(tokenizer.decode(prefix_ids, skip_special_tokens=False))
    print(prefix_ids)

    # Full: user + assistant + tool (with generation prompt)
    full_ids = _apply_chat_template(
        tokenizer, [user_msg, assistant_msg, tool_msg], add_generation_prompt=True
    )["input_ids"]
    print(tokenizer.decode(full_ids, skip_special_tokens=False))
    print(full_ids)

    # Align on EOS boundary (like trl's _get_tool_suffix_ids)
    eos_token_id = tokenizer.eos_token_id
    print(eos_token_id)
    if eos_token_id in prefix_ids:
        last_eos_idx = max(i for i, tok_id in enumerate(prefix_ids) if tok_id == eos_token_id)
        prefix_ids_trimmed = prefix_ids[: last_eos_idx + 1]
    else:
        prefix_ids_trimmed = prefix_ids

    # The suffix is the difference
    if full_ids[:len(prefix_ids_trimmed)] == prefix_ids_trimmed:
        return full_ids[len(prefix_ids_trimmed):]
    else:
        print("here")
        # Fallback: just encode the tool content directly
        return tokenizer.encode(info_content, add_special_tokens=False)


def generate_with_search(
    model,
    tokenizer,
    messages: list[dict],
    search_func,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_search_calls: int = 5,
    max_iterations: int = 10,
) -> tuple[list[int], list[int], int]:
    """
    Generate completion with search tool calls.

    Same logic as search_rollout_v2's SearchRolloutManagerV2.generate_with_search:
    - Each turn, apply_chat_template(current_messages) builds input_ids
    - On <search>, append assistant + tool messages to current_messages
    - tool_suffix_ids computed via diff method for completion_ids tracking

    Returns:
        (completion_ids, tool_mask, search_count)
    """
    search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
    device = model.device

    all_completion_ids = []
    all_tool_mask = []
    search_count = 0

    # Use messages to track conversation (same as search_rollout_v2)
    current_messages = list(messages)

    for _ in range(max_iterations):
        # Build input_ids via chat template (messages-based)
        input_ids = _apply_chat_template(
            tokenizer, current_messages, add_generation_prompt=True
        )["input_ids"]
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        gen_kwargs = transformers.GenerationConfig(**gen_kwargs)

        with torch.no_grad():
            output = model.generate(input_tensor, gen_kwargs)

        new_ids = output[0][input_tensor.shape[1]:].tolist()

        if not new_ids:
            break

        generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)

        # Model-generated tokens → mask=1
        all_completion_ids.extend(new_ids)
        all_tool_mask.extend([1] * len(new_ids))

        # Check for <search> tag
        search_match = search_pattern.search(generated_text)
        if search_match and search_count < max_search_calls:
            search_count += 1

            query = search_match.group(1).strip()
            try:
                search_result = search_func(query)
            except Exception as e:
                search_result = f"Error during search: {str(e)}"

            info_content = f"<information>\n{search_result}\n</information>"

            # Append assistant + tool messages (same as search_rollout_v2)
            current_messages.append({"role": "assistant", "content": generated_text})
            current_messages.append({"role": "user", "content": info_content})

            # Compute tool suffix IDs for completion_ids tracking
            tool_suffix_ids = compute_tool_suffix_ids(tokenizer, generated_text, info_content)
            # print(tokenizer.decode(tool_suffix_ids, skip_special_tokens=False))
            all_completion_ids.extend(tool_suffix_ids)
            all_tool_mask.extend([0] * len(tool_suffix_ids))
            continue

        # No search — append assistant message and finalize
        current_messages.append({"role": "assistant", "content": generated_text})
        break

    return all_completion_ids, all_tool_mask, search_count


# ============================================================================
# Evaluate one model
# ============================================================================

def evaluate_model(model_path: str, dataset, args) -> dict:
    print(f"\n{'='*70}")
    print(f"Loading model: {model_path}")
    print(f"{'='*70}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    scores = []
    details = []

    for example in tqdm(dataset, desc=f"Evaluating {model_path.split('/')[-1]}"):
        question = example["question"]
        golden = example["golden_answers"]

        # Format prompt — same as train_grpo_v2.py: user role, system prompt + question
        prompt_text = SYSTEM_PROMPT + f"\n\nQuestion: {question}"
        messages = [{"role": "user", "content": prompt_text}]

        # Run search rollout (same as search_rollout_v2)
        completion_ids, _, search_count = generate_with_search(
            model, tokenizer, messages,
            search_func=search,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_search_calls=args.max_search_calls,
            max_iterations=args.max_iterations,
        )

        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        prediction = extract_answer(completion_text)
        has_answer = prediction is not None

        if prediction is None:
            score = 0
        else:
            score = em_check(prediction, golden)

        scores.append(score)
        details.append({
            "question": question,
            "golden": golden,
            "prediction": prediction,
            "score": score,
            "search_count": search_count,
            "has_answer": has_answer,
            "completion_text": completion_text,
        })

    em_score = sum(scores) / len(scores) if scores else 0.0
    answer_rate = sum(1 for d in details if d["has_answer"]) / len(details) if details else 0.0
    avg_searches = sum(d["search_count"] for d in details) / len(details) if details else 0.0

    return {
        "model_path": model_path,
        "em_score": em_score,
        "answer_rate": answer_rate,
        "avg_searches": avg_searches,
        "num_samples": len(scores),
        "details": details,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO-trained search models")
    parser.add_argument("model_paths", nargs="+", help="One or more model paths to evaluate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy, >0 = sampling")
    parser.add_argument("--max_search_calls", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_examples", type=int, default=3, help="Number of examples to print in detail")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    test_dataset = dataset["test"]

    if args.num_samples < len(test_dataset):
        indices = random.sample(range(len(test_dataset)), args.num_samples)
        test_dataset = test_dataset.select(indices)

    print(f"Evaluating on {len(test_dataset)} samples")

    # Evaluate each model
    all_results = []
    for model_path in args.model_paths:
        result = evaluate_model(model_path, test_dataset, args)
        all_results.append(result)

    # Print summary comparison
    print(f"\n\n{'='*80}")
    print(f"{'EVALUATION SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<50} {'EM':>8} {'Ans%':>8} {'AvgSrch':>8}")
    print(f"{'-'*80}")

    for result in all_results:
        model_name = result["model_path"].split("/")[-1] or result["model_path"]
        print(f"{model_name:<50} {result['em_score']:>8.3f} {result['answer_rate']:>8.3f} {result['avg_searches']:>8.2f}")

    print(f"{'='*80}")

    # Print detailed examples from the last model
    if all_results and args.show_examples > 0:
        result = all_results[-1]
        model_name = result["model_path"].split("/")[-1] or result["model_path"]
        print(f"\n\nDetailed examples from: {model_name}")
        print(f"{'='*80}")

        correct = [d for d in result["details"] if d["score"] == 1]
        incorrect = [d for d in result["details"] if d["score"] == 0]

        shown = 0

        if correct:
            print(f"\n--- Correct Example ---")
            d = random.choice(correct)
            print(f"Q: {d['question']}")
            print(f"Golden: {d['golden']}")
            print(f"Predicted: {d['prediction']}")
            print(f"Searches: {d['search_count']}")
            shown += 1

        for d in incorrect[:args.show_examples - shown]:
            print(f"\n--- Incorrect Example ---")
            print(f"Q: {d['question']}")
            print(f"Golden: {d['golden']}")
            print(f"Predicted: {d['prediction']}")
            print(f"Has answer: {d['has_answer']}")
            print(f"Searches: {d['search_count']}")
            if not d['has_answer']:
                text = d['completion_text']
                print(f"Completion (tail): ...{text[-500:]}")


if __name__ == "__main__":
    main()
