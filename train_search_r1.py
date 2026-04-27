"""
GRPO Training Script following test.py's approach.

Uses search_rollout.py (v1) which:
- Uses stopping_criteria to stop at </search> (same as test.py)
- Injects search results via string concatenation (same as test.py)
- Calls model.generate directly (supports vLLM)

Compared to train_grpo_v2.py (which uses search_rollout_v2 with chat template
diff method and _generate_single_turn), this script follows the simpler
test.py pattern for search result injection.

Usage:
    python train_search_r1.py \
        --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
        --output_dir outputs/search-r1 \
        --max_completion_length 8192 \
        --num_generations 4 \
        --max_steps 500

With vLLM:
    python train_search_r1.py \
        --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
        --output_dir outputs/search-r1 \
        --use_vllm True \
        --vllm_mode colocate \
        --max_completion_length 8192 \
        --num_generations 4
"""

import os
import random
import re
import string

import requests
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser
from search_rollout import create_search_rollout_func


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ============================================================================
# System Prompt (same as test.py)
# ============================================================================

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <arg_key> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""


# ============================================================================
# Search Function (same as test.py)
# ============================================================================

def search(query: str) -> str:
    """
    Search for relevant documents using dense retrieval service.
    Same as test.py's search function.
    """
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True,
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/retrieve",
            json=payload,
            timeout=10,
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
# Reward Functions
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (same as test.py/grpo_agent.py)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_answer(solution_str: str) -> str | None:
    """Extract answer from <answer>...</answer> tag."""
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def em_check(prediction: str, golden_answers: str | list) -> int:
    """Check exact match between prediction and golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1
    return 0


def compute_score_em(completions, solution, **kwargs):
    """
    Exact match scoring function.

    Args:
        completions: List of model completions (each is a list of message dicts)
        solution: List of golden answers

    Returns:
        List of scores (1 for correct, 0 for incorrect)
    """
    scores = []

    for completion, golden in zip(completions, solution):
        content = completion[-1]["content"]
        prediction = extract_answer(content)

        if prediction is None:
            scores.append(0)
        else:
            scores.append(em_check(prediction, golden))

    # Debug print (1/64 chance)
    if random.randint(1, 64) == 1:
        idx = random.randint(0, len(completions) - 1)
        print(f"\n{'='*50}")
        print(f"Golden: {solution[idx]}")
        print(f"Predicted: {extract_answer(completions[idx][-1]['content'])}")
        print(f"{'='*50}\n")

    return scores


# ============================================================================
# Dataset Formatting (same as test.py's prompt format)
# ============================================================================

def format_example(example: dict) -> dict:
    """Format dataset example into prompt format matching test.py.

    Uses user role with system prompt + question (same as test.py).
    """
    query = example["question"]

    prompt = SYSTEM_PROMPT + f"\n\nQuestion: {query}"

    messages = [{"role": "user", "content": prompt}]

    return {
        "prompt": messages,
        "solution": example["golden_answers"][0],
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # --------------------------------------------------------
    # Load and format dataset
    # --------------------------------------------------------
    print("Loading dataset...")
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    dataset = dataset.map(format_example, remove_columns=["question", "golden_answers"])

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # --------------------------------------------------------
    # Create search rollout function (v1, same as test.py)
    # --------------------------------------------------------
    print("Creating search rollout function (v1, test.py style)...")
    search_rollout = create_search_rollout_func(
        search_func=search,
        max_search_calls=10,
        max_iterations=20,
    )

    # --------------------------------------------------------
    # Initialize trainer
    # --------------------------------------------------------
    print("Initializing GRPOTrainer with search_rollout (v1)...")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        rollout_func=search_rollout,
        reward_funcs=[compute_score_em],
        args=training_args,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    print("Starting training...")
    trainer.train()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    print(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Training complete!")
