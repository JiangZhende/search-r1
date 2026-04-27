"""
GRPO Training Script using search_rollout_v2 (based on _generate_single_turn).

Compared to grpo_agent.py (which uses search_rollout.py with manual model.generate),
this script leverages trainer._generate_single_turn for generation, which:
- Supports vLLM acceleration
- Properly handles chat template formatting for tool results
- Follows the same pattern as trl's _tool_call_loop

Usage:
    python train_grpo_v2.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --output_dir outputs/search-r1-v2 \
        --max_completion_length 4096 \
        --num_generations 4 \
        --max_steps 500

With vLLM:
    python train_grpo_v2.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --output_dir outputs/search-r1-v2 \
        --use_vllm True \
        --vllm_mode colocate \
        --max_completion_length 4096 \
        --num_generations 4
"""

import os
import random
import re
import string

import requests
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser
from search_rollout_v2 import create_search_rollout_func


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""


# ============================================================================
# Search Function
# ============================================================================

def search(query: str) -> str:
    """
    Search for relevant documents using dense retrieval service.

    Args:
        query: The search query

    Returns:
        Formatted search results string
    """
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
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
    """Normalize answer for comparison."""
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


def format_reward(completions, **kwargs):
    """
    Reward for proper format structure.

    Rewards:
    - +0.1 for having exactly one <answer>...</answer> tag
    - +0.05 per <search>...</search> tag (up to 3)
    - -0.1 for multiple <answer> tags
    - -0.1 for unpaired <search> tags
    """
    rewards = []

    for completion in completions:
        content = completion[-1]["content"]
        reward = 0.0

        # Check for <answer> tag
        answer_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if len(answer_matches) == 1:
            reward += 0.1
        elif len(answer_matches) > 1:
            reward -= 0.1

        # Check for <search> tag
        search_opens = content.count('<search>')
        search_closes = content.count('</search>')
        if search_opens == search_closes and search_opens > 0:
            reward += 0.05 * min(search_opens, 3)
        elif search_opens != search_closes:
            reward -= 0.1

        rewards.append(reward)

    return rewards


# ============================================================================
# Dataset Formatting
# ============================================================================

def format_example(example: dict) -> dict:
    """Format dataset example into conversational prompt format.

    Returns messages format for compatibility with chat template:
        [{"role": "user", "content": "..."}]
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

    # Enable thinking mode for chat template
    # training_args.chat_template_kwargs = {"enable_thinking": False}

    # --------------------------------------------------------
    # Create search rollout function (v2, uses _generate_single_turn)
    # --------------------------------------------------------
    print("Creating search rollout function (v2)...")
    search_rollout = create_search_rollout_func(
        search_func=search,
        max_search_calls=10,
        max_iterations=20,
    )

    # --------------------------------------------------------
    # Initialize trainer
    # --------------------------------------------------------
    print("Initializing GRPOTrainer with search_rollout_v2...")
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