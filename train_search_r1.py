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
You must conduct reasoning inside <thought> and </thought> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
IMPORTANT: If the returned <information> does NOT clearly contain the answer, you MUST issue another <search> with a refined or alternative query rather than guessing. Try different keywords, entities, or aliases. \
You can search up to 5 times. Only output <answer> when you are confident the information you have is sufficient. \
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


def em_check(prediction: str, golden_answers: str | list) -> float:
    """Strict EM with length penalty.

    Rationale: the previous substring-EM let the policy game the reward by
    stuffing the golden phrase plus paragraphs of filler inside <answer>.
    Now we still accept substring match (robust to articles/punctuation),
    but multiply by a length-ratio penalty:
        ratio = len(golden_tokens) / max(len(pred_tokens), len(golden_tokens))
    A concise correct answer gets ~1.0; a 50× verbose one gets ~0.02.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    pred_len = max(len(normalized_prediction.split()), 1)

    best = 0.0
    for golden_answer in golden_answers:
        norm_gold = normalize_answer(golden_answer)
        if norm_gold and norm_gold in normalized_prediction:
            gold_len = max(len(norm_gold.split()), 1)
            ratio = gold_len / max(pred_len, gold_len)
            best = max(best, ratio)
    return best


def f1_check(prediction: str, golden_answers: str | list) -> float:
    """Token-level F1 score against the best matching golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for golden in golden_answers:
        gold_tokens = normalize_answer(golden).split()
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        # multiset intersection count
        from collections import Counter
        pc, gc = Counter(pred_tokens), Counter(gold_tokens)
        num_same = sum((pc & gc).values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


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
            scores.append(0.0)
        else:
            scores.append(float(em_check(prediction, golden)))

    # Debug print (1/64 chance)
    if random.randint(1, 64) == 1:
        idx = random.randint(0, len(completions) - 1)
        print(f"\n{'='*50}")
        print(f"Golden: {solution[idx]}")
        print(f"Predicted: {extract_answer(completions[idx][-1]['content'])}")
        print(f"{'='*50}\n")

    return scores


def compute_score_f1(completions, solution, **kwargs):
    """Token-F1 partial-credit reward to densify the signal."""
    scores = []
    for completion, golden in zip(completions, solution):
        content = completion[-1]["content"]
        prediction = extract_answer(content)
        if prediction is None:
            scores.append(0.0)
        else:
            scores.append(f1_check(prediction, golden))
    return scores


_FORMAT_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_FORMAT_SEARCH_RE = re.compile(r"<search>.*?</search>", re.DOTALL)
_FORMAT_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def compute_format_reward(completions, **kwargs):
    """
    Small shaping reward for following the protocol:
      +0.05 has at least one <thought>...</thought>
      +0.05 has at least one <search>...</search>  (encourage tool use)
      +0.10 has exactly one <answer>...</answer>
    Total in [0, 0.20].
    """
    scores = []
    for completion in completions:
        content = completion[-1]["content"]
        s = 0.0
        if _FORMAT_THOUGHT_RE.search(content):
            s += 0.05
        if _FORMAT_SEARCH_RE.search(content):
            s += 0.05
        ans_matches = _FORMAT_ANSWER_RE.findall(content)
        if len(ans_matches) == 1:
            s += 0.10
        scores.append(s)
    return scores


def _golden_in_information(completion_text: str, golden) -> bool:
    """Check if any golden answer string appears inside any <information> block."""
    if isinstance(golden, str):
        golden = [golden]
    info_blocks = re.findall(r"<information>(.*?)</information>", completion_text, re.DOTALL)
    if not info_blocks:
        return False
    info_text_norm = normalize_answer(" ".join(info_blocks))
    for g in golden:
        if normalize_answer(g) and normalize_answer(g) in info_text_norm:
            return True
    return False


def compute_search_persistence(completions, solution, **kwargs):
    """Encourage the policy to keep searching when retrieved info is insufficient.

    Per-sample reward (capped at 0.15):
      base = 0.0
      + 0.05 * min(num_searches - 1, 3)         # bonus for additional searches
      + 0.10 if final answer is correct AND num_searches >= 2  # only credit
                                                                # multi-turn that
                                                                # pays off, to
                                                                # avoid spam-search
      - 0.10 if golden answer is NOT in any retrieved <information>
              AND the policy still emitted <answer>           # discourage
                                                                # "answer with bad info"

    Returns scores in roughly [-0.10, 0.25].
    """
    scores = []
    for completion, golden in zip(completions, solution):
        content = completion[-1]["content"]
        num_search = len(re.findall(r"<search>.*?</search>", content, re.DOTALL))
        prediction = extract_answer(content)
        em = em_check(prediction, golden) if prediction else 0.0

        s = 0.0
        if num_search >= 2:
            s += 0.05 * min(num_search - 1, 3)
            if em > 0:
                s += 0.10

        # Penalize giving an answer when retrieval clearly missed the target.
        if prediction is not None and not _golden_in_information(content, golden) and num_search < 5:
            # only when info was insufficient AND model didn't keep trying
            s -= 0.10

        scores.append(s)
    return scores


# ============================================================================
# Dataset Formatting (same as test.py's prompt format)
# ============================================================================

def format_example(example: dict) -> dict:
    """Format dataset example into prompt format matching test.py.

    Uses user role with system prompt + question (same as test.py).
    """
    query = example["question"]

    prompt = SYSTEM_PROMPT + f" Question: {query}\n"

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

    # Match test.py's chat-template path.
    training_args.chat_template_kwargs = {"enable_thinking": False}

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
        reward_funcs=[compute_score_em, compute_score_f1, compute_format_reward, compute_search_persistence],
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
