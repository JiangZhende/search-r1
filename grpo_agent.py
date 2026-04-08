# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python examples/scripts/grpo_agent.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir grpo_biogrid_qwen_3g-1.7b \
    --push_to_hub True \
    --use_vllm True \
    --vllm_mode colocate \
    --max_completion_length 1024 \
    --report_to trackio \
    --log_completions True \
    --max_steps 400
```
"""

import os
import random
import re
import signal
import sqlite3
import string
import textwrap
import requests
from contextlib import contextmanager

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def structure_reward(completions, **kwargs):
    """
    Reward proper assistant structure.
    Encourages a logical sequence: tool call + response + optional extra content.
    """
    rewards = []

    for completion in completions:
        has_call = False
        has_response = False
        has_other = False

        for turn in completion:
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            else:
                content = turn.get("content")
                if content and content.strip() not in ["", "<think>"]:
                    has_other = True

        # Reward sequences
        if has_call and has_response:
            if has_other:
                reward = 0.1
            else:
                reward = 0.05  # still positive even without extra text
        elif has_call and not has_response:
            reward = -0.15
        else:
            reward = 0.0  # neutral if no call

        rewards.append(reward)

    return rewards


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        print(text)
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def compute_score_em(completions, solution, method='strict', format_score=0., score=1., **kwargs):
    """The scoring function for exact match (EM).

    Args:
        completions: the model completions
        answer: the extracted answer
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    # answer = extract_solution(solution_str=solution_str)
    print("1:", completions)
    print("2:", solution)
    scores = []
    
    solution_str = completions[-1][0]["content"]
    # answer = extract_solution(solution_str=solution_str)
    
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {solution}")
        # print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    contents = [completion[-1]["content"] for completion in completions]
    for pre, s in zip(contents, solution):
        # pre = completion[-1]["content"]
        pre = extract_solution(solution_str=pre)
        if pre is None:
            scores.append(0)
            
        else:
            if em_check(pre, solution):
                scores.append(1)
            else:
                scores.append(0)
    return scores
# ------------------------
# Database tool function
# ------------------------
class TimeoutError(Exception):
    """Raised when a function call times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager that raises TimeoutError if execution exceeds time limit."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def search(query: str):
    """
    Search for relevant documents based on the query using dense retrieval.

    Args:
        query: The search query.

    Returns:
        A string containing the formatted search results.
    """
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# ------------------------
# Dataset formatting
# ------------------------
def format_example(example):
    query = example["question"]
    messages = [
    {"role": "system", "content": "Answer the given question. Each time you obtain new information, you must think and reason. \
     After thinking, if you find that you lack certain knowledge, you can acquire it through tools and obtain relevant information. \
     You can search as many times as your want.If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\
     The following is the user's question."},
    {"role": "user", "content": query}
    ]
    solution = example["golden_answers"][0]
    # content = f"{preamble}\nQuestion: {question}"
    # prompt = [{"role": "user", "content": content}]
    return {"prompt": messages, "solution": solution}


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # ------------------------
    # Create DB
    # ------------------------
    # print("Creating biogrid.db...")
    # Load dataset
    # biogrid_dataset = load_dataset("qgallouedec/biogrid", split="train")
    # df = biogrid_dataset.to_pandas()

    # Normalize column names: remove spaces, replace with underscores
    # df.columns = [c.replace(" ", "_") for c in df.columns]
    # conn = sqlite3.connect("biogrid.db")
    # try:
        # df.to_sql("interactions", conn, if_exists="replace", index=False)
        # print(f"biogrid.db created. Rows stored: {len(df)}")
    # finally:
        # conn.close()

    # ------------------------
    # Load and format dataset
    # ------------------------
    dataset = load_dataset(script_args.dataset_name, "nq")
    # dataset = dataset.filter(
        # lambda example: example["question"].startswith("Does the gene ")
    # )  # keep only simple questions for example
    dataset = dataset.map(format_example, remove_columns=["question", "golden_answers"])

    train_dataset = dataset["train"]
    # print(train_dataset)
    # dataset = load_dataset(script_args.dataset+"/test.parquet")
    # dataset = dataset.filter(
        # lambda example: example["question"].startswith("Does the gene ")
    # )  # keep only simple questions for example
    eval_dataset = dataset["test"] # No eval by default, can be added if needed

    training_args.chat_template_kwargs = {"enable_thinking": False}

    # ------------------------
    # Initialize trainer
    # ------------------------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tools=[search],
        reward_funcs=[compute_score_em, structure_reward],
        args=training_args,
    )

    # ------------------------
    # Train
    # ------------------------
    trainer.train()

    # ------------------------
    # Save and push
    # ------------------------
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)