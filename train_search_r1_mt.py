"""train_search_r1_mt.py

Multi-turn GRPO Training Script.

Conversation pattern used for generation:

  [{"role": "user",      "content": "<system_prompt> Question: ..."},
   {"role": "assistant", "content": "<thought>...</thought><search>query</search>"},
   {"role": "user",      "content": "<information>\\nresults\\n</information>"},
   {"role": "assistant", "content": "<thought>...</thought><answer>answer</answer>"}]

<information> is encoded as a proper "user" message via apply_chat_template,
so the model sees:
  <|im_start|>user\\n<information>...\\n</information><|im_end|>
before generating the next <thought>.

Loss masking:
  completion_mask=1  for model-generated (assistant) tokens
  completion_mask=0  for injected user-turn tokens (<information> blocks)
"""

import copy
import os
import random
import re
import string

import requests
import torch
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser

try:
    from trl.models import unwrap_model_for_generation
except ImportError:
    from trl.extras.profiling import unwrap_model_for_generation  # type: ignore

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ============================================================================
# Prompts
# ============================================================================

SYSTEM_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <thought> and </thought> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "IMPORTANT: If the returned <information> does NOT clearly contain the answer, you MUST issue "
    "another <search> with a refined or alternative query rather than guessing. "
    "Try different keywords, entities, or aliases. "
    "You can search up to 5 times. Only output <answer> when you are confident the information "
    "you have is sufficient. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."
)


# ============================================================================
# Retrieval
# ============================================================================

def search(query: str) -> str:
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    try:
        resp = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=10)
        results = resp.json()["result"]
    except Exception as e:
        return f"Search error: {e}"

    rows = []
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        lines = content.split("\n")
        title = lines[0] if lines else "Unknown"
        text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        rows.append(f"Doc {idx+1}(Title: {title}) {text}")
    return "\n".join(rows)


# ============================================================================
# Multi-turn rollout
# ============================================================================

_STOP_SEQS = [
    "</search>", " </search>",
    "</search>\n", " </search>\n",
    "</search>\n\n", " </search>\n\n",
]
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class _StopOnSeq(StoppingCriteria):
    def __init__(self, seqs, tokenizer):
        self.targets = [tokenizer.encode(s, add_special_tokens=False) for s in seqs]
        self.min_len = min(len(t) for t in self.targets)

    def __call__(self, input_ids, scores, **kwargs):
        for row in input_ids:
            row_hit = any(
                row.shape[0] >= len(t) and
                torch.equal(row[-len(t):], torch.as_tensor(t, device=row.device))
                for t in self.targets
            )
            if not row_hit:
                return False
        return True


class MultiTurnSearchRollout:
    """
    Drives multi-turn search generation for one prompt.

    After each </search> stop:
      1. Decode generated tokens as the assistant turn text.
      2. Append {"role": "assistant", "content": asst_text} to current_messages.
      3. Append {"role": "user", "content": "<information>...</information>"}.
      4. Re-encode the full conversation via apply_chat_template.
      5. The delta tokens (new_full_ids beyond prev_context + new_tokens) are
         the injected user-turn tokens → completion_mask=0.
    """

    def __init__(
        self,
        search_func,
        tokenizer,
        max_search_calls: int = 5,
        max_iterations: int = 15,
        max_completion_length: int = 4096,
        chunk_max_new_tokens: int = 512,
    ):
        self.search_func = search_func
        self.tokenizer = tokenizer
        self.max_search_calls = max_search_calls
        self.max_iterations = max_iterations
        self.max_completion_length = max_completion_length
        self.chunk_max_new_tokens = chunk_max_new_tokens
        self.eos_ids = {tokenizer.eos_token_id, 151645, 151643} - {None}
        self.stopping_criteria = StoppingCriteriaList([_StopOnSeq(_STOP_SEQS, tokenizer)])

    def _tmpl(self, messages, add_gen_prompt=True):
        """Apply chat template and return list[int]."""
        result = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_gen_prompt,
            tokenize=True,
            enable_thinking=False,
        )
        # apply_chat_template may return list[int] or a tokenizers.Encoding /
        # BatchEncoding depending on the tokenizer version.  Normalise to list[int].
        if isinstance(result, (list, tuple)):
            return list(result)
        if hasattr(result, "input_ids"):
            ids = result.input_ids
            return ids.tolist() if hasattr(ids, "tolist") else list(ids)
        # dict-like (BatchEncoding)
        ids = result["input_ids"]
        return ids.tolist() if hasattr(ids, "tolist") else list(ids)

    @torch.no_grad()
    def generate(self, unwrapped_model, generation_config, messages, device):
        """
        Args:
            messages: initial list of message dicts (already includes user question)
        Returns:
            (completion_ids, env_mask)
            completion_ids: all tokens after the initial prompt
            env_mask: 1 = model-generated (assistant), 0 = environment-injected (user info turn)
        """
        current_messages = list(messages)
        running_ids = self._tmpl(current_messages, add_gen_prompt=True)

        completion_ids: list[int] = []
        env_mask: list[int] = []
        search_count = 0

        for _ in range(self.max_iterations):
            remaining = self.max_completion_length - len(completion_ids)
            if remaining <= 0:
                break

            input_tensor = torch.tensor([running_ids], device=device)

            if generation_config is not None:
                gen_cfg = copy.deepcopy(generation_config)
                gen_cfg.max_new_tokens = min(self.chunk_max_new_tokens, remaining)
                outputs = unwrapped_model.generate(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    stopping_criteria=self.stopping_criteria,
                    generation_config=gen_cfg,
                )
            else:
                outputs = unwrapped_model.generate(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    max_new_tokens=min(self.chunk_max_new_tokens, remaining),
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=1.0,
                )

            new_tokens = outputs[0][len(running_ids):].tolist()
            if not new_tokens:
                break

            # Clip to budget
            if len(completion_ids) + len(new_tokens) > self.max_completion_length:
                new_tokens = new_tokens[: self.max_completion_length - len(completion_ids)]

            # Assistant-generated → mask=1
            completion_ids.extend(new_tokens)
            env_mask.extend([1] * len(new_tokens))

            # Natural EOS → done
            if new_tokens[-1] in self.eos_ids:
                break

            chunk_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            if _ANSWER_RE.search(chunk_text):
                break

            query_m = _SEARCH_RE.search(chunk_text)
            if query_m and search_count < self.max_search_calls:
                search_count += 1
                try:
                    result = self.search_func(query_m.group(1).strip())
                except Exception as e:
                    result = f"Search error: {e}"

                # ── Build new multi-turn context ──────────────────────────
                # Decode the assistant turn content (skip special tokens so the
                # text round-trips cleanly through the chat template).
                asst_content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                prev_running_ids = running_ids  # save before update

                current_messages.append({"role": "assistant", "content": asst_content})
                current_messages.append({
                    "role": "user",
                    "content": f"<information>\n{result}\n</information>",
                })

                # New full context: prev turns + injected user turn + next gen prompt
                new_running_ids = self._tmpl(current_messages, add_gen_prompt=True)

                # Injected tokens = everything in new context beyond
                # (previous context length + new assistant tokens).
                #
                # Assumption: tokenizing the assistant content inside the template
                # produces the same token IDs as the raw generation (new_tokens).
                # This holds for Qwen3's BPE when enable_thinking=False because
                # special tokens (<|im_end|>) don't cause cross-boundary merges.
                prefix_end = len(prev_running_ids) + len(new_tokens)
                if len(new_running_ids) > prefix_end:
                    injected_ids = new_running_ids[prefix_end:]
                else:
                    # Fallback: encode the user info turn from scratch using the
                    # diff between [asst] and [asst, user_info].
                    dummy_u = {"role": "user", "content": "q"}
                    asst_msg = {"role": "assistant", "content": asst_content}
                    info_msg = {"role": "user", "content": f"<information>\n{result}\n</information>"}
                    base_ids = self._tmpl([dummy_u, asst_msg], add_gen_prompt=False)
                    full_ids = self._tmpl([dummy_u, asst_msg, info_msg], add_gen_prompt=True)
                    # Align on last EOS
                    eos = self.tokenizer.eos_token_id
                    if eos in base_ids:
                        cut = max(i for i, t in enumerate(base_ids) if t == eos) + 1
                        base_ids = base_ids[:cut]
                    injected_ids = (
                        full_ids[len(base_ids):]
                        if full_ids[: len(base_ids)] == base_ids
                        else self.tokenizer.encode(
                            f"<information>\n{result}\n</information>",
                            add_special_tokens=False,
                        )
                    )

                budget = self.max_completion_length - len(completion_ids)
                if budget <= 0:
                    break
                injected_ids = injected_ids[:budget]

                completion_ids.extend(injected_ids)
                env_mask.extend([0] * len(injected_ids))

                running_ids = new_running_ids
                continue

            # No search query — model didn't trigger tool; keep generating with
            # the context extended by the new tokens.
            running_ids = running_ids + new_tokens

        if not completion_ids:
            eos = self.tokenizer.eos_token_id or 0
            completion_ids = [eos]
            env_mask = [1]

        return completion_ids, env_mask


def create_mt_rollout_func(search_func, max_search_calls=5, max_iterations=15):
    """Factory returning a GRPOTrainer-compatible rollout_func."""

    def rollout_func(prompts: list, trainer) -> dict:
        tokenizer = trainer.processing_class
        device = trainer.accelerator.device

        manager = MultiTurnSearchRollout(
            search_func=search_func,
            tokenizer=tokenizer,
            max_search_calls=max_search_calls,
            max_iterations=max_iterations,
            max_completion_length=trainer.args.max_completion_length,
        )

        with unwrap_model_for_generation(
            trainer.model, trainer.accelerator
        ) as unwrapped:
            gen_cfg = trainer.model.generation_config

            all_prompt_ids, all_completion_ids, all_env_masks = [], [], []

            for messages in prompts:
                # Use manager._tmpl() which already handles the Encoding → list[int]
                # conversion (via ["input_ids"]).  Calling apply_chat_template
                # directly can return a tokenizers.Encoding object when
                # enable_thinking=False is forwarded to the fast tokenizer.
                prompt_ids = manager._tmpl(messages, add_gen_prompt=True)
                completion_ids, env_mask = manager.generate(
                    unwrapped, gen_cfg, messages, device
                )
                all_prompt_ids.append(prompt_ids)
                all_completion_ids.append(completion_ids)
                all_env_masks.append(env_mask)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": None,
            "env_mask": all_env_masks,
        }

    return rollout_func


# ============================================================================
# Reward helpers
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
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def em_check(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    pred_len = max(len(norm_pred.split()), 1)
    best = 0.0
    for gold in golden_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold and norm_gold in norm_pred:
            gold_len = max(len(norm_gold.split()), 1)
            best = max(best, gold_len / max(pred_len, gold_len))
    return best


def f1_check(prediction: str, golden_answers) -> float:
    from collections import Counter
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_toks = normalize_answer(prediction).split()
    if not pred_toks:
        return 0.0
    best = 0.0
    for gold in golden_answers:
        gold_toks = normalize_answer(gold).split()
        if not gold_toks:
            continue
        pc, gc = Counter(pred_toks), Counter(gold_toks)
        num_same = sum((pc & gc).values())
        if num_same == 0:
            continue
        p = num_same / len(pred_toks)
        r = num_same / len(gold_toks)
        best = max(best, 2 * p * r / (p + r))
    return best


def _golden_in_information(completion_text: str, golden) -> bool:
    if isinstance(golden, str):
        golden = [golden]
    blocks = re.findall(r"<information>(.*?)</information>", completion_text, re.DOTALL)
    if not blocks:
        return False
    info_norm = normalize_answer(" ".join(blocks))
    return any(normalize_answer(g) and normalize_answer(g) in info_norm for g in golden)


# ============================================================================
# Reward functions (same as train_search_r1.py v8, except prompts kwarg)
# ============================================================================

def _format_completion_dialog(content: str) -> str:
    """Re-parse the flat completion text into a readable turn-by-turn format.

    The completion arrives as one flat string where multi-turn boundaries are
    indicated by the bare word 'assistant' or 'user' (special tokens stripped).
    We reconstruct the dialog by splitting on <information> blocks and
    <search>/<answer> tags.
    """
    lines = []
    # Split on <information>...</information> boundaries to find turn boundaries.
    parts = re.split(r"(<information>.*?</information>)", content, flags=re.DOTALL)
    turn = 1
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("<information>"):
            lines.append(f"  [user / env turn]")
            # Show only first 300 chars of info to keep log readable.
            preview = part[:300] + ("…" if len(part) > 300 else "")
            lines.append(f"    {preview}")
        else:
            # Assistant segment — may contain multiple <thought>/<search>/<answer> blocks.
            # Strip leading 'assistant' role marker left by skip_special_tokens.
            asst_text = re.sub(r"^\s*(assistant|user)\s*", "", part).strip()
            if asst_text:
                lines.append(f"  [assistant turn {turn}]")
                lines.append(f"    {asst_text[:400]}{'…' if len(asst_text) > 400 else ''}")
                turn += 1
    return "\n".join(lines)


def compute_score_em(completions, solution, **kwargs):
    scores = []
    for completion, golden in zip(completions, solution):
        content = completion[-1]["content"]
        pred = extract_answer(content)
        scores.append(0.0 if pred is None else float(em_check(pred, golden)))
    if random.randint(1, 64) == 1:
        idx = random.randint(0, len(completions) - 1)
        content = completions[idx][-1]["content"]
        golden = solution[idx]
        pred = extract_answer(content)
        n_search = len(re.findall(r"<search>.*?</search>", content, re.DOTALL))
        print(
            f"\n{'='*60}\n"
            f"  Golden : {golden}\n"
            f"  Pred   : {pred!r}\n"
            f"  EM     : {em_check(pred, golden) if pred else 0:.3f}\n"
            f"  Searches: {n_search}\n"
            f"  --- Dialog ---\n"
            f"{_format_completion_dialog(content)}\n"
            f"{'='*60}\n"
        )
    return scores


def compute_score_f1(completions, solution, **kwargs):
    scores = []
    for completion, golden in zip(completions, solution):
        pred = extract_answer(completion[-1]["content"])
        scores.append(0.0 if pred is None else f1_check(pred, golden))
    return scores


def compute_search_persistence(completions, solution, **kwargs):
    scores = []
    for completion, golden in zip(completions, solution):
        content = completion[-1]["content"]
        num_search = len(re.findall(r"<search>.*?</search>", content, re.DOTALL))
        prediction = extract_answer(content)

        queries = re.findall(r"<search>(.*?)</search>", content, re.DOTALL)
        queries_norm = [q.strip().lower() for q in queries]
        num_dups = sum(
            1 for i in range(1, len(queries_norm))
            if queries_norm[i] == queries_norm[i - 1]
        )

        s = -0.10 * num_dups  # duplicate-query penalty

        if prediction is None:
            s -= 0.30  # strong penalty: must output an answer
        else:
            em = em_check(prediction, golden)
            if em > 0:
                if num_search <= 1:
                    s += 0.20
                elif num_search == 2:
                    s += 0.10
            else:
                if num_search >= 4:
                    s -= 0.10
                if num_search <= 1 and not _golden_in_information(content, golden):
                    s -= 0.10

        scores.append(s)
    return scores


def compute_retrieval_quality(completions, solution, **kwargs):
    return [
        0.20 if _golden_in_information(completion[-1]["content"], golden) else 0.0
        for completion, golden in zip(completions, solution)
    ]


def compute_answer_quality(completions, solution, **kwargs):
    scores = []
    prompts = kwargs.get("prompts", [None] * len(completions))
    for completion, golden, prompt in zip(completions, solution, prompts):
        pred = extract_answer(completion[-1]["content"])
        if pred is None:
            scores.append(0.0)
            continue
        s = 0.0
        n = len(pred.split())
        if n > 60:
            s -= 0.20
        if re.search(r"\bDoc\s*[123]\b", pred):
            s -= 0.10
        if prompt is not None:
            try:
                q_text = prompt[-1].get("content", "") if isinstance(prompt, list) else str(prompt)
                q_norm = set(normalize_answer(q_text).split())
                p_norm = set(normalize_answer(pred).split())
                if q_norm and p_norm:
                    overlap = len(p_norm & q_norm) / max(len(p_norm), 1)
                    if overlap >= 0.7 and len(p_norm - q_norm) <= 1:
                        s -= 0.10
            except Exception:
                pass
        scores.append(s)
    return scores


# ============================================================================
# Dataset
# ============================================================================

def format_example(example: dict) -> dict:
    prompt = SYSTEM_PROMPT + f" Question: {example['question']}\n"
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "solution": example["golden_answers"],
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    training_args.chat_template_kwargs = {"enable_thinking": False}

    print("Loading dataset...")
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    dataset = dataset.map(format_example, remove_columns=["question", "golden_answers"])

    print(f"Train: {len(dataset['train'])}  Eval: {len(dataset['test'])}")

    mt_rollout = create_mt_rollout_func(
        search_func=search,
        max_search_calls=5,
        max_iterations=15,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        rollout_func=mt_rollout,
        reward_funcs=[
            compute_score_em,
            compute_score_f1,
            compute_search_persistence,
            compute_retrieval_quality,
            compute_answer_quality,
        ],
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print("Done.")
