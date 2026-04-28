"""
Search-enabled rollout function for GRPO Trainer.

Mirrors `test.py`'s generation loop:
  1. Generate until ``</search>`` is detected (via stopping_criteria).
  2. Extract the query, call the search backend, inject
     ``<information>...</information>`` into the running completion.
  3. Continue generation until ``<answer>`` / EOS / budget exhaustion.

TRL ``rollout_func`` contract notes:
  * Prompts arriving here are ALREADY repeated ``num_generations`` times by
    the trainer's sampler (see ``RepeatSampler``).  Therefore we generate
    EXACTLY ONE completion per incoming prompt.
  * ``logprobs`` is returned as ``None`` so the trainer skips the
    importance-sampling correction (same model is used for both sampling
    and the policy logprobs computation, identical to the non-vLLM
    default path).
  * Tokens injected by the environment (``<information>...``) are masked
    out of the loss via the optional ``env_mask`` field (1=model, 0=env).

Usage:
    from search_rollout import create_search_rollout_func

    trainer = GRPOTrainer(
        model=model_path,
        rollout_func=create_search_rollout_func(search, max_search_calls=10),
        train_dataset=dataset,
        reward_funcs=[compute_score_em],
        args=training_args,
    )
"""

from __future__ import annotations

import copy
import re
from typing import Callable

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList

try:
    from trl.models import unwrap_model_for_generation
except ImportError:  # older trl layout
    from trl.extras.profiling import unwrap_model_for_generation  # type: ignore


# ---------------------------------------------------------------------------
# Stopping criterion
# ---------------------------------------------------------------------------

class StopOnSequence(StoppingCriteria):
    """Stop generation when any of the target token sequences appears at the tail."""

    def __init__(self, target_sequences: list[str], tokenizer: PreTrainedTokenizerBase):
        self.target_ids = [
            tokenizer.encode(t, add_special_tokens=False) for t in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]
        self._min_len = min(self.target_lengths) if self.target_lengths else 1

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> bool:
        if input_ids.shape[1] < self._min_len:
            return False
        device = input_ids.device
        # Stop only when *every* row has hit a target (batch-aware).
        for row in input_ids:
            row_hit = False
            for tlen, tids in zip(self.target_lengths, self.target_ids):
                if row.shape[0] >= tlen and torch.equal(
                    row[-tlen:], torch.as_tensor(tids, device=device)
                ):
                    row_hit = True
                    break
            if not row_hit:
                return False
        return True


# ---------------------------------------------------------------------------
# Rollout manager
# ---------------------------------------------------------------------------

_SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEARCH_TEMPLATE = "\n\n<information>{search_results}</information>\n\n"
_STOP_SEQUENCES = [
    "</search>", " </search>",
    "</search>\n", " </search>\n",
    "</search>\n\n", " </search>\n\n",
]


class SearchRolloutManager:
    """Drive a single prompt through the search-aware generation loop."""

    def __init__(
        self,
        search_func: Callable[[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_search_calls: int = 10,
        max_iterations: int = 20,
        max_completion_length: int = 1024,
        temperature: float = 0.9,
        top_p: float = 1.0,
        chunk_max_new_tokens: int = 512,
    ):
        self.search_func = search_func
        self.tokenizer = tokenizer
        self.max_search_calls = max_search_calls
        self.max_iterations = max_iterations
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.chunk_max_new_tokens = chunk_max_new_tokens

        # Qwen2.5/Qwen3 EOS variants.
        self.eos_ids = {tokenizer.eos_token_id, 151645, 151643}
        self.eos_ids.discard(None)

        self.stopping_criteria = StoppingCriteriaList(
            [StopOnSequence(_STOP_SEQUENCES, tokenizer)]
        )

    @staticmethod
    def extract_first_search(text: str) -> str | None:
        m = _SEARCH_PATTERN.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def has_answer(text: str) -> bool:
        return bool(_ANSWER_PATTERN.search(text))

    @torch.no_grad()
    def generate(
        self,
        unwrapped_model,
        generation_config,
        prompt_ids: list[int],
        device: torch.device,
    ) -> tuple[list[int], list[int]]:
        """Generate one completion. Returns ``(completion_ids, env_mask)``.

        ``env_mask`` is 1 for model-generated tokens and 0 for tokens
        injected by the environment (``<information>...`` blocks).
        """
        completion_ids: list[int] = []
        env_mask: list[int] = []
        search_count = 0

        running_input = torch.tensor([prompt_ids], device=device)

        for _ in range(self.max_iterations):
            remaining = self.max_completion_length - len(completion_ids)
            if remaining <= 0:
                break

            # Clone the trainer's generation_config and override max_new_tokens
            # for this chunk. Passing generation_config together with the same
            # kwarg is deprecated, hence the copy.
            if generation_config is not None:
                gen_cfg = copy.deepcopy(generation_config)
                gen_cfg.max_new_tokens = min(self.chunk_max_new_tokens, remaining)
                outputs = unwrapped_model.generate(
                    input_ids=running_input,
                    attention_mask=torch.ones_like(running_input),
                    stopping_criteria=self.stopping_criteria,
                    generation_config=gen_cfg,
                )
            else:
                outputs = unwrapped_model.generate(
                    input_ids=running_input,
                    attention_mask=torch.ones_like(running_input),
                    max_new_tokens=min(self.chunk_max_new_tokens, remaining),
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

            new_tokens = outputs[0][running_input.shape[1]:].tolist()
            if not new_tokens:
                break

            # Respect the completion budget.
            if len(completion_ids) + len(new_tokens) > self.max_completion_length:
                new_tokens = new_tokens[: self.max_completion_length - len(completion_ids)]

            completion_ids.extend(new_tokens)
            env_mask.extend([1] * len(new_tokens))

            # Natural EOS → done.
            if new_tokens[-1] in self.eos_ids:
                break

            chunk_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # print(chunk_text)
            if self.has_answer(chunk_text):
                break

            query = self.extract_first_search(chunk_text)
            if query and search_count < self.max_search_calls:
                search_count += 1
                try:
                    result = self.search_func(query)
                except Exception as e:  # propagate as observation rather than crash
                    result = f"Search error: {e}"

                info_text = _SEARCH_TEMPLATE.format(search_results=result)
                info_ids = self.tokenizer.encode(info_text, add_special_tokens=False)

                remaining = self.max_completion_length - len(completion_ids)
                if remaining <= 0:
                    break
                info_ids = info_ids[:remaining]

                completion_ids.extend(info_ids)
                env_mask.extend([0] * len(info_ids))

                running_input = torch.tensor(
                    [prompt_ids + completion_ids], device=device
                )
                continue

            # Neither search nor answer: keep generating.
            running_input = torch.tensor(
                [prompt_ids + completion_ids], device=device
            )

        # Avoid empty completions (would crash downstream tensor packing).
        if not completion_ids:
            eos = self.tokenizer.eos_token_id
            completion_ids = [eos if eos is not None else 0]
            env_mask = [1]

        return completion_ids, env_mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_search_rollout_func(
    search_func: Callable[[str], str],
    max_search_calls: int = 10,
    max_iterations: int = 20,
    chunk_max_new_tokens: int = 512,
):
    """Create a ``rollout_func`` for ``GRPOTrainer``.

    The returned callable receives the per-process prompt slice (already
    repeated ``num_generations`` times by the trainer's sampler) and
    returns one completion per prompt.
    """

    def rollout_func(prompts: list, trainer) -> dict:
        tokenizer = trainer.processing_class
        device = (
            trainer.accelerator.device
            if hasattr(trainer, "accelerator")
            else trainer.model.device
        )

        manager = SearchRolloutManager(
            search_func=search_func,
            tokenizer=tokenizer,
            max_search_calls=max_search_calls,
            max_iterations=max_iterations,
            max_completion_length=trainer.max_completion_length,
            temperature=getattr(trainer, "temperature", 0.9),
            top_p=getattr(trainer, "top_p", 1.0),
            chunk_max_new_tokens=chunk_max_new_tokens,
        )

        # Trainer-side tokenization (handles chat templating consistently).
        prompt_ids_list, _images, _mm_fields = trainer._tokenize_prompts(prompts)

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_env_masks: list[list[int]] = []

        model_wrapped = getattr(trainer, "model_wrapped", trainer.model)
        gather_ds3 = getattr(getattr(trainer, "args", None), "ds3_gather_for_generation", True)
        generation_config = getattr(trainer, "generation_config", None)

        with unwrap_model_for_generation(
            model_wrapped,
            trainer.accelerator,
            gather_deepspeed3_params=gather_ds3,
        ) as unwrapped_model:
            # One completion per prompt (prompts are pre-duplicated).
            for pids in prompt_ids_list:
                pid_list = pids.tolist() if hasattr(pids, "tolist") else list(pids)
                completion_ids, env_mask = manager.generate(
                    unwrapped_model, generation_config, pid_list, device
                )
                all_prompt_ids.append(pid_list)
                all_completion_ids.append(completion_ids)
                all_env_masks.append(env_mask)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            # Same model is used for sampling and policy logprobs → no IS
            # correction needed; the trainer recomputes exact logprobs.
            "logprobs": None,
            "env_mask": all_env_masks,
        }

    return rollout_func
