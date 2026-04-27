"""
Search-enabled rollout function for GRPO Trainer (v2).

Uses trainer._generate_single_turn for generation, following the same pattern
as trl's _tool_call_loop: token IDs are concatenated (prompt + completion + tool_suffix)
and passed to _generate_single_turn for continuation.

Conversation flow:
    Input: [{"role": "user", "content": "Question?"}]
    Model generates: "Let me think...<search>query</search>"
    Inject search result: <information>result</information> (as tool suffix)
    Model continues until <answer> tag

Usage:
    from search_rollout_v2 import create_search_rollout_func

    trainer = GRPOTrainer(
        model=model_path,
        rollout_func=create_search_rollout_func(search, max_calls=10),
        train_dataset=dataset,
        reward_funcs=[compute_score_em],
        args=training_args,
    )
"""

import re
from typing import Callable




class SearchRolloutManagerV2:
    """
    Manages search-enabled generation using trainer._generate_single_turn.

    Uses messages-based conversation building (same as eval_search.py):
    1. Build input_ids via apply_chat_template(current_messages)
    2. Call _generate_single_turn(input_ids) -> completion_ids
    3. If <search> found, append assistant + tool messages to current_messages
    4. Repeat until <answer> or max iterations
    """

    def __init__(
        self,
        search_func: Callable[[str], str],
        tokenizer,
        max_search_calls: int = 10,
        max_iterations: int = 20,
    ):
        self.search_func = search_func
        self.tokenizer = tokenizer
        self.max_search_calls = max_search_calls
        self.max_iterations = max_iterations

        # Patterns
        self.search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    def _apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        """Apply chat template to messages and return token IDs."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
            )
        except Exception:
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            return self.tokenizer.encode(text, add_special_tokens=False)

    def extract_first_search(self, text: str) -> str | None:
        """Extract first <search>query</search> from text."""
        match = self.search_pattern.search(text)
        return match.group(1).strip() if match else None

    def has_answer(self, text: str) -> bool:
        """Check if text contains <answer> tag."""
        return bool(self.answer_pattern.search(text))

    def generate_with_search(
        self,
        trainer,
        messages: list[dict],
        image,
        multimodal_fields: dict,
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Generate completion with search tool calls.

        Uses messages-based conversation building (same as eval_search.py):
        - Each turn, apply_chat_template(current_messages) builds input_ids
        - On <search>, append assistant + tool messages to current_messages
        - tool_suffix_ids computed via diff method for completion_ids tracking

        Args:
            trainer: The GRPOTrainer instance
            messages: The original prompt messages (e.g. [{"role": "user", "content": "..."}])
            image: Image for multimodal (can be None)
            multimodal_fields: Multimodal fields dict

        Returns:
            (completion_ids, tool_mask, logprobs)
        """
        all_completion_ids = []
        all_tool_mask = []
        all_logprobs = []
        search_count = 0

        # Use messages to track conversation (same as eval_search.py)
        current_messages = list(messages)

        for _ in range(self.max_iterations):
            # Build input_ids via chat template (messages-based)
            input_ids = self._apply_chat_template(
                current_messages, add_generation_prompt=True
            )["input_ids"]

            # Generate continuation using _generate_single_turn
            gen_ids, gen_logprobs = trainer._generate_single_turn(
                [input_ids],
                [image],
                multimodal_fields,
            )

            if not gen_ids or not gen_ids[0]:
                break

            new_ids = gen_ids[0]
            new_logprobs = gen_logprobs[0] if gen_logprobs else None

            # Decode to check for tags
            generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

            # Model-generated tokens → mask=1
            all_completion_ids.extend(new_ids)
            all_tool_mask.extend([1] * len(new_ids))
            if new_logprobs is not None:
                all_logprobs.extend(new_logprobs)
            else:
                all_logprobs.extend([0.0] * len(new_ids))

            # Check for <search> tag
            query = self.extract_first_search(generated_text)

            if query and search_count < self.max_search_calls:
                search_count += 1

                # Call search
                try:
                    search_result = self.search_func(query)
                except Exception as e:
                    search_result = f"Error during search: {str(e)}"

                info_content = f"<information>\n{search_result}\n</information>"

                # Append assistant + tool messages (same as eval_search.py)
                current_messages.append({"role": "assistant", "content": generated_text})
                current_messages.append({"role": "user", "content": info_content})

                # Compute tool suffix IDs for completion_ids tracking
                tool_suffix_ids = self._compute_tool_suffix(generated_text, info_content)
                all_completion_ids.extend(tool_suffix_ids)
                all_tool_mask.extend([0] * len(tool_suffix_ids))
                all_logprobs.extend([0.0] * len(tool_suffix_ids))
                continue

            # No search — append assistant message and finalize
            current_messages.append({"role": "assistant", "content": generated_text})
            break

        return all_completion_ids, all_tool_mask, all_logprobs

    def _compute_tool_suffix(
        self,
        assistant_text: str,
        info_content: str,
    ) -> list[int]:
        """
        Compute the tool suffix IDs by comparing chat template outputs.

        Same approach as trl's _get_tool_suffix_ids:
        1. Prefix: user + assistant (no generation prompt)
        2. Full: user + assistant + tool (with generation prompt)
        3. suffix = full_ids - prefix_ids (with EOS alignment)
        """
        user_msg = {"role": "user", "content": "dummy"}
        assistant_msg = {"role": "assistant", "content": assistant_text}
        tool_msg = {"role": "user", "content": info_content}

        # Prefix: user + assistant (no generation prompt)
        prefix_ids = self._apply_chat_template(
            [user_msg, assistant_msg], add_generation_prompt=False
        )["input_ids"]

        # Full: user + assistant + tool (with generation prompt)
        full_ids = self._apply_chat_template(
            [user_msg, assistant_msg, tool_msg], add_generation_prompt=True
        )["input_ids"]

        # Align on EOS boundary (like trl's _get_tool_suffix_ids)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id in prefix_ids:
            last_eos_idx = max(i for i, tok_id in enumerate(prefix_ids) if tok_id == eos_token_id)
            prefix_ids_trimmed = prefix_ids[: last_eos_idx + 1]
        else:
            prefix_ids_trimmed = prefix_ids

        # The suffix is the difference
        if full_ids[:len(prefix_ids_trimmed)] == prefix_ids_trimmed:
            return full_ids[len(prefix_ids_trimmed):]
        else:
            # Fallback: just encode the tool content directly
            return self.tokenizer.encode(info_content, add_special_tokens=False)


def create_search_rollout_func(
    search_func: Callable[[str], str],
    max_search_calls: int = 10,
    max_iterations: int = 20,
):
    """
    Create a rollout function using trainer._generate_single_turn.

    Follows the same pattern as trl's _tool_call_loop:
    - Token IDs are concatenated (prompt + completion + tool_suffix)
    - _generate_single_turn is called with the concatenated IDs for continuation
    - tool_mask marks model-generated tokens (1) vs injected tokens (0)

    Args:
        search_func: Function that takes a query and returns search results
        max_search_calls: Maximum search calls per generation
        max_iterations: Maximum tool calling iterations

    Returns:
        A rollout function compatible with GRPOTrainer
    """

    def rollout_func(prompts: list, trainer) -> dict:
        """
        Custom rollout function for GRPOTrainer.

        Args:
            prompts: List of prompts (each is a list of message dicts).
                     NOTE: trl's RepeatSampler already repeats each prompt
                     num_generations times, so we only generate ONE completion
                     per prompt entry.
            trainer: The GRPOTrainer instance

        Returns:
            dict with prompt_ids, completion_ids, logprobs, env_mask
        """
        # Tokenize prompts using trainer's method (handles chat template correctly)
        prompt_ids_list, images, multimodal_fields = trainer._tokenize_prompts(prompts)

        # Initialize manager
        manager = SearchRolloutManagerV2(
            search_func=search_func,
            tokenizer=trainer.processing_class,
            max_search_calls=max_search_calls,
            max_iterations=max_iterations,
        )

        # Generate one completion per prompt (RepeatSampler already handles num_generations)
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_tool_masks = []

        for idx, prompt_ids in enumerate(prompt_ids_list):
            image = images[idx] if images else None

            # Pass original messages (not prompt_ids) — messages-based rollout
            completion_ids, tool_mask, logprobs = manager.generate_with_search(
                trainer,
                prompts[idx],
                image,
                multimodal_fields,
            )
            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            all_tool_masks.append(tool_mask)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_tool_masks,
        }

    return rollout_func
