"""
Search-enabled rollout function for GRPO Trainer.

This module provides a custom rollout function that handles the <search> tag format
for tool calling during GRPO training.

Usage:
    from search_rollout import create_search_rollout_func
    
    trainer = GRPOTrainer(
        model=model_path,
        search_func=search,
        rollout_func=create_search_rollout_func(search, max_calls=10),
        train_dataset=dataset,
        reward_funcs=[compute_score_em],
        args=training_args,
    )
"""

import re
from typing import Callable

import torch
import transformers
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList


class StopOnSequence(StoppingCriteria):
    """Stop generation when target sequence is generated."""
    
    def __init__(self, target_sequences: list[str], tokenizer: PreTrainedTokenizerBase):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False) 
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


class SearchRolloutManager:
    """
    Manages the search-enabled generation process.
    
    This class handles:
    1. Parsing <search>query</search> tags
    2. Calling the search function
    3. Injecting <information>result</information> 
    4. Continuing generation until <answer> tag
    """
    
    def __init__(
        self,
        search_func: Callable[[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_search_calls: int = 10,
        max_iterations: int = 20,
        temperature: float = 0.9,
        max_completion_length: int = 512,
    ):
        """
        Args:
            search_func: Function that takes a query and returns search results
            tokenizer: Tokenizer for encoding/decoding
            max_search_calls: Maximum search calls per generation
            max_iterations: Maximum tool calling iterations
            temperature: Sampling temperature
            max_completion_length: Maximum completion length
        """
        self.search_func = search_func
        self.tokenizer = tokenizer
        self.max_search_calls = max_search_calls
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        
        # Patterns
        self.search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
        # EOS tokens for Qwen models
        self.curr_eos = [151645, 151643]
        
        # Stopping sequences for </search> tag
        self.search_stop_sequences = [
            "</search>", " </search>", "</search>\n", 
            " </search>\n", "</search>\n\n", " </search>\n\n"
        ]
        self.stopping_criteria = StoppingCriteriaList([
            StopOnSequence(self.search_stop_sequences, tokenizer)
        ])
    
    def extract_first_search(self, text: str) -> tuple[str | None, int, int]:
        """
        Extract first <search>query</search> from text.
        
        Returns:
            (query, start_pos, end_pos) or (None, -1, -1)
        """
        match = self.search_pattern.search(text)
        if match:
            return match.group(1).strip(), match.start(), match.end()
        return None, -1, -1
    
    def has_answer(self, text: str) -> bool:
        """Check if text contains <answer> tag."""
        return bool(self.answer_pattern.search(text))
    
    def inject_information(self, text: str, position: int, result: str) -> str:
        """
        Inject <information>result</information> at position.
        """
        info_tag = f"<information>\n{result}\n</information>"
        return text[:position] + info_tag + text[position:]
    
    def generate_with_search(
        self,
        trainer,
        prompt_ids: list[int],
        images: list,
        multimodal_fields: dict,
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Generate completion with search tool calls using stopping_criteria.
        
        This method mimics test.py behavior:
        1. Generate until </search> is detected (via stopping_criteria)
        2. Extract query, call search, inject <information>result</information>
        3. Continue generation
        4. Repeat until <answer> or max_iterations
        
        Args:
            trainer: The GRPOTrainer instance
            prompt_ids: Prompt token IDs (list)
            images: Images list for multimodal (can be None)
            multimodal_fields: Multimodal fields dict
            
        Returns:
            (completion_ids, tool_mask, logprobs)
            - completion_ids: List of completion token IDs
            - tool_mask: List of 1s (model-generated) and 0s (injected)
            - logprobs: List of log probabilities
        """
        device = trainer.accelerator.device if hasattr(trainer, 'accelerator') else trainer.model.device
        model = trainer.model
        
        current_input_ids = torch.tensor([prompt_ids], device=device)
        completion_ids = []
        tool_mask = []
        all_logprobs = []
        search_count = 0
        
        for iteration in range(self.max_iterations):
            # Check if we've exceeded max completion length
            if len(completion_ids) >= self.max_completion_length:
                break
            
            # Generate with stopping criteria
            with torch.no_grad():
                outputs = model.generate(
                    current_input_ids,
                    attention_mask=torch.ones_like(current_input_ids),
                    max_new_tokens=min(512, self.max_completion_length - len(completion_ids)),
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=self.temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Extract new tokens
            new_tokens = outputs.sequences[0][current_input_ids.shape[1]:].tolist()
            
            if not new_tokens:
                break
            
            # Check if generation stopped at EOS
            if new_tokens[-1] in self.curr_eos:
                # Model finished generation naturally
                completion_ids.extend(new_tokens[:-1])  # Exclude EOS
                tool_mask.extend([1] * (len(new_tokens) - 1))
                all_logprobs.extend([0.0] * (len(new_tokens) - 1))
                break
            
            # Decode to check for tags
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            # Check for <search> tag with stopping
            query, _, search_end = self.extract_first_search(generated_text)
            
            if query and search_count < self.max_search_calls:
                # Found search query - stopping_criteria stopped at </search>
                search_count += 1
                
                # All tokens are model-generated (stopping at </search>)
                completion_ids.extend(new_tokens)
                tool_mask.extend([1] * len(new_tokens))
                all_logprobs.extend([0.0] * len(new_tokens))
                
                # Call search
                try:
                    search_result = self.search_func(query)
                except Exception as e:
                    search_result = f"Error during search: {str(e)}"
                
                # Inject <information> tag
                info_text = f"\n\n<information>\n{search_result}\n</information>\n\n"
                info_tokens = self.tokenizer.encode(info_text, add_special_tokens=False)
                
                # Add injected tokens
                completion_ids.extend(info_tokens)
                tool_mask.extend([0] * len(info_tokens))
                all_logprobs.extend([0.0] * len(info_tokens))
                
                # Update current_input_ids for next iteration
                current_input_ids = torch.tensor(
                    [prompt_ids + completion_ids], 
                    device=device
                )
                
                continue
            
            # Check for <answer> tag
            if self.has_answer(generated_text):
                # Add all remaining tokens
                completion_ids.extend(new_tokens)
                tool_mask.extend([1] * len(new_tokens))
                all_logprobs.extend([0.0] * len(new_tokens))
                break
            
            # No search or answer - add tokens and continue
            completion_ids.extend(new_tokens)
            tool_mask.extend([1] * len(new_tokens))
            all_logprobs.extend([0.0] * len(new_tokens))
            
            # Update current_input_ids
            current_input_ids = torch.tensor(
                [prompt_ids + completion_ids], 
                device=device
            )
        
        return completion_ids, tool_mask, all_logprobs


def create_search_rollout_func(
    search_func: Callable[[str], str],
    max_search_calls: int = 10,
    max_iterations: int = 20,
):
    """
    Create a rollout function for GRPOTrainer that handles search tool calls.
    
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
            prompts: List of prompts (each is a string or list of message dicts)
            trainer: The GRPOTrainer instance
            
        Returns:
            dict with:
                - prompt_ids: List of prompt token ID lists
                - completion_ids: List of completion token ID lists
                - logprobs: List of log probability lists
                - tool_mask: List of tool masks (optional)
        """
        mode = "train" if trainer.model.training else "eval"
        num_generations = trainer.num_generations if mode == "train" else trainer.num_generations_eval
        
        # Initialize manager
        manager = SearchRolloutManager(
            search_func=search_func,
            tokenizer=trainer.processing_class,
            max_search_calls=max_search_calls,
            max_iterations=max_iterations,
            temperature=trainer.temperature,
            max_completion_length=trainer.max_completion_length,
        )
        
        # Tokenize prompts using trainer's method
        prompt_ids_list, images, multimodal_fields = trainer._tokenize_prompts(prompts)
        
        # Generate completions
        all_completion_ids = []
        all_logprobs = []
        all_tool_masks = []
        
        for prompt_idx, prompt_ids in enumerate(prompt_ids_list):
            # Get images for this prompt (if any)
            prompt_images = [images[prompt_idx]] if images else [None]
            
            # Generate multiple completions per prompt
            for _ in range(num_generations):
                completion_ids, tool_mask, logprobs = manager.generate_with_search(
                    trainer,
                    prompt_ids,
                    prompt_images,
                    multimodal_fields,
                )
                all_completion_ids.append(completion_ids)
                all_logprobs.append(logprobs)
                all_tool_masks.append(tool_mask)
        
        # Repeat prompt_ids for num_generations
        repeated_prompt_ids = []
        for ids in prompt_ids_list:
            repeated_prompt_ids.extend([ids] * num_generations)
        
        return {
            "prompt_ids": repeated_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_tool_masks,  # Will be treated as tool_mask
        }
    
    return rollout_func
