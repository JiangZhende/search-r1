# python grpo_agent.py \
#     --model_name_or_path /home/l33500/models/Qwen/Qwen3-14B \
#     --output_dir grpo_biogrid_qwen_3g-1.7b \
#     --push_to_hub False \
#     --use_vllm False \
#     --vllm_mode colocate \
#     --max_completion_length 4096 \
#     --report_to none \
#     --log_completions True \
#     --max_steps 10000 \
#     --save_strategy no \
#     --eval_strategy no \
    
# python train_grpo_v2.py \
#         --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
#         --output_dir outputs/search-r1-v2 \
#         --max_completion_length 8192 \
#         --num_generations 4 \
#         --log_completions True \
#         --max_steps 500

# python train_search_r1.py \
#     --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
#     --output_dir outputs/search-r1 \
#     --max_completion_length 4096 \
#     --num_generations 4 \
#     --log_completions True \
#     --max_steps 500 \
#     --report_to none

python train_search_r1.py \
    --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
    --output_dir outputs/search-r1-v5 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --max_steps 800 \
    --save_steps 100 \
    --beta 0.02 \
    --reward_weights 1.0 1.0 0.1 1.0 \
    --temperature 1.0 \
    --log_completions True \
    --report_to none