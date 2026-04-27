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
    
python train_grpo_v2.py \
        --model_name_or_path /home/l33500/models/Qwen/Qwen3-0.6B \
        --output_dir outputs/search-r1-v2 \
        --max_completion_length 8192 \
        --num_generations 4 \
        --log_completions True \
        --max_steps 500