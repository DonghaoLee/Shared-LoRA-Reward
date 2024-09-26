# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="3"

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

# Evaluate the model
torchrun --nproc_per_node=1 --master_port=${port} src/reward_modeling.py \
    --model_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name openai/summarize_from_feedback \
    --dataset_subset comparisons \
    --selected_labeler personalized \
    --output_dir ./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora/checkpoint-630 \
    --checkpoint_paths ./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora/checkpoint-630/model-00001-of-00003.safetensors ./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora/checkpoint-630/model-00002-of-00003.safetensors ./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora/checkpoint-630/model-00003-of-00003.safetensors \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 0 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --learning_rate 5.0e-5 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --save_strategy epoch \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_task_type SEQ_CLS \
    --lora_target_modules q_proj v_proj \
    --lora_type lora \
    --save_only_model True \
    --report_to none \
    --eval_mode True \
