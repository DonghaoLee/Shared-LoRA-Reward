# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="2"

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

# accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=${port} src/reward_modeling.py \
#     --model_name_or_path EleutherAI/gpt-j-6b \
#     --dataset_name openai/summarize_from_feedback \
#     --dataset_subset comparisons \
#     --selected_labeler 4 \
#     --output_dir ./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/debug \
#     --per_device_train_batch_size 8 \
#     --num_train_epochs 3 \
#     --gradient_accumulation_steps 4 \
#     --remove_unused_columns False \
#     --gradient_checkpointing True \
#     --bf16 true \
#     --learning_rate 5.0e-5 \
#     --logging_steps 5 \
#     --eval_strategy steps \
#     --eval_steps 0.2 \
#     --max_length 2048 \
#     --use_peft \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --lora_task_type SEQ_CLS \
#     --report_to wandb \

# Train the model
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=${port} src/reward_modeling.py \
    --model_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name openai/summarize_from_feedback \
    --dataset_subset comparisons \
    --selected_labeler 4 \
    --output_dir ./exp/gpt-j-6b-Reward/lr-5e-5-4epochs-lorar32-user-5/pslora \
    --checkpoint_paths ./exp/gpt-j-6b-Reward/lr-5e-5-4epochs-lorar32-all/pslora/checkpoint-630/model-00001-of-00003.safetensors ./exp/gpt-j-6b-Reward/lr-5e-5-4epochs-lorar32-all/pslora/checkpoint-630/model-00002-of-00003.safetensors ./exp/gpt-j-6b-Reward/lr-5e-5-4epochs-lorar32-all/pslora/checkpoint-630/model-00003-of-00003.safetensors \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --learning_rate 1.25e-5 \
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
    --report_to wandb \
