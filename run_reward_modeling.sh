# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES="2"
=======
export CUDA_VISIBLE_DEVICES="3"
export WANDB_PROJECT="ensemble reward model with LoRA"
>>>>>>> bfb82f16480fff7142082efc3032c02b33194ff1

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

######## Reward Model base models ########
# model_name='EleutherAI/gpt-j-6b'
model_name='meta-llama/Meta-Llama-3-8B'

######## Reward Model output directory ########
output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/lora_all_warmup'

# Train the model
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=${port} src/reward_modeling.py \
    --model_name_or_path ${model_name} \
    --dataset_name openai/summarize_from_feedback \
    --dataset_subset comparisons \
    --selected_labeler all \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --warmup_ratio 0.1 \
    --learning_rate 5.0e-5 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --save_strategy steps \
    --save_steps 0.1 \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_task_type SEQ_CLS \
    --lora_target_modules q_proj v_proj \
    --lora_type lora \
    --save_only_model True \
    --report_to wandb \
