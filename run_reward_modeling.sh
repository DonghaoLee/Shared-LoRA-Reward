# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="2,3"
export WANDB_PROJECT="Federated LoRA"

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

######## Reward Model base models ########
model_name='EleutherAI/gpt-j-6b'
# model_name='meta-llama/Meta-Llama-3-8B'

######## Dataset ########
# dataset_name='openai/summarize_from_feedback'
dataset_name='nvidia/HelpSteer2'

######## Reward Model output directory ########
# output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/lora_all_warmup'
output_dir='./exp/gpt-j-6b/lr-5e-5-3epochs-lorar32/'

# Train the model
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=${port} src/reward_modeling.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --dataset_subset comparisons \
    --selected_labeler personalized \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --learning_rate 5.0e-5 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 1.0 \
    --save_strategy steps \
    --save_steps 1.0 \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_task_type SEQ_CLS \
    --lora_target_modules q_proj v_proj \
    --lora_type lora \
    --save_only_model True \
    --report_to wandb \


    # --warmup_ratio 0.1 \