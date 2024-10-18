# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="2"
export WANDB_PROJECT="ensemble reward model with LoRA"

port=$(shuf -i 6000-9000 -n 1)
echo $port

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

######## Dataset ########
dataset_name='openai/summarize_from_feedback'
local_trainset_path='./data/summarize_from_feedback/downsampled_trainset'
local_validset_path='./data/summarize_from_feedback/per_labeler_validset'
local_testset_path='./data/summarize_from_feedback/downsampled_testset'

######## Reward Model base models ########
model_name='EleutherAI/gpt-j-6b'
# model_name='meta-llama/Meta-Llama-3-8B'

######## Reward Model output directory ########
output_dir='./exp/gpt-j-6b-Reward/lr-1e-4-2epochs-lorar32/lora_all_warmup'
# output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/pslora/personalized_A'
# output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/pslora/personalized_A_2d_init_expand'
# output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/pslora/personalized_B'

######## Reward Model training parameters ########
learning_rate=1.0e-4
lora_personalize_strategy='personalized_A'          # 'personalized_A', 'personalized_B'
lora_optimize_modules="lora_A lora_B"               # Default is "lora_A lora_B lora_kernel lora_singular"
selected_labeler='all'                              # 'personalized', 'all', or specific labeler index number

######## Reward Model training command ########
# Train the model
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=1 --main_process_port=${port} src/reward_modeling.py \
    --model_name_or_path ${model_name} \
    --dataset_name openai/summarize_from_feedback \
    --dataset_subset comparisons \
    --selected_labeler ${selected_labeler} \
    --local_trainset_path ${local_trainset_path} \
    --local_validset_path ${local_validset_path} \
    --local_testset_path ${local_testset_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --learning_rate ${learning_rate} \
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
    --lora_personalize_strategy ${lora_personalize_strategy} \
    --lora_optimize_modules ${lora_optimize_modules} \
    --save_only_model True \
    --report_to wandb \
    --seed 42 \
