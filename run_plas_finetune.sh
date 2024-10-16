# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0,1"
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
# dataset_name='nvidia/HelpSteer2'

######## Reward Model base models ########
model_name='EleutherAI/gpt-j-6b'
# model_name='meta-llama/Meta-Llama-3-8B'

######## Reward Model checkpoint directory ########
# checkpoint_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/lora/checkpoint-420'
# checkpoint_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/lora_all_warmup/checkpoint-63'
checkpoint_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/lora_all_warmup/checkpoint-126'
checkpoint_paths="${checkpoint_dir}/model.safetensors"
# num_safe_tensors=4      # Number of safe tensors to load, this is 4 for the llama3-8b model and 3 for the gpt-j-6b model
# # Get the number of digits for the total number of files
# num_digits=$(printf "%05g" ${num_safe_tensors})
# # Get the checkpoint files in the directory using a for loop
# for i in $(seq -f "%05g" 1 ${num_safe_tensors})
# do
#     checkpoint_paths="${checkpoint_paths} ${checkpoint_dir}/model-${i}-of-${num_digits}.safetensors"
# done

# # checkpoint_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/lora_all/checkpoint-420'
# checkpoint_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/lora_all_warmup/checkpoint-63'
# num_safe_tensors=4      # Number of safe tensors to load, this is 4 for the llama3-8b model and 3 for the gpt-j-6b model
# # Get the number of digits for the total number of files
# num_digits=$(printf "%05g" ${num_safe_tensors})
# # Get the checkpoint files in the directory using a for loop
# for i in $(seq -f "%05g" 1 ${num_safe_tensors})
# do
#     checkpoint_paths="${checkpoint_paths} ${checkpoint_dir}/model-${i}-of-${num_digits}.safetensors"
# done

######## Reward Model output directory ########
#### GPT-J 6B ####
# output_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora_ft_warmup/shared_A_personalized_B/single_gpu_run1'
# output_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora_ft_warmup/personalized_A_shared_B/single_gpu_run1'

# output_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora_ft_warmup/shared_A_personalized_B/multi_gpu_run3'
output_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/pslora_ft_warmup/personalized_A_shared_B/multi_gpu_run3'

#### LLAMA 3-8B ####
# output_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/pslora_ft_warmup/shared_A_personalized_B/single_gpu_run1'

######## Reward Model training parameters ########
learning_rate=5.0e-5
lora_personalize_strategy='personalized_A'          # 'personalized_A', 'personalized_B'
lora_optimize_modules="lora_A lora_B"               # Default is "lora_A lora_B lora_kernel lora_singular"

# Train the model
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=2 --main_process_port=${port} src/reward_modeling.py \
    --model_name_or_path ${model_name} \
    --dataset_name openai/summarize_from_feedback \
    --dataset_subset comparisons \
    --local_trainset_path ${local_trainset_path} \
    --local_validset_path ${local_validset_path} \
    --local_testset_path ${local_testset_path} \
    --selected_labeler personalized \
    --output_dir ${output_dir} \
    --checkpoint_paths ${checkpoint_paths} \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --bf16 true \
    --learning_rate ${learning_rate} \
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
    --lora_personalize_strategy ${lora_personalize_strategy} \
    --lora_optimize_modules ${lora_optimize_modules} \
    --save_only_model True \
    --report_to wandb \
    --seed 42 \
