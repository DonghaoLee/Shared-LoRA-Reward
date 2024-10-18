# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0"
# export WANDB_PROJECT="PSLoRA"

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

######## Reward Model checkpoint directory ########
checkpoint_dir='./exp/gpt-j-6b-Reward/lr-5e-5-3epochs-lorar32/lora/checkpoint-210'
# checkpoint_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/lora_all/checkpoint-630'           # 3 epochs, vanilla LoRA
# checkpoint_dir='./exp/llama3-8b-Reward/lr-5e-5-3epochs-lorar32/pslora/checkpoint-630'               # 3 epochs, PS-LoRA
num_safe_tensors=4      # Number of safe tensors to load, this is 4 for the llama3-8b model and 3 for the gpt-j-6b model
# Get the number of digits for the total number of files
num_digits=$(printf "%05g" ${num_safe_tensors})
# Get the checkpoint files in the directory using a for loop
for i in $(seq -f "%05g" 1 ${num_safe_tensors})
do
    checkpoint_paths="${checkpoint_paths} ${checkpoint_dir}/model-${i}-of-${num_digits}.safetensors"
done

# Evaluate the model
python src/reward_modeling.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --dataset_subset comparisons \
    --selected_labeler personalized \
    --output_dir ${checkpoint_dir} \
    --checkpoint_paths ${checkpoint_paths} \
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
    # --report_to wandb \
    --eval_mode True \
