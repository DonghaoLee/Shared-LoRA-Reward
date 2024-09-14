import time
import os
import wandb
import logging

import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from model import RewardModel
from lora import LinearLayer_LoRA, convert_linear_layer_to_lora, only_optimize_lora_parameters
from utils import rdit_top_N

# wandb is used as a logging tool. This is the initialization of wandb.
wandb_flag = False
if wandb_flag:
    wandb.init(project = 'ensemble reward model with LoRA')
    os.environ["WANDB_PROJECT"] = "ensemble reward model with LoRA"
np.random.seed(42)

def main():
    ###############
    # Load datasets
    ###############
    # Here we load the reddit TL;DR dataset: https://huggingface.co/datasets/openai/summarize_from_feedback
    rdit_comp_dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
    # The following is to find the top 5 workers and pick the preference indices.
    N_user = 5
    user_dict, dataset = rdit_top_N(rdit_comp_dataset, N_user=N_user)

    # load model. More details can be found in
    # https://huggingface.co/EleutherAI/gpt-j-6b
    model = AutoModel.from_pretrained('EleutherAI/gpt-j-6b')
    # load tokenizer. It will embeds the input sentence.
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b', padding_side = 'right', truncation_side = 'right')
    if tokenizer.pad_token_id is None:
        logging.info("pad_token_id is None, setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048
    logging.info(f"Max number of tokens in a sequence: {tokenizer.model_max_length}")

    # Set the device. No parallel.
    device = "cuda:0"

    # Build a reward model
    reward_model = RewardModel(tokenizer, model)
    convert_linear_layer_to_lora(reward_model, 'rwtransformer', lora_dim = 64, inds = N_user) # "attn."
    only_optimize_lora_parameters(reward_model)
    # Set part of the parameters fixed in the training of reward model
    # TODO: May not necessary
    # keyword_list = ["layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"]
    # for n, p in reward_model.named_parameters():
    #     flag = False
    #     for key_word in keyword_list:
    #         if key_word in n.lower():
    #             flag = True
    #             break
    #     if flag:
    #         p.requires_grad = False
    reward_model = reward_model.to(device)

    for n, p in reward_model.named_parameters():
        print(n, p.shape, p.requires_grad)

    # Set the optimizer. The following optimizer may not be optimal.
    # It just works in this program.
    # lr schedule is expected in the future.
    optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))

    start_time = time.time()
    batch = 1
    for epoch in range(0): # 2 epochs
        # Shuffle
        train_set = dataset.shuffle()
        for i in range(train_set.num_rows // batch):
            text = []
            for posi in range(2):
                for j in range(batch):
                    label = dataset[i * batch + j]['choice']
                    prompt = dataset[i * batch + j]['info']['post']
                    response = dataset[i * batch + j]['summaries'][(posi + label) % 2]['text']
                    text.append(prompt + '[pad]' + response)

            p = []
            
            # TODO:
            # Current computation process can not use the parallel in one batch
            for _ in range(batch):
                LinearLayer_LoRA.index = user_dict[dataset[i * batch + j]['worker']]
                reward_model.index = user_dict[dataset[i * batch + j]['worker']]
                token = tokenizer([text[j], text[batch + j]],
                                  padding = True, truncation = True,
                                  return_tensors = 'pt', max_length=512)
                for k, v in token.items():
                    token[k] = v.to(device)
                output = reward_model(**token)
                p.append(output["probability"])
            
            p = torch.stack(p, dim=1) # bs, ...
            print("p:", p.shape)
            loss = - torch.mean(torch.log(p))

            if wandb_flag:
                wandb.log({
                    'loss': loss.item(),
                })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(reward_model.state_dict(), 'ckpt/federated_lora_epoch_' + str(epoch) + '.ckpt')

    end_time = time.time()
    print('time:', end_time - start_time)

    if wandb_flag:
        wandb.finish()


if __name__ == '__main__':
    main()
