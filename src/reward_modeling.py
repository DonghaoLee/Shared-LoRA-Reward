# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState, Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from trl.extras.dataset_formatting import conversations_formatting_function

from utils import reddit_comp_top_N, reddit_prompt_template
from pslora import LinearLayer_PSLoRA, convert_linear_layer_to_lora, only_optimize_lora_parameters


tqdm.pandas()

os.environ["WANDB_PROJECT"] = "ensemble reward model with LoRA"
# os.environ["WANDB_PROJECT"] = "PSLoRA_RewardModel_Debugging"

warnings.simplefilter("once")

@dataclass
class RewardScriptArguments:
    dataset_name: str = field(
        default="HuggingFaceH4/ultrafeedback_binarized",
        metadata={"help": "the dataset name"},
    )
    dataset_subset: str = field(
        default="comparisons",
        metadata={"help": "the dataset subset to use"},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to evaluate on"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )
    num_labelers: int = field(default=5, metadata={"help": "Number of workers/labelers to consider for Reward Model"})
    apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to apply chat template to the dataset"},
    )
    selected_labeler: str = field(
        default="all",
        metadata={"help": "The selected labeler to use for the reward model. Default is all labelers. Options: all, [0-num_labelers]"},
    )


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
                or "labeler_index" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected`, `attention_mask_rejected` and `labeler_index`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            verbose=False,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            verbose=False,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "labeler_index": torch.tensor([feature["labeler_index"] for feature in features]),
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


class PSRewardTrainer(RewardTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for the model.
        Args:
            model (`torch.nn.Module`):
                The model to compute the loss for.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs to the model.
            return_outputs (`bool`, `optional`, defaults to `False`):
                Whether to return the outputs of the model.
        Returns:
            `Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`:
                The loss of the model.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean() if self.args.n_gpu > 1 else loss
        return (loss, outputs) if return_outputs else loss
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # if not self.use_reward_data_collator:
        #     warnings.warn(
        #         "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
        #         " if you are using a custom data collator make sure you know what you are doing or"
        #         " implement your own compute_loss method."
        #     )
        
        #### Debugging ####
        # print("\n######## Inputs ########")
        # print(inputs)
        # print(inputs["input_ids_chosen"].shape)
        #### Debugging ####

        # print("\n######## Chosen ########\n")
        # Fine the sub-module that is a LinearLayer_PSLoRA
        for name, module in model.named_modules():
            if isinstance(module, LinearLayer_PSLoRA):
                module.labeler_index = inputs["labeler_index"]

        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        
        # print("\n######## Rejectet ########\n")

        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss



if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ###################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True, 
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token_id is None:
        warnings.warn("pad_token_id is None, setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # TODO: Apply chat template. Currently, we directly concatenate the prompt and the response w/o chat template.
    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        # warnings.warn("No chat template found. Using ChatML as the default template.")
        # model, tokenizer = setup_chat_format(model, tokenizer)
        warnings.warn("No chat template found. Directly concatenate the prompt and the response.")

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    #############################
    # Load and preprocess dataset
    #############################
    raw_datasets = load_dataset(args.dataset_name, args.dataset_subset)
    raw_trainset, raw_testset, worker_dict = reddit_comp_top_N(raw_datasets, args.num_labelers)

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "labeler_index": examples["worker"],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen, padding="longest", truncation=True, max_length=config.max_length)
            tokenized_rejected = tokenizer(rejected, padding="longest", truncation=True, max_length=config.max_length)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        if args.apply_chat_template:
            chosen_fn = conversations_formatting_function(tokenizer, "chosen")
            rejected_fn = conversations_formatting_function(tokenizer, "rejected")
            raw_datasets = raw_datasets.map(
                lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)},
                num_proc=config.dataset_num_proc
            )
            # Tokenize inputs
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=config.dataset_num_proc,
            )
            # Filter out examples that are too long
            raw_datasets = raw_datasets.filter(
                lambda x: len(x["input_ids_chosen"]) <= config.max_length
                and len(x["input_ids_rejected"]) <= config.max_length,
                num_proc=config.dataset_num_proc,
            )
        else:
            raw_trainset = raw_trainset.map(
                lambda x: {"chosen": reddit_prompt_template(x, "chosen"),
                           "rejected": reddit_prompt_template(x, "rejected")},
                num_proc=config.dataset_num_proc
            )
            raw_testset = raw_testset.map(
                lambda x: {"chosen": reddit_prompt_template(x, "chosen"),
                           "rejected": reddit_prompt_template(x, "rejected")},
                num_proc=config.dataset_num_proc
            )
            # Remove unnecessary columns (specific to the Reddit TL;DR dataset)
            raw_trainset = raw_trainset.remove_columns(["info", "summaries", "batch", "split", "extra"])
            raw_testset = raw_testset.remove_columns(["info", "summaries", "batch", "split", "extra"])
            # Tokenize inputs
            raw_trainset = raw_trainset.map(
                preprocess_function,
                batched=True,
                num_proc=config.dataset_num_proc,
            )
            raw_testset = raw_testset.map(
                preprocess_function,
                batched=True,
                num_proc=config.dataset_num_proc,
            )
            # Select the labeler
            if args.selected_labeler == "all" or args.selected_labeler == "personalized":
                pass
            else:
                print(f"Selecting labeler: {args.selected_labeler}")
                raw_trainset = raw_trainset.filter(lambda x: x["worker"] == int(args.selected_labeler))
                raw_testset = raw_testset.filter(lambda x: x["worker"] == int(args.selected_labeler))
            # TODO: Filter out examples that are too long
            # shuffle the dataset
            raw_trainset = raw_trainset.shuffle()
            raw_testset = raw_testset.shuffle()

    if args.apply_chat_template:
        train_dataset = raw_datasets[args.dataset_train_split]
        eval_dataset = raw_datasets[args.dataset_test_split]
    else:
        train_dataset = raw_trainset
        eval_dataset = raw_testset

    # #### Debugging ####
    # train_dataset = train_dataset.select(range(128))
    # eval_dataset = eval_dataset.select(range(32))

    ##########
    # Training
    ##########
    if args.selected_labeler == "personalized":
        print("LoRA Target Modules: ", model_config.lora_target_modules)

        model = convert_linear_layer_to_lora(
            model=model,
            target_modules=model_config.lora_target_modules,
            lora_r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            num_labelers=args.num_labelers)
        
        print("LoRA model")
        print(model)
        
        only_optimize_lora_parameters(model)

        data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=config.max_length)
        
        trainer = PSRewardTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None,
        )
    else:
        # trainer = RewardTrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     args=config,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     peft_config=get_peft_config(model_config),
        # )
        # # Now you can print the trainable parameters
        # trainer.model.print_trainable_parameters()

        print("LoRA Target Modules: ", model_config.lora_target_modules)

        model = convert_linear_layer_to_lora(
            model=model,
            target_modules=model_config.lora_target_modules,
            lora_r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            num_labelers=-1)
        
        print("LoRA model")
        print(model)
        
        only_optimize_lora_parameters(model)

        data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=config.max_length)
        
        trainer = PSRewardTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None,
        )

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(f"Activate Layers: {name}")

    trainer.train()

    ###########################
    # Save model and evaluation
    ###########################
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    # trainer.save_model(config.output_dir)