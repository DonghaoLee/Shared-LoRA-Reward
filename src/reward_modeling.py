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
from safetensors import safe_open
from safetensors.torch import load_file
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    HfArgumentParser,
    set_seed,
)
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_safetensors_available,
    logging,
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

from data_loader import get_dataset
from pslora import (
    LinearLayer_PSLoRA,
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    convert_lora_checkpoint_to_plas
)

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


tqdm.pandas()
warnings.simplefilter("once")
TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.get_logger(__name__)


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
    local_trainset_path: Optional[str] = field(default=None, metadata={"help": "Path to the local training dataset"})
    local_testset_path: Optional[str] = field(default=None, metadata={"help": "Path to the local test dataset"})
    local_validset_path: Optional[str] = field(default=None, metadata={"help": "Path to the local validation dataset"})
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
    lora_type: str = field(
        default="lora",
        metadata={"help": "The type of LoRA to use. Options: pslora, kernel, svd"},
    )
    lora_personalize_strategy: str = field(
        default="personalized_A",
        metadata={"help": "Which layer should be expanded to personalized structure. Options: personalized_A, personalized_B."},
    )
    lora_optimize_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The modules to optimize for LoRA. Default is all modules."},
    )
    eval_mode: bool = field(
        default=False,
        metadata={"help": "Whether to train or evaluate the model. If True, only evaluate the model."},
    )
    checkpoint_paths: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Path of all the safetensor files.")},
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


class PreTrainingEvalCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of training.
        """
        # This will run the evaluation before training begins
        control.should_evaluate = True


class PSRewardTrainer(RewardTrainer):
    def __init__(self, *args, layers_to_save, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers_to_save = layers_to_save
    
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

        # Find the sub-module that is a LinearLayer_PSLoRA
        for name, module in model.named_modules():
            if isinstance(module, LinearLayer_PSLoRA):
                module.labeler_index = inputs["labeler_index"]

        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
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

    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        if isinstance(self.eval_dataset, dict):
            # TODO: Implement visualization for fine-grained evaluation subsets
            pass
        else:
            super().visualize_samples(num_print_samples)

    def get_partial_state_dict(self, state_dict=None):
        """
        Construct the partial state dict to save only the layers that have gradients.
        Specifically, we only save the LoRA layers and the last score layer in this task.

        Args:
            state_dict (`Optional[Dict[str, torch.Tensor]]`, defaults to `None`):
                The state dict of the model.
        """
        if self.layers_to_save is not None:
            # self.layers_to_save is a list of layer name patterns (e.g., ["lora", "score"])
            state_dict = {k: v for k, v in state_dict.items() if any(layer_name in k for layer_name in self.layers_to_save)}
        return state_dict
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()
            state_dict = self.get_partial_state_dict(state_dict)

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            state_dict = self.get_partial_state_dict(state_dict)
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))



if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Set seed for reproducibility
    set_seed(config.seed)

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
    train_dataset, eval_dataset = get_dataset(args, config, tokenizer)

    ######################
    # For Debugging
    ######################
    # # sample a small subset of the eval_dataset
    # eval_dataset = eval_dataset.select(range(1280))
    # print("Eval Dataset: ", eval_dataset)

    ######################
    # Model Initialization
    ######################
    if args.selected_labeler == "personalized":
        """
        Personalized Reward Model: Train a partial separate/shared reward model for each labeler.
        """
        print("LoRA Target Modules: ", model_config.lora_target_modules)
        print("LoRA Type: ", args.lora_type)
        model = convert_linear_layer_to_lora(
            model=model,
            target_modules=model_config.lora_target_modules,
            lora_r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            num_labelers=args.num_labelers,
            lora_type=args.lora_type,
            personalize_strategy=args.lora_personalize_strategy)
        print("LoRA model")
        print(model)
    else:
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
    
    if args.checkpoint_paths is not None:
        # Load the model from the output directory
        def load_multiple_safetensor_checkpoints(model, checkpoint_paths):
            combined_state_dict = {}
            
            for checkpoint_path in checkpoint_paths:
                print(f"Loading checkpoint from {checkpoint_path}")
                state_dict = load_file(checkpoint_path)
                combined_state_dict.update(state_dict)
            
            if args.selected_labeler == "personalized" and not args.eval_mode:
                # Modify the pre-trained LoRA parameters to fit the personalized reward model
                print("Converting LoRA parameters to PISA parameters")
                combined_state_dict = convert_lora_checkpoint_to_plas(combined_state_dict, args.num_labelers, args.lora_personalize_strategy)

            # Load the combined state dict into your model
            model_state_dict = model.state_dict()
            print("Load the following modules:", combined_state_dict.keys())
            model_state_dict.update(combined_state_dict)
            model.load_state_dict(model_state_dict)
            
            return model
        model = load_multiple_safetensor_checkpoints(model, args.checkpoint_paths)
        
        if args.eval_mode:
            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=config.max_length)
            trainer = PSRewardTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                args=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=None,
                layers_to_save=["lora", "score"],
            )
            trainer.train()             # Bypass solution. Directly calling evaluate() may raise error that the model and the data are not on the same device.
            metrics = trainer.evaluate()
            # Compute the average accuracy and loss of all fine-grained evaluation subsets
            warnings.warn("[Merge Subset Evaluation] Fine-grained evaluation subsets.")
            if isinstance(eval_dataset, dict):
                # Get the name of each fine-grained evaluation subset
                eval_subset_names = list(eval_dataset.keys())
                # Get the number of examples in each fine-grained evaluation subset
                eval_subset_sizes = [len(eval_dataset[eval_subset_name]) for eval_subset_name in eval_subset_names]
                # Get the accuracy of each fine-grained evaluation subset from the metrics
                eval_subset_accuracies = [metrics[f"eval_{eval_subset_name}_accuracy"] for eval_subset_name in eval_subset_names]
                # Get the loss of each fine-grained evaluation subset from the metrics
                eval_subset_losses = [metrics[f"eval_{eval_subset_name}_loss"] for eval_subset_name in eval_subset_names]
                # Get the average accuracy and loss of all fine-grained evaluation subsets, weighted by the number of examples
                metrics["eval_all_accuracy"] = sum([size * accuracy for size, accuracy in zip(eval_subset_sizes, eval_subset_accuracies)]) / sum(eval_subset_sizes)
                metrics["eval_all_loss"] = sum([size * loss for size, loss in zip(eval_subset_sizes, eval_subset_losses)]) / sum(eval_subset_sizes)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
            print(metrics)
            exit()

    ##########
    # Training
    ##########
    if args.selected_labeler == "personalized":
        if args.lora_optimize_modules is not None:
            only_optimize_lora_parameters(model, args.lora_optimize_modules)
        else:
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
            layers_to_save=["lora", "score"],
            callbacks=[PreTrainingEvalCallback],
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
        if args.lora_optimize_modules is not None:
            only_optimize_lora_parameters(model, args.lora_optimize_modules)
        else:
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
            layers_to_save=["lora", "score"],
            callbacks=[PreTrainingEvalCallback],
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
    if config.save_strategy != "epoch":
        trainer.save_model(config.output_dir)     # Save the model. If we save checkpoint per epoch, we don't need to save the model here.
