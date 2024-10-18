from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional, Union, Any

import torch
from torch import nn
from tqdm import tqdm

from peft import LoraModel, LoraConfig
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties

from .config import PSLoraConfig
from .layer import PSLoraLayer, PSLoraLinear, dispatch_default


class PSLoraModel(LoraModel):
    """
    Creates Personalized-Shared Low Rank Adapter (LoRA) model from a pretrained transformer model.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`PSLoraConfig`]): The configuration of the PSLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The PSLora model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`PSLoraConfig`]): The configuration of the PSLora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
    
    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "num_labelers": lora_config.num_labelers,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, PSLoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                num_labelers=lora_config.num_labelers,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = dispatch_default(target, adapter_name, lora_config=lora_config, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    # FIXME:
    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)