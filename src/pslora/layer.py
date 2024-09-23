from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft import LoraModel, LoraConfig
from peft.tuners.lora import LoraLayer
from peft.utils import PeftType
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import PSLoraConfig
 

class PSLoraLayer(LoraLayer):
    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.num_labelers = {}
    
    def update_layer(
        self, adapter_name, r, lora_alpha, num_labelers, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # Override the update_layer method to initialize A matrices for each worker
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.num_labelers[adapter_name] = num_labelers
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Parameter(torch.zeros(num_labelers, self.in_features, r))       # Personalized-LoRA Matrix A
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        
        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        # NOTE: Currently only support the default init_lora_weights
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            raise NotImplementedError("Pissa initialization is not supported for Personalized-Shared LoRA")
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            raise NotImplementedError("OLoRA initialization is not supported for Personalized-Shared LoRA")
        elif init_lora_weights == "loftq":
            raise NotImplementedError("LoftQ initialization is not supported for Personalized-Shared LoRA")
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            raise NotImplementedError("DoRA is not supported for Personalized-Shared LoRA")
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name], std=1 / self.r[adapter_name])
            elif init_lora_weights.lower() == "zero":
                nn.init.zeros_(self.lora_A[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class PSLoraLinear(nn.Module, PSLoraLayer):
    # Lora implemented in a dense layer
    # TODO: merge(), unmerge(), and get_delta_weights() methods are not supported yet as lora_A is now a 3D tensor rather than a nn.Linear layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        num_labelers: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        PSLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            num_labelers=num_labelers,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, labeler_index: Union[int, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                # self.unmerge()
                raise NotImplementedError("Merged mode is not supported for Personalized-Shared LoRA")
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            # result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            raise NotImplementedError("Mixed batch forward is not supported for Personalized-Shared LoRA")
        elif self.merged:
            # result = self.base_layer(x, *args, **kwargs)
            raise NotImplementedError("Merged forward is not supported for Personalized-Shared LoRA")
        else:
            result = self.base_layer(x, *args, **kwargs)

            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = x.to(lora_A.dtype)

                # Ensure that labeler_index is a 1D tensor
                if isinstance(labeler_index, int):
                    labeler_index = torch.tensor([labeler_index], device=x.device)
                labeler_index = labeler_index.view(-1)
                
                # Handle potential broadcasting
                if labeler_index.size(0) == 1 and x.size(0) > 1:
                    labeler_index = labeler_index.expand(x.size(0))     # Expand the labeler_index to match the batch size
                
                # Select the appropriate A matrices for each sample in the batch
                batch_size = x.size(0)
                labelers_A = lora_A[labeler_index]      # Shape: (batch_size, in_features, r)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(torch.bmm(dropout(x).unsqueeze(1), labelers_A).squeeze()) * scaling
                else:
                    raise NotImplementedError("DoRA is not supported for Personalized-Shared LoRA")

            result = result.to(torch_result_dtype)
        
        return result

    
def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: PSLoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        raise NotImplementedError("Embedding layer is not supported for Personalized-Shared-LoRA")
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        raise NotImplementedError("Conv2d layer is not supported for Personalized-Shared-LoRA")
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = PSLoraLinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        raise NotImplementedError("Conv1D layer is not supported for Personalized-Shared-LoRA")

    return new_module