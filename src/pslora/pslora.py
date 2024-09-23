"""
Implementation of Personalized-Shared LoRA (PSLoRA) Model, currently only support Linear Layer.
Code adapted from: https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/module/lora.py

Copyright (c) Microsoft Corporation.
SPDX-License-Identifier: Apache-2.0
"""

import math
import warnings
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer_PSLoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    
    # labeler_index = None  # Labelers being activated in the forward pass (per batch)
    
    def __init__(self,
                 weight,
                 lora_r=32,
                 lora_alpha=16,
                 lora_droppout=0,
                 num_labelers=5,
                 bias=None):
        super(LinearLayer_PSLoRA, self).__init__()
        self.weight = weight
        self.bias = bias
        self.num_labelers = num_labelers
        self.labeler_index = None  # Labelers being activated in the forward pass (per batch)

        if lora_r <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            out_features, in_features = weight.ds_shape
        except:
            out_features, in_features = weight.shape
        
        if num_labelers <= 0:
            self.lora_A = nn.Linear(in_features, lora_r, bias=False)
        else:
            self.lora_A = nn.Parameter(torch.zeros(num_labelers, in_features, lora_r))
        # self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
        self.lora_B = nn.Linear(lora_r, out_features, bias=False)
        self.lora_scaling = lora_alpha / lora_r

        if lora_droppout > 0.0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def fuse_lora_weight(self):
        # FIXME: Add support for PSLoRa when using additional dimension of labelers
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_B.t(), self.lora_A.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        # FIXME: Add support for PSLoRa when using additional dimension of labelers
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_B.t(), self.lora_A.t())
        self.fuse_lora = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # print("######## LinearLayer_PSLoRA.forward ########")
        # print("labeler_index:", self.labeler_index)
        # print("input shape:", input.shape)

        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            result = F.linear(input, self.weight, self.bias)
            torch_result_dtype = result.dtype
            input = input.to(self.lora_A.dtype)
            
            if self.num_labelers <= 0:
                # Ensure that labeler_index is a 1D tensor
                if isinstance(self.labeler_index, int):
                    self.labeler_index = torch.tensor([self.labeler_index], device=input.device)
                self.labeler_index = self.labeler_index.view(-1)
                # Handle potential broadcasting
                if self.labeler_index.size(0) == 1 and input.size(0) > 1:
                    self.labeler_index = self.labeler_index.expand(input.size(0))     # Expand the labeler_index to match the batch size
                # Select the appropriate A matrices for each sample in the batch
                batch_size = input.size(0)
                labelers_A = self.lora_A[self.labeler_index]

                # input: (batch_size, seq_length, in_features)
                # labelers_A: (batch_size, in_features, r)
                # lora_A: (batch_size, seq_length, r) (after torch.bmm)
                # lora_B: nn.Linear(r, out_features)
                # result: (batch_size, seq_length, out_features)
                
                result = result + self.lora_B(torch.bmm(self.lora_dropout(input), labelers_A)) * self.lora_scaling
            else:
                result = result + self.lora_B(self.lora_A(self.lora_dropout(input))) * self.lora_scaling

            result = result.to(torch_result_dtype)

            return result


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def convert_linear_layer_to_lora(model,
                                 target_modules,              # target modules to convert to LoRA
                                 lora_r=0,
                                 lora_alpha=1,
                                 lora_dropout=0,
                                 num_labelers=5):
    replace_name = []
    for name, module in model.named_modules():
        # what is part_module_name is a list of strings?
        # if isinstance(module, nn.Linear) and part_module_name in name:
        if isinstance(module, nn.Linear) and any([key in name for key in target_modules]):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_PSLoRA(
            module.weight, lora_r, lora_alpha, lora_dropout, num_labelers,
            module.bias).to(module.weight.dtype).to(module.weight.device)
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model, force_optimize_params=['score',]):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif any([key in name for key in force_optimize_params]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model