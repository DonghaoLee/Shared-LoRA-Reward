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
                                 target_modules,
                                 lora_r=0,
                                 lora_alpha=1,
                                 lora_dropout=0,
                                 num_labelers=5,
                                 lora_type='lora',
                                 personalize_strategy='personalized_A'):
    """
    Convert the linear layer within the pre-trained model to LoRA layer.
    Args:
        model (`torch.nn.Module`)
            The model to convert the linear layer to LoRA layer.
        target_modules (`List[str]`)
            The list of module names to match the module names for which LoRA should be applied.
        lora_r (`int`)
            The reduced dimension of LoRA.
        lora_alpha (`int`)
            The scaling factor of LoRA.
        lora_dropout (`float`)
            The dropout rate of LoRA.
        num_labelers (`int`)
            The number of labelers to consider.
        lora_type (`str`)
            The architecture of LoRA to use. Currently supports 'lora', 'kernel', and 'svd'.
        personalize_strategy (`str`)
            The strategy to use for personalization. Currently supports 'personalized_A' and 'personalized_B'.
    """
    replace_name = []
    for name, module in model.named_modules():
        # target_modules is a list of name patterns to match the module names for which LoRA should be applied. Example: ['q_proj', 'v_proj']
        if isinstance(module, nn.Linear) and any([key in name for key in target_modules]):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_PSLoRA(
            module.weight, lora_r, lora_alpha, lora_dropout,
            num_labelers, lora_type, personalize_strategy,
            module.bias).to(module.weight.dtype).to(module.weight.device)
        recursive_setattr(model, name, tmp)
    return model


def convert_lora_checkpoint_to_plas(checkpoint, num_labelers=5, personalize_strategy='personalized_A'):
    """
    Convert the LoRA checkpoint to PSLoRA state dict. This function is used when we want to initialize the PSLoRA model with a pre-trained LoRA model (with gloabl A and B matrices).
    Args:
        checkpoint (`Dict[str, torch.Tensor]`)
            The checkpoint of the LoRA model.
        num_labelers (`int`)
            The number of labelers to consider.
        personalize_strategy (`str`)
            The strategy to use for personalization. Currently supports 'personalized_A' and 'personalized_B'.
    """
    replace_name = []
    # Get all the layer names from the checkpoint
    names = list(checkpoint.keys())
    for name in names:
        if personalize_strategy == 'personlized_A':
            # If the strategy is 'personalized_A', then we need to copy the A matrix for each labeler
            if 'lora_A' in name:
                replace_name.append(name)
                plas_lora_A = torch.transpose(checkpoint[name], 0, 1).unsqueeze(0).repeat(num_labelers, 1, 1)
                # Replace the name of the layer to match the PSLoRA's architecture which uses nn.Parameter instead of nn.Linear
                plas_lora_A_name = name.replace('lora_A.weight', 'lora_A')
                checkpoint[plas_lora_A_name] = plas_lora_A
                del checkpoint[name]
        else:
            # If the strategy is 'personalized_B', then we need to copy the B matrix for each labeler
            if 'lora_B' in name:
                replace_name.append(name)
                plas_lora_B = torch.transpose(checkpoint[name], 0, 1).unsqueeze(0).repeat(num_labelers, 1, 1)
                # Replace the name of the layer to match the PSLoRA's architecture which uses nn.Parameter instead of nn.Linear
                plas_lora_B_name = name.replace('lora_B.weight', 'lora_B')
                checkpoint[plas_lora_B_name] = plas_lora_B
                del checkpoint[name]
    return checkpoint


def only_optimize_lora_parameters(model, lora_modules=['lora_A', 'lora_B', 'lora_kernel', 'lora_singular'], force_optimize_params=['score',]):
    """
    Correctly set the `requires_grad` flag to only optimize the LoRA parameters in the model.
    Args:
        model (`torch.nn.Module`)
            The model to optimize the LoRA parameters.
        force_optimize_params (`List[str]`)
            The list of parameter names to force optimize. Example: ['score'] (i.e., the final score layer for sequence classification).
    """
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        # If the parameter name contains 'lora_*', then set `requires_grad` to True
        # if "lora_A" in name or "lora_B" in name or "lora_kernel" in name or "lora_singular" in name:
        if any([key in name for key in lora_modules]):
            param.requires_grad = True
        elif any([key in name for key in force_optimize_params]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


class LinearLayer_PSLoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    
    def __init__(self,
                 weight,
                 lora_r=32,
                 lora_alpha=16,
                 lora_droppout=0,
                 num_labelers=5,
                 lora_type='lora',
                 personalize_strategy='personalized_A',
                 bias=None):
        super(LinearLayer_PSLoRA, self).__init__()
        self.weight = weight
        self.bias = bias
        self.num_labelers = num_labelers
        self.lora_type = lora_type
        self.labeler_index = None                           # Labelers being activated in the forward pass (per batch)
        self.debug_mode = False
        self.personalize_strategy = personalize_strategy    # 'personalized_A' or 'personalized_B'

        if lora_r <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            out_features, in_features = weight.ds_shape
        except:
            out_features, in_features = weight.shape
        
        if self.personalize_strategy == 'personalized_A':
            if num_labelers <= 0:
                # Train all the labelers together
                if self.lora_type == 'lora':
                    self.lora_A = nn.Linear(in_features, lora_r, bias=False)
                elif self.lora_type == 'kernel':
                    self.lora_A = nn.Linear(in_features, lora_r, bias=False)
                    self.lora_kernel = nn.Linear(lora_r, lora_r, bias=False)
                elif self.lora_type == 'svd':
                    self.lora_A = nn.Linear(in_features, lora_r, bias=False)
                    self.lora_singular = nn.Parameter(torch.ones(lora_r))
                else:
                    raise ValueError(f"Invalid LoRA type: {self.lora_type}")

            else:
                # Train each labeler separately with a different A matrix but a shared B matrix
                if self.lora_type == 'lora':
                    self.lora_A = nn.Parameter(torch.zeros(num_labelers, in_features, lora_r))
                elif self.lora_type == 'kernel':
                    self.lora_A = nn.Parameter(torch.zeros(in_features, lora_r))
                    self.lora_kernel = nn.Parameter(torch.zeros(num_labelers, lora_r, lora_r))
                elif self.lora_type == 'svd':
                    self.lora_A = nn.Parameter(torch.zeros(in_features, lora_r))
                    self.lora_singular = nn.Parameter(torch.ones(num_labelers, lora_r))
                else:
                    raise ValueError(f"Invalid LoRA type: {self.lora_type}")
            
            # self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
            self.lora_B = nn.Linear(lora_r, out_features, bias=False)
        else:
            # self.lora_A = nn.Parameter(torch.zeros(in_features, lora_r))
            self.lora_A = nn.Linear(in_features, lora_r, bias=False)

            if num_labelers <= 0:
                # Train all the labelers together
                if self.lora_type == 'lora':
                    self.lora_B = nn.Linear(lora_r, out_features, bias=False)
                elif self.lora_type == 'kernel':
                    self.lora_B = nn.Linear(lora_r, out_features, bias=False)
                    self.lora_kernel = nn.Linear(lora_r, lora_r, bias=False)
                elif self.lora_type == 'svd':
                    self.lora_B = nn.Linear(lora_r, out_features, bias=False)
                    self.lora_singular = nn.Parameter(torch.ones(lora_r))
                else:
                    raise ValueError(f"Invalid LoRA type: {self.lora_type}")

            else:
                # Train each labeler separately with a different B matrix but a shared A matrix
                if self.lora_type == 'lora':
                    self.lora_B = nn.Parameter(torch.zeros(num_labelers, lora_r, out_features))
                elif self.lora_type == 'kernel':
                    self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
                    self.lora_kernel = nn.Parameter(torch.zeros(num_labelers, lora_r, lora_r))
                elif self.lora_type == 'svd':
                    self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
                    self.lora_singular = nn.Parameter(torch.ones(num_labelers, lora_r))
                else:
                    raise ValueError(f"Invalid LoRA type: {self.lora_type}")

        self.lora_scaling = lora_alpha / lora_r

        if lora_droppout > 0.0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
        if self.personalize_strategy == 'personalized_A':
            if self.num_labelers <= 0:
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                if self.lora_type == 'kernel':
                    nn.init.kaiming_uniform_(self.lora_kernel.weight, a=math.sqrt(5))
                elif self.lora_type == 'svd':
                    # singular matrix is initialized to a diagonal matrix with ones
                    # nn.init.kaiming_uniform_(self.lora_singular, a=math.sqrt(5))
                    pass
            else:
                # TODO: When initializing the A matrices for each labeler, we have the following options:
                # 1. Initialize the 3-dimensional tensor using the `kaiming_uniform_()` all at once
                # 2. Initialize the 3-dimensional tensor using the `kaiming_uniform_()` one by one for each labeler on the first dimension
                # 3. Initialize a 2-dimensional tensor using the `kaiming_uniform_()` and then expand it to the 3-dimensional tensor so that all labelers' A matrices are the same
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                if self.lora_type == 'kernel':
                    nn.init.kaiming_uniform_(self.lora_kernel, a=math.sqrt(5))
                elif self.lora_type == 'svd':
                    # singular matrix is initialized to a diagonal matrix with ones
                    # nn.init.kaiming_uniform_(self.lora_singular, a=math.sqrt(5))
                    pass
            nn.init.zeros_(self.lora_B.weight)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            if self.num_labelers <= 0:
                nn.init.zeros_(self.lora_B.weight)
                if self.lora_type == 'kernel':
                    # kernel matrix is initialized to a diagonal matrix with ones
                    # nn.init.zeros_(self.lora_kernel.weight)
                    nn.init.kaiming_uniform_(self.lora_kernel.weight, a=math.sqrt(5))
                elif self.lora_type == 'svd':
                    # singular matrix is initialized to a diagonal matrix with ones
                    # nn.init.zeros_(self.lora_singular)
                    pass
            else:
                nn.init.zeros_(self.lora_B)
                if self.lora_type == 'kernel':
                    # kernel matrix is initialized to a diagonal matrix with ones
                    # nn.init.zeros_(self.lora_kernel)
                    nn.init.kaiming_uniform_(self.lora_kernel, a=math.sqrt(5))
                elif self.lora_type == 'svd':
                    # singular matrix is initialized to a diagonal matrix with ones
                    # nn.init.zeros_(self.lora_singular)
                    pass

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
        
        if self.debug_mode:
            print("######## LinearLayer_PSLoRA.forward ########")
            print(f"[Device{input.device}] labeler_index: {self.labeler_index}")
            # print("input shape:", input.shape)

        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            result = F.linear(input, self.weight, self.bias)
            torch_result_dtype = result.dtype
            if self.personalize_strategy == 'personalized_A':
                input = input.to(self.lora_B.weight.dtype)
            else:
                input = input.to(self.lora_A.weight.dtype)
            
            if self.num_labelers <= 0:
                if self.lora_type == 'lora':
                    result = result + self.lora_B(self.lora_A(self.lora_dropout(input))) * self.lora_scaling
                elif self.lora_type == 'kernel':
                    result = result + self.lora_B(self.lora_kernel(self.lora_A(self.lora_dropout(input)))) * self.lora_scaling
                elif self.lora_type == 'svd':
                    result = result + self.lora_B(self.lora_A(self.lora_dropout(input)) @ torch.diag(self.lora_singular)) * self.lora_scaling
            else:
                # Ensure that labeler_index is a 1D tensor
                if self.debug_mode:
                    print(f"[Device{input.device}] After getting base model's result, labeler_index: {self.labeler_index}")
                
                if isinstance(self.labeler_index, int):
                    self.labeler_index = torch.tensor([self.labeler_index], device=input.device)
                self.labeler_index = self.labeler_index.view(-1)
                
                if self.debug_mode:
                    print(f"[Device{input.device}] After view(-1), labeler_index: {self.labeler_index}")
                
                # Handle potential broadcasting
                if self.labeler_index.size(0) == 1 and input.size(0) > 1:
                    self.labeler_index = self.labeler_index.expand(input.size(0))     # Expand the labeler_index to match the batch size
                
                if self.debug_mode:
                    print(f"[Device{input.device}] After broadcasting, labeler_index: {self.labeler_index}")
                
                if self.lora_type == 'lora':
                    if self.personalize_strategy == 'personalized_A':
                        # Select the appropriate A matrices for each sample in the batch
                        labelers_A = self.lora_A[self.labeler_index]

                        # input: (batch_size, seq_length, in_features)
                        # labelers_A: (batch_size, in_features, r)
                        # lora_A: (batch_size, seq_length, r) (after torch.bmm)
                        # lora_B: nn.Linear(r, out_features)
                        # result: (batch_size, seq_length, out_features)
                        result = result + self.lora_B(torch.bmm(self.lora_dropout(input), labelers_A)) * self.lora_scaling
                    else:
                        # Select the appropriate B matrices for each sample in the batch
                        labelers_B = self.lora_B[self.labeler_index]
                        result = result + torch.bmm(self.lora_A(self.lora_dropout(input)), labelers_B) * self.lora_scaling

                elif self.lora_type == 'kernel':
                    if self.personalize_strategy == 'personalized_A':
                        # Select the appropriate kernel matrices for each sample in the batch
                        labelers_kernel = self.lora_kernel[self.labeler_index]
                        # input: (batch_size, seq_length, in_features)
                        # lora_A: nn.Parameter(in_features, r)
                        # labelers_kernel: (batch_size, r, r)
                        # lora_B: nn.Linear(r, out_features)
                        result = result + self.lora_B(torch.bmm(self.lora_dropout(input) @ self.lora_A, labelers_kernel)) * self.lora_scaling
                    else:
                        # Select the appropriate kernel matrices for each sample in the batch
                        labelers_kernel = self.lora_kernel[self.labeler_index]
                        result = result + (torch.bmm(self.lora_A(self.lora_dropout(input)), labelers_kernel) @ self.lora_B) * self.lora_scaling

                elif self.lora_type == 'svd':
                    if self.personalize_strategy == 'personalized_A':
                        # Select the appropriate singular values for each sample in the batch
                        labelers_singular = torch.diag_embed(self.lora_singular[self.labeler_index])
                        # input: (batch_size, seq_length, in_features)
                        # lora_A: nn.Parameter(in_features, r)
                        # labelers_singular: (batch_size, r, r), torch.diag_embed is used to create a batch of diagonal matrices
                        # lora_B: nn.Linear(r, out_features)
                        result = result + self.lora_B(torch.bmm(self.lora_dropout(input) @ self.lora_A, labelers_singular)) * self.lora_scaling
                    else:
                        # Select the appropriate singular values for each sample in the batch
                        labelers_singular = torch.diag_embed(self.lora_singular[self.labeler_index])
                        result = result + (torch.bmm(self.lora_A(self.lora_dropout(input)), labelers_singular) @ self.lora_B) * self.lora_scaling


            result = result.to(torch_result_dtype)

            return result
