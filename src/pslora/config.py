from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType
from peft import LoraConfig


@dataclass
class PSLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`PSLoraModel`].
    We inherit from the [`LoraConfig`] class and add the `num_labelers` attribute to store the number of labelers in the personalized-shared LORA model.

    Args:
        num_labelers (`int`):
            The number of labelers for personalization.
    """
    num_labelers: int = field(default=5, metadata={"help": "Number of labelers for personalization."})