# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower type definitions."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = list[NDArray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4

@dataclass
class Parameters:
    """Model parameters."""

    tensors: list[bytes]
    tensor_type: str

@dataclass
class Status:
    """Client status."""

    code: Code
    message: str



@dataclass
class FitIns:
    """Fit instructions for a client."""
    parameters: Parameters
    ckks_blocks : Union[list[bytes] ,None] 
    config: dict[str, Scalar]
    enc_lines: Union[list,None]  


@dataclass
class FitRes:
    """Fit response from a client."""
    status: Status
    parameters: Parameters
    ckks_blocks : Union[list[bytes] ,None] 
    num_examples: int
    metrics: dict[str, Scalar]

@dataclass
class FitResNeo:
    he_budget: int
    Sens_layer: dict[str,dict[int,float]]