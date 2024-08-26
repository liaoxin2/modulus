# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import paddle
import paddle.nn as nn

import modulus  # noqa: F401 for docs

Tensor = paddle.Tensor


class Identity(nn.Layer):
    """Identity activation function

    Dummy function for removing activations from a model

    Example
    -------
    >>> idnt_func = modulus.models.layers.Identity()
    >>> input = paddle.randn(2, 2)
    >>> output = idnt_func(input)
    >>> paddle.allclose(input, output)
    True
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Stan(nn.Layer):
    """Self-scalable Tanh (Stan) for 1D Tensors

    Parameters
    ----------
    out_features : int, optional
        Number of features, by default 1

    Note
    ----
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.

    Example
    -------
    >>> stan_func = modulus.models.layers.Stan(out_features=1)
    >>> input = paddle.Tensor([[0],[1],[2]])
    >>> stan_func(input)
    tensor([[0.0000],
            [1.5232],
            [2.8921]], grad_fn=<MulBackward0>)
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.beta = nn.Parameter(paddle.ones(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return paddle.tanh(x) * (1.0 + self.beta * x)


class SquarePlus(nn.Layer):
    """Squareplus activation

    Note
    ----
    Reference: arXiv preprint arXiv:2112.11687

    Example
    -------
    >>> sqr_func = modulus.models.layers.SquarePlus()
    >>> input = paddle.Tensor([[1,2],[3,4]])
    >>> sqr_func(input)
    tensor([[1.6180, 2.4142],
            [3.3028, 4.2361]])
    """

    def __init__(self):
        super().__init__()
        self.b = 4

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + paddle.sqrt(x * x + self.b))


class CappedLeakyReLU(paddle.nn.Layer):
    """
    Implements a ReLU with capped maximum value.

    Example
    -------
    >>> capped_leakyReLU_func = modulus.models.layers.CappedLeakyReLU()
    >>> input = paddle.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_leakyReLU_func(input)
    tensor([[-0.0200, -0.0100],
            [ 0.0000,  1.0000],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        **kwargs:
             Keyword arguments to be passed to the `paddle.nn.LeakyReLU` function
        """
        super().__init__()
        self.add_module("leaky_relu", paddle.nn.LeakyReLU(**kwargs))
        self.register_buffer("cap", paddle.tensor(cap_value, dtype=paddle.float32))

    def forward(self, inputs):
        x = self.leaky_relu(inputs)
        x = paddle.clamp(x, max=self.cap)
        return x


class CappedGELU(paddle.nn.Layer):
    """
    Implements a GELU with capped maximum value.

    Example
    -------
    >>> capped_gelu_func = modulus.models.layers.CappedGELU()
    >>> input = paddle.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_gelu_func(input)
    tensor([[-0.0455, -0.1587],
            [ 0.0000,  0.8413],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Parameters:
        ----------
        cap_value: float, optional
            Maximum that values will be capped at
        **kwargs:
             Keyword arguments to be passed to the `paddle.nn.GELU` function
        """

        super().__init__()
        self.add_module("gelu", paddle.nn.GELU(**kwargs))
        self.register_buffer("cap", paddle.tensor(cap_value, dtype=paddle.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = paddle.clamp(x, max=self.cap)
        return x


# Dictionary of activation functions
ACT2FN = {
    "relu": nn.ReLU,
    "leaky_relu": (nn.LeakyReLU, {"negative_slope": 0.1}),
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.Silu,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": (nn.Threshold, {"threshold": 1.0, "value": 1.0}),
    "hardtanh": nn.Hardtanh,
    "identity": Identity,
    "stan": Stan,
    "squareplus": SquarePlus,
    "cappek_leaky_relu": CappedLeakyReLU,
    "capped_gelu": CappedGELU,
}


def get_activation(activation: str) -> nn.Layer:
    """Returns an activation function given a string

    Parameters
    ----------
    activation : str
        String identifier for the desired activation function

    Returns
    -------
    Activation function

    Raises
    ------
    KeyError
        If the specified activation function is not found in the dictionary
    """
    try:
        activation = activation.lower()
        module = ACT2FN[activation]
        if isinstance(module, tuple):
            return module[0](**module[1])
        else:
            return module()
    except KeyError:
        raise KeyError(
            f"Activation function {activation} not found. Available options are: {list(ACT2FN.keys())}"
        )
