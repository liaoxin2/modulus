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

from typing import List, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor


class SpectralConv1d(nn.Layer):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1, 2],
            default_initializer=nn.initializer.Assign(
                default_initializer=nn.initializer.Assign(
                    paddle.empty([in_channels, out_channels, self.modes1, 2])
                )
            ),
        )
        self.reset_parameters()

    def compl_mul1d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = paddle.as_complex(weights)
        return paddle.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            bsize,
            self.out_channels,
            x.shape[-1] // 2 + 1,
            device=x.place,
            dtype=paddle.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = paddle.fft.irfft(out_ft, n=x.shape[-1])
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Layer):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty([in_channels, out_channels, self.modes1, self.modes2, 2])
            ),
        )
        self.weights2 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty([in_channels, out_channels, self.modes1, self.modes2, 2])
            ),
        )
        self.reset_parameters()

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = paddle.as_complex(weights)
        return paddle.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            dtype=paddle.cfloat,
            device=x.place,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * paddle.rand(self.weights2.data.shape)


class SpectralConv3d(nn.Layer):
    """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights2 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights3 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.weights4 = self.create_parameter(
            [in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        2,
                    ]
                )
            ),
        )
        self.reset_parameters()

    def compl_mul3d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = paddle.as_complex(weights)
        return paddle.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axis=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.shape[-3],
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            dtype=paddle.cfloat,
            device=x.place,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * paddle.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * paddle.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * paddle.rand(self.weights4.data.shape)


class SpectralConv4d(nn.Layer):
    """4D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights2 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights3 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights4 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights5 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights6 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights7 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.weights8 = self.create_parameter(
            [
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            ],
            default_initializer=nn.initializer.Assign(
                paddle.empty(
                    [
                        in_channels,
                        out_channels,
                        self.modes1,
                        self.modes2,
                        self.modes3,
                        self.modes4,
                        2,
                    ]
                )
            ),
        )
        self.reset_parameters()

    def compl_mul4d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = paddle.as_complex(weights)
        return paddle.einsum("bixyzt,ioxyzt->boxyzt", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axis=[-4, -3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.shape[-4],
            x.shape[-3],
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            dtype=paddle.cfloat,
            device=x.place,
        )

        # print(f'mod: shape[x:]{x_ft.shape[)}] out: {out_ft.shape[)}])
        # print(f'mod: x_ft[weight4]: {x_ft[:, :, self.modes1 :, self.modes2 :, : -self.modes3, :self.modes4].shape[)}]weight4: {self.weights4.shape[)}])

        out_ft[
            :, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4],
            self.weights1,
        )
        out_ft[
            :, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4],
            self.weights2,
        )
        out_ft[
            :, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4],
            self.weights3,
        )
        out_ft[
            :, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4],
            self.weights4,
        )
        out_ft[
            :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4],
            self.weights5,
        )
        out_ft[
            :, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4],
            self.weights6,
        )
        out_ft[
            :, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4],
            self.weights7,
        )
        out_ft[
            :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4],
            self.weights8,
        )

        # Return to physical space
        x = paddle.fft.irfftn(
            out_ft, s=(x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        )
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * paddle.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * paddle.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * paddle.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * paddle.rand(self.weights4.data.shape)
        self.weights5.data = self.scale * paddle.rand(self.weights5.data.shape)
        self.weights6.data = self.scale * paddle.rand(self.weights6.data.shape)
        self.weights7.data = self.scale * paddle.rand(self.weights7.data.shape)
        self.weights8.data = self.scale * paddle.rand(self.weights8.data.shape)


# ==========================================
# Utils for PINO exact gradients
# ==========================================


def fourier_derivatives(x: Tensor, ell: List[float]) -> Tuple[Tensor, Tensor]:
    """
    Fourier derivative function for PINO
    """

    # check that input shape maches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.place

    # compute fourier transform
    x_h = paddle.fft.fftn(x, axis=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            paddle.concat(
                (
                    paddle.arange(start=0, end=nx // 2, step=1, device=device),
                    paddle.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = paddle.complex(
        paddle.tensor([0.0], device=device), paddle.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * x_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]
    wxx_h = [
        j * k_x_i * wx_h_i * (2 * pi / ell[i])
        for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, k_x))
    ]

    # inverse fourier transform out
    wx = paddle.concat(
        [
            paddle.fft.ifftn(wx_h_i, axis=list(range(2, dim + 2))).real
            for wx_h_i in wx_h
        ],
        axis=1,
    )
    wxx = paddle.concat(
        [
            paddle.fft.ifftn(wxx_h_i, axis=list(range(2, dim + 2))).real
            for wxx_h_i in wxx_h
        ],
        axis=1,
    )
    return (wx, wxx)


@paddle.jit.ignore
def calc_latent_derivatives(
    x: Tensor, domain_length: List[int] = 2
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Compute first and second order derivatives of latent variables
    """

    dim = len(x.shape) - 2
    # Compute derivatives of latent variables via fourier methods
    # Padd domain by factor of 2 for non-periodic domains
    padd = [(i - 1) // 2 for i in list(x.shape[2:])]
    # Scale domain length by padding amount
    domain_length = [
        domain_length[i] * (2 * padd[i] + x.shape[i + 2]) / x.shape[i + 2]
        for i in range(dim)
    ]
    padding = padd + padd
    x_p = F.pad(x, padding, mode="replicate")
    dx, ddx = fourier_derivatives(x_p, domain_length)
    # Trim padded domain
    if len(x.shape) == 3:
        dx = dx[..., padd[0] : -padd[0]]
        ddx = ddx[..., padd[0] : -padd[0]]
        dx_list = paddle.split(dx, x.shape[1], axis=1)
        ddx_list = paddle.split(ddx, x.shape[1], axis=1)
    elif len(x.shape) == 4:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
        dx_list = paddle.split(dx, x.shape[1], axis=1)
        ddx_list = paddle.split(ddx, x.shape[1], axis=1)
    else:
        dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
        dx_list = paddle.split(dx, x.shape[1], axis=1)
        ddx_list = paddle.split(ddx, x.shape[1], axis=1)

    return dx_list, ddx_list


def first_order_pino_grads(
    u: Tensor,
    ux: List[Tensor],
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:  # pragma: no cover
    """
    Compute first order derivatives of output variables
    """

    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(-1).unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1).unsqueeze(-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / paddle.cosh(u_hidden) ** 2

    # compute diff(f(g))
    diff_fg = paddle.einsum(
        "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
        weights_1,
        diff_tanh,
        weights_2,
    )

    # compute diff(f(g)) * diff(g)
    vx = [
        paddle.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in ux
    ]
    vx = [paddle.unsqueeze(w, axis=1) for w in vx]

    return vx


def second_order_pino_grads(
    u: Tensor,
    ux: Tensor,
    uxx: Tensor,
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:  # pragma: no cover
    """
    Compute second order derivatives of output variables
    """

    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1)
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(-1).unsqueeze(-1)
        weights_2 = weights_2.unsqueeze(-1).unsqueeze(-1)
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / paddle.cosh(u_hidden) ** 2

    # compute diff(f(g))
    diff_fg = paddle.einsum(
        "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
        weights_1,
        diff_tanh,
        weights_2,
    )

    # compute diagonal of hessian
    # double derivative of hidden layer
    diff_diff_tanh = -2 * diff_tanh * paddle.tanh(u_hidden)

    # compute diff(g) * hessian(f) * diff(g)
    vxx1 = [
        paddle.einsum(
            "bi"
            + dim_str
            + ",mi"
            + dim_str
            + ",bm"
            + dim_str
            + ",mj"
            + dim_str
            + ",bj"
            + dim_str
            + "->b"
            + dim_str,
            w,
            weights_1,
            weights_2 * diff_diff_tanh,
            weights_1,
            w,
        )
        for w in ux
    ]  # (b,x,y,t)

    # compute diff(f) * hessian(g)
    vxx2 = [
        paddle.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in uxx
    ]
    vxx = [paddle.unsqueeze(a + b, axis=1) for a, b in zip(vxx1, vxx2)]

    return vxx
