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

from typing import Sequence, Tuple, Union

import paddle
import paddle as pd

from .healpix_layers import HEALPixLayer

#
# RECURRENT BLOCKS
#


class ConvGRUBlock(pd.nn.Layer):
    """Class that implements a Convolutional GRU
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        kernel_size: int = 1,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer=paddle.nn.Conv2D,
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.conv_can = geometry_layer(
            layer=paddle.nn.Conv2D,
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.h = pd.zeros([1, 1, 1, 1])

    def forward(self, inputs: Sequence) -> Sequence:
        """Forward pass of the ConvGRUBlock

        Parameters
        ----------
        inputs: Sequence
            Input to the forward pass

        Returns
        -------
        Sequence
            Result of the forward pass
        """
        if inputs.shape != self.h.shape:
            self.h = pd.zeros_like(inputs)
        combined = pd.concat([inputs, self.h], axis=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = pd.split(combined_conv, self.channels, axis=1)
        reset_gate = pd.sigmoid(gamma)
        update_gate = pd.sigmoid(beta)

        combined = pd.concat([inputs, reset_gate * self.h], axis=1)
        cc_cnm = self.conv_can(combined)
        cnm = pd.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        """Reset the update gates"""
        self.h = pd.zeros_like(self.h)


#
# CONV BLOCKS
#


class BasicConvBlock(pd.nn.Layer):
    """Convolution block consisting of n subsequent convolutions and activations"""

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: pd.nn.Layer = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernel
        dilation: int, optional
            Spacing between kernel points, passed to paddle.nn.Conv2D
        n_layers:
            Number of convolutional layers
        latent_channels:
            Number of latent channels
        activation: paddle.nn.Layer, optional
            Activation function to use
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                geometry_layer(
                    layer=paddle.nn.Conv2D,
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = pd.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the BasicConvBlock

        Parameters
        ----------
        x: paddle.Tensor
            inputs to the forward pass

        Returns
        -------
        paddle.Tensor
            result of the forward pass
        """
        return self.convblock(x)


class ConvNeXtBlock(pd.nn.Layer):
    """Class implementing a modified ConvNeXt network as described in https://arxiv.org/pdf/2201.03545.pdf
    and shown in figure 4
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: pd.nn.Layer = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to paddle.nn.Conv2D
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        latent_channels: int, optional
            Number of latent channels
        activation: paddle.nn.Layer, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        self.convblock = pd.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the ConvNextBlock

        Parameters
        ----------
        x: paddle.Tensor
            inputs to the forward pass

        Returns
        -------
        paddle.Tensor
            result of the forward pass
        """
        return self.skip_module(x) + self.convblock(x)


class DoubleConvNeXtBlock(pd.nn.Layer):
    """Modification of ConvNeXtBlock block this time putting two sequentially
    in a single block with the number of channels in the middle being the
    number of latent channels
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        latent_channels: int = 1,
        activation: pd.nn.Layer = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters:
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to paddle.nn.Conv2D
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: paddle.nn.Layer, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        if in_channels == int(latent_channels):
            self.skip_module1 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module1 = geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        if out_channels == int(latent_channels):
            self.skip_module2 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module2 = geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )

        # 1st ConvNeXt block, the output of this one remains internal
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        # 1x1 convolution establishing increased channels
        convblock1.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        # 1x1 convolution returning to latent channels
        convblock1.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock1.append(activation)
        self.convblock1 = pd.nn.Sequential(*convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        # 1x1 convolution establishing increased channels
        convblock2.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        # 1x1 convolution reducing to output channels
        convblock2.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock2.append(activation)
        self.convblock2 = pd.nn.Sequential(*convblock2)

    def forward(self, x):
        """Forward pass of the DoubleConvNextBlock

        Parameters
        ----------
        x: paddle.Tensor
            inputs to the forward pass

        Returns
        -------
        paddle.Tensor
            result of the forward pass
        """
        # internal convnext result
        x1 = self.skip_module1(x) + self.convblock1(x)
        # return second convnext result
        return self.skip_module2(x1) + self.convblock2(x1)


class SymmetricConvNeXtBlock(pd.nn.Layer):
    """Another modification of ConvNeXtBlock block this time using 4 layers and adding
    a layer that instead of going from in_channels to latent*upscale channesl goes to
    latent channels first
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,  # not used, but required for hydra instantiation
        upscale_factor: int = 4,
        activation: pd.nn.Layer = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry layer
        in_channels: int, optional
            The number of input channels
        latent_channels: int, optional
            Number of latent channels
        out_channels: int, optional
            The number of output channels
        kernel_size: int, optional
            Size of the convolutioonal kernels
        dilation: int, optional
            Spacing between kernel points, passed to paddle.nn.Conv2D
        upscale_factor: int, optional
            Upscale factor to apply on the number of latent channels
        activation: paddle.nn.Layer, optional
            Activation function to use between layers
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()

        if in_channels == int(latent_channels):
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )

        # 1st ConvNeXt block, the output of this one remains internal
        convblock = []
        # 3x3 convolution establishing latent channels channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 1x1 convolution establishing increased channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 1x1 convolution returning to latent channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 3x3 convolution from latent channels to latent channels
        convblock.append(
            geometry_layer(
                layer=paddle.nn.Conv2D,
                in_channels=int(latent_channels),
                out_channels=out_channels,  # int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        self.convblock = pd.nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the SymmetricConvNextBlock

        Parameters
        ----------
        x: paddle.Tensor
            inputs to the forward pass

        Returns
        -------
        paddle.Tensor
            result of the forward pass
        """
        # residual connection with reshaped inpute and output of conv block
        return self.skip_module(x) + self.convblock(x)


#
# DOWNSAMPLING BLOCKS
#


class MaxPool(pd.nn.Layer):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the paddle.nn.MaxPool2D class.
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2D
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.maxpool = geometry_layer(
            layer=paddle.nn.MaxPool2D,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(self, x):
        """Forward pass of the MaxPool

        Parameters
        ----------
        x: paddle.Tensor
            The values to MaxPool

        Returns
        -------
        paddle.Tensor
            The MaxPooled values
        """
        return self.maxpool(x)


class AvgPool(pd.nn.Layer):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the paddle.nn.AvgPool2D class.
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2D
        pooling: int, optional
            Pooling kernel size passed to geometry layer
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        self.avgpool = geometry_layer(
            layer=paddle.nn.AvgPool2D,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(self, x):
        """Forward pass of the AvgPool layer

        Parameters
        ----------
        x: paddle.Tensor
            The values to average

        Returns
        -------
        paddle.Tensor
            The averaged values
        """
        return self.avgpool(x)


#
# UPSAMPLING BLOCKS
#


class TransposedConvUpsample(pd.nn.Layer):
    """This class provides a wrapper for a HEALPix (or other) tensor data
    around the paddle.nn.Conv2DTranspose class.
    """

    def __init__(
        self,
        geometry_layer: pd.nn.Layer = HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: pd.nn.Layer = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Parameters
        ----------
        geometry_layer: paddle.nn.Layer, optional
            The wrapper for the geometry of the tensor being bassed to MaxPool2D
        in_channels: int, optional
            The number of input channels
        out_channels: int, optional
            The number of output channels
        upsampling: int, optional
            Size used for upsampling
        activation: paddle.nn.Layer, optional
            Activation function used in upsampling
        enable_nhwc: bool, optional
            Enable nhwc format, passed to wrapper
        enable_healpixpad: bool, optional
            If HEALPixPadding should be enabled, passed to wrapper
        """
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            geometry_layer(
                layer=paddle.nn.Conv2DTranspose,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = pd.nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the TransposedConvUpsample layer

        Parameters
        ----------
        x: paddle.Tensor
            The values to upsample

        Returns
        -------
        paddle.Tensor
            The upsampled values
        """
        return self.upsampler(x)


#
# Helper classes
#


class Interpolate(pd.nn.Layer):
    """Helper class that handles interpolation
    This is done as a class so that scale and mode can be stored
    """

    def __init__(self, scale_factor: Union[int, Tuple], mode: str = "nearest"):
        """
        Parameters:
        ----------
        scale_factor: Union[int , Tuple]
            Multiplier for spatial size, passed to paddle.nn.functional.interpolate
        mode: str, optional
            Interpolation mode used for upsampling, passed to paddle.nn.functional.interpolate
        """
        super().__init__()
        self.interp = pd.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        """Forward pass of the Interpolate layer

        Parameters
        ----------
        x: paddle.Tensor
            inputs to interpolate

        Returns
        -------
        paddle.Tensor
            the interpolated values
        """
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)
