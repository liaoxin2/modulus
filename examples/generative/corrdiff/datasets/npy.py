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

import numpy as np
import paddle
from paddle.io import Dataset
from typing import List, Tuple


class NpyDataset(Dataset):
    """
    Custom Dataset for loading data and labels from NumPy files.

    Args:
        data_file (str): Path to the NumPy file containing the input data.
        labels_file (str): Path to the NumPy file containing the labels.
        transform (callable, optional): Optional transform to be applied
            on a sample. Default is None.
        kwargs: Additional keyword arguments (currently unused).
    """

    def __init__(self, data_file, labels_file, transform=None, **kwargs):
        self.data = np.load(data_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_label = self.labels[idx]

        # Apply any transformations if specified
        if self.transform:
            sample_data = self.transform(sample_data)
            sample_label = self.transform(sample_label)

        return (sample_label, sample_data, idx)

    def longitude(self) -> int:
        """Get longitude values from the dataset."""
        return 266

    def latitude(self) -> int:
        """Get latitude values from the dataset."""
        return 200

    def input_channels(self) -> List:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return [
            "resu1",
            "resv1",
            "respsf",
            "resrh2",
            "reshgt",
            "resv500",
            "resu500",
            "resv200",
            "resu200",
            "restmp2",
            "restmp925",
            "restmp850",
            "restmp700",
            "restmp500",
        ]

    def output_channels(self) -> List:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return ["resu1", "resv1"]

    def time(self) -> List:
        """Get time values from the dataset."""
        return [0, 1, 2, 3, 4]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        return (266, 200)

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        scale = np.array(
            [
                41.633 + 32.601242,
                35.006054 + 34.006493,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
        bias = np.array([-32.601242, -34.006493, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 使用 reshape 或 np.newaxis 调整形状
        scale = scale[:, np.newaxis, np.newaxis]
        bias = bias[:, np.newaxis, np.newaxis]
        return x * scale + bias

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        scale = np.array([46.190834 + 34.765285, 39.674255 + 36.528397])
        bias = np.array([-34.765285, -36.528397])

        # 使用 reshape 或 np.newaxis 调整形状
        scale = scale[:, np.newaxis, np.newaxis]
        bias = bias[:, np.newaxis, np.newaxis]
        return x * scale + bias
