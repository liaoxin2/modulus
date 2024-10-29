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
