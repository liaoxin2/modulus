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

from typing import Iterable, Tuple, Union
import copy

from modulus.utils.generative import InfiniteSampler
from modulus.distributed import DistributedManager

from . import base, cwb, hrrrmini, npy

# this maps all known dataset types to the corresponding init function
known_datasets = {
    "cwb": cwb.get_zarr_dataset,
    "hrrr_mini": hrrrmini.HRRRMiniDataset,
    "npy": npy.NpyDataset,
}


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    train_test_split: bool = False,
) -> Tuple[
    base.DownscalingDataset,
    Iterable,
    Union[base.DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - train_test_split (bool): A flag to determine whether to create a validation dataset. Defaults to True.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]: A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if train_test_split is True.
    """

    config = copy.deepcopy(dataset_cfg)

    if dataset_cfg["type"] == "npy":
        (
            dataset,
            dataset_iter,
            valid_dataset,
            valid_dataset_iter,
        ) = init_dataset_from_config_npy(
            config, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        train_test_split = config.pop("train_test_split", True)
        dataset, dataset_iter = init_dataset_from_config(
            config, dataloader_cfg, batch_size=batch_size, seed=seed
        )

        if train_test_split:
            valid_dataset_cfg = copy.deepcopy(config)
            validation_dataset_cfg = {"train": False, "all_times": False}
            valid_dataset_cfg.update(validation_dataset_cfg)
            (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
                valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
            )
        else:
            valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "train_test_split" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["train_test_split"]
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dataset_iterator = iter(
        paddle.io.DataLoader(
            dataset=dataset_obj,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return dataset_obj, dataset_iterator


def init_dataset_from_config_npy(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg["type"]
    if "train_test_split" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["train_test_split"]

    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)

    if "vaild_size" in dataset_cfg:
        val_size = int(len(dataset_obj) * dataset_cfg["vaild_size"])
        train_size = len(dataset_obj) - val_size
        train_dataset, val_dataset = paddle.io.random_split(
            dataset_obj, [train_size, val_size]
        )

        if dataloader_cfg is None:
            dataloader_cfg = {}

        dataset_iterator = iter(
            paddle.io.DataLoader(
                dataset=dataset_obj,
                batch_size=batch_size,
                worker_init_fn=None,
                **dataloader_cfg,
            )
        )

        val_iterator = iter(
            paddle.io.DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                worker_init_fn=None,
                **dataloader_cfg,
            )
        )

        return (train_dataset, dataset_iterator, val_dataset, val_iterator)
    else:
        if dataloader_cfg is None:
            dataloader_cfg = {}

        dataset_iterator = iter(
            paddle.io.DataLoader(
                dataset=dataset_obj,
                batch_size=batch_size,
                worker_init_fn=None,
                **dataloader_cfg,
            )
        )

        return (dataset_obj, dataset_iterator)
