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

import os

import paddle
import pytest
from paddle import distributed as dist

from modulus.distributed import DistributedManager
from modulus.distributed.autograd import (
    all_gather_v,
    gather_v,
    indexed_all_to_all_v,
    scatter_v,
)


def run_test_scatter_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    sizes = [r + 2 for r in range(world_size)]

    tensor = paddle.arange(world_size, dtype=paddle.float32) + 1
    tensor = tensor.reshape([-1, 1]).expand([-1, tensor_dim]).contiguous()
    tensor = tensor.repeat_interleave(
        repeats=paddle.to_tensor(sizes, place="gpu:0"), axis=0
    )
    tensor.stop_gradient = False

    scattered_tensor = scatter_v(tensor, sizes, dim=0, src=0, group=None)
    expected_tensor = paddle.ones((sizes[rank], tensor_dim), dtype=paddle.float32) * (
        rank + 1
    )
    assert paddle.allclose(expected_tensor, scattered_tensor)

    grad_out = paddle.ones_like(scattered_tensor) * (-1)
    scattered_tensor.backward(grad_out)

    if rank == 0:
        expected_grad = paddle.ones_like(tensor) * (-1)
        assert paddle.allclose(tensor.grad, expected_grad)

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_gather_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    tensor = (rank + 1) * paddle.ones((rank + 2, tensor_dim), dtype=paddle.float32)
    tensor.stop_gradient = False
    sizes = [r + 2 for r in range(world_size)]

    gathered_tensor = gather_v(tensor, sizes, axis=0, dst=0, group=None)

    if rank == 0:
        expected_tensor = paddle.arange(world_size, dtype=paddle.float32) + 1
        expected_tensor = (
            expected_tensor.reshape([-1, 1]).expand([-1, tensor_dim]).contiguous()
        )
        expected_tensor = expected_tensor.repeat_interleave(
            repeats=paddle.to_tensor(sizes, place="gpu:0"), axis=0
        )

        assert paddle.allclose(expected_tensor, gathered_tensor)

    grad_out = paddle.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(grad_out)

    expected_grad = paddle.ones_like(tensor) * (-1)
    assert paddle.allclose(tensor.grad, expected_grad)

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_all_gather_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    tensor = (rank + 1) * paddle.ones((rank + 2, tensor_dim), dtype=paddle.float32)
    tensor.stop_gradient = False
    sizes = [r + 2 for r in range(world_size)]

    gathered_tensor = all_gather_v(tensor, sizes, axis=0, group=None)

    expected_tensor = paddle.arange(world_size, dtype=paddle.float32) + 1
    expected_tensor = (
        expected_tensor.reshape([-1, 1]).expand([-1, tensor_dim]).contiguous()
    )
    expected_tensor = expected_tensor.repeat_interleave(
        repeats=paddle.to_tensor(sizes).to(device=manager.device), axis=0
    )

    assert paddle.allclose(expected_tensor, gathered_tensor)

    grad_out = paddle.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(grad_out)

    expected_grad = paddle.ones_like(tensor) * (-1) * world_size
    assert paddle.allclose(tensor.grad, expected_grad)

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_indexed_all_to_all_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    # this test case is not ideal as it quite similar to the non-indexed case
    # however, it is a first start to test correctness in general
    tensor_dim = 4
    tensor = paddle.arange(1, world_size + 1, dtype=paddle.float32)
    tensor = tensor.reshape([-1, 1]).expand([-1, tensor_dim]).contiguous()
    tensor = tensor.repeat_interleave(repeats=rank + 1, axis=0)
    tensor.stop_gradient = False

    sizes = [[r + 1 for _ in range(world_size)] for r in range(world_size)]

    indices = [
        paddle.nonzero(tensor[:, 0] == (r + 1)).reshape([-1]) for r in range(world_size)
    ]

    gathered_tensor = indexed_all_to_all_v(
        tensor, indices, sizes, axis=0, use_fp32=True, group=None
    )

    expected_size_along_dim = sum([sizes[r][rank] for r in range(world_size)])
    expected_tensor = paddle.ones(
        (expected_size_along_dim, tensor_dim),
        device=manager.device,
        dtype=paddle.float32,
    ) * (rank + 1)

    assert paddle.allclose(expected_tensor, gathered_tensor)

    grad_out = paddle.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(grad_out)

    expected_grad = paddle.ones_like(tensor) * (-1)
    assert paddle.allclose(tensor.grad, expected_grad)

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_autograd_prim(func):
    num_gpus = paddle.device.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    rank = dist.get_rank()
    world_size = 2

    # dist.set_start_method("spawn", force=True)

    dist.spawn(
        func,
        args=(rank, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


@pytest.mark.multigpu
def test_scatter_v():
    run_test_autograd_prim(run_test_scatter_v)


@pytest.mark.multigpu
def test_gather_v():
    run_test_autograd_prim(run_test_gather_v)


@pytest.mark.multigpu
def test_all_gather_v():
    run_test_autograd_prim(run_test_all_gather_v)


@pytest.mark.multigpu
def test_indexed_all_to_all_v():

    run_test_autograd_prim(run_test_indexed_all_to_all_v)


if __name__ == "__main__":
    pytest.main([__file__])
