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
import time, psutil, hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from modulus import Module
from modulus.models.diffusion import UNet, EDMPrecondSR
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import RegressionLoss, ResLoss
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from datasets.dataset import init_train_valid_datasets_from_config
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)
from visualdl import LogWriter


@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        with LogWriter(logdir="visualdl_log") as writer:
            writer.add_scalar(tag="example_metric", step=1, value=0.5)
    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    OmegaConf.resolve(cfg)

    OmegaConf.set_struct(cfg.dataset, False)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    if hasattr(cfg, "validation_dataset"):
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation_dataset)
    else:
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = "float16" if fp_optimizations == "amp-fp16" else "bfloat16"
    logger.info(f"Saving the outputs in {os.getcwd()}")
    set_seed(dist.rank)
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2,
    }

    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        cfg.dataset,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
    )
    test_data = dataset_iterator
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels
    if cfg.model.name == "patched_diffusion":
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
    patch_shape = patch_shape_y, patch_shape_x
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")
    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels
    if cfg.model.name not in ("regression", "diffusion", "patched_diffusion"):
        raise ValueError("Invalid model")
    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
    }
    standard_model_cfgs = {
        "regression": {
            "img_channels": 4,
            "N_grid_channels": 4,
            "embedding_type": "zero",
        },
        "diffusion": {
            "img_channels": img_out_channels,
            "gridtype": "sinusoidal",
            "N_grid_channels": 4,
        },
        "patched_diffusion": {
            "img_channels": img_out_channels,
            "gridtype": "learnable",
            "N_grid_channels": 100,
        },
    }
    model_args.update(standard_model_cfgs[cfg.model.name])
    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))
    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    else:
        model = EDMPrecondSR(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    model.train()
    for param in model.parameters():
        param.stop_gradient = False
    if dist.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.place,
            find_unused_parameters=dist.find_unused_parameters,
        )
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected a this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        out_1 = regression_net.eval()
        out_1.stop_gradient = not False
        out_1.to(dist.place)
        logger0.success("Loaded the pre-trained regression model")
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if cfg.model.name in ("diffusion", "patched_diffusion"):
        loss_fn = ResLoss(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=cfg.training.hp.lr,
        epsilon=1e-08,
        beta1=[0.9, 0.999][0],
        beta2=[0.9, 0.999][1],
        weight_decay=0.0,
    )
    start_time = time.time()
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")
    if dist.world_size > 1:
        paddle.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=f"checkpoints_{cfg.model.name}",
            models=model,
            optimizer=optimizer,
            device=dist.place,
        )
    except:
        cur_nimg = 0
    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False
    while not done:
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        optimizer.clear_gradients(set_to_zero=not True)
        loss_accum = 0
        for _ in range(num_accumulation_rounds):
            img_clean, img_lr, labels = next(dataset_iterator)
            img_clean = paddle.to_tensor(img_clean, dtype="float32")
            img_lr = paddle.to_tensor(img_lr, dtype="float32")
            labels = paddle.to_tensor(labels, dtype="float32")
            with paddle.amp.auto_cast(dtype=amp_dtype, enable=enable_amp):
                loss = loss_fn(
                    net=model,
                    img_clean=img_clean,
                    img_lr=img_lr,
                    labels=labels,
                    augment_pipe=None,
                )
            loss = loss.sum() / batch_gpu_total
            loss_accum += loss / num_accumulation_rounds
            loss.backward()
        loss_sum = paddle.to_tensor(data=[loss_accum], place=dist.place)
        if dist.world_size > 1:
            paddle.distributed.barrier()
            paddle.distributed.all_reduce(
                tensor=loss_sum, op=paddle.distributed.ReduceOp.SUM
            )
        average_loss = (loss_sum / dist.world_size).cpu().item()
        if dist.rank == 0:
            writer.add_scalar("training_loss", average_loss, cur_nimg)
        lr_rampup = cfg.training.hp.lr_rampup
        for g in optimizer.param_groups:
            if lr_rampup > 0:
                g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
            g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // 5000000.0)
            current_lr = g["lr"]
            if dist.rank == 0:
                writer.add_scalar("learning_rate", current_lr, cur_nimg)
        handle_and_clip_gradients(
            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        optimizer.step()
        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if is_time_for_periodic_task(
                cur_nimg,
                cfg.training.io.validation_freq,
                done,
                cfg.training.hp.total_batch_size,
                dist.rank,
            ):
                with paddle.no_grad():
                    for _ in range(cfg.training.io.validation_steps):
                        img_clean_valid, img_lr_valid, labels_valid = next(
                            validation_dataset_iterator
                        )
                        img_clean_valid = (
                            img_clean_valid.to(dist.place).to("float32").contiguous()
                        )
                        img_lr_valid = (
                            img_lr_valid.to(dist.place).to("float32").contiguous()
                        )
                        labels_valid = labels_valid.to(dist.place).contiguous()
                        loss_valid = loss_fn(
                            net=model,
                            img_clean=img_clean_valid,
                            img_lr=img_lr_valid,
                            labels=labels_valid,
                            augment_pipe=None,
                        )
                        loss_valid = (loss_valid.sum() / batch_gpu_total).cpu().item()
                        valid_loss_accum += (
                            loss_valid / cfg.training.io.validation_steps
                        )
                    valid_loss_sum = paddle.to_tensor(
                        data=[valid_loss_accum], place=dist.place
                    )
                    if dist.world_size > 1:
                        paddle.distributed.barrier()
                        paddle.distributed.all_reduce(
                            tensor=valid_loss_sum, op=paddle.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        writer.add_scalar(
                            "validation_loss", average_valid_loss, cur_nimg
                        )
        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            tick_end_time = time.time()
            fields = []
            fields += [f"samples {cur_nimg:<9.1f}"]
            fields += [f"training_loss {average_loss:<7.2f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {tick_end_time - start_time:<7.1f}"]
            fields += [f"sec_per_tick {tick_end_time - tick_start_time:<7.1f}"]
            fields += [
                f"sec_per_sample {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg):<7.2f}"
            ]
            fields += [
                f"cpu_mem_gb {psutil.Process(os.getpid()).memory_info().rss / 2 ** 30:<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_gb {paddle.device.cuda.max_memory_allocated(device=dist.place) / 2 ** 30:<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_reserved_gb {paddle.device.cuda.max_memory_reserved(device=dist.place) / 2 ** 30:<6.2f}"
            ]
            logger0.info(" ".join(fields))
            torch.cuda.reset_peak_memory_stats()
        if dist.world_size > 1:
            paddle.distributed.barrier()
        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.save_checkpoint_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            save_checkpoint(
                path=f"checkpoints_{cfg.model.name}",
                models=model,
                optimizer=optimizer,
                epoch=cur_nimg,
            )
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
