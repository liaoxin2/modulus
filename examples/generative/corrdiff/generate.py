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

import sys 
sys.path.append('/workspace/workspace/modulus-sym')
import paddle
import hydra
from omegaconf import OmegaConf, DictConfig
import nvtx
import json
import numpy as np
import netCDF4 as nc
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus import Module
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from einops import rearrange
from hydra.utils import to_absolute_path
from modulus.utils.generative import deterministic_sampler, stochastic_sampler
from modulus.models.diffusion.preconditioning import EDMPrecondSR
from modulus.models.diffusion.unet import UNet
from modulus.utils.corrdiff import NetCDFWriter, get_time_from_range, regression_step, diffusion_step
from helpers.generate_helpers import get_dataset_and_sampler, save_images
from helpers.train_helpers import set_patch_shape


@hydra.main(version_base='1.2', config_path='conf', config_name=
    'config_generate')
def main(cfg: DictConfig) ->None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger('generate')
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging('generate.log')

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = ((len(seeds) - 1) // (cfg.generation.seed_batch_size *
        dist.world_size) + 1) * dist.world_size
    all_batches = paddle.to_tensor(data=seeds).tensor_split(num_or_indices=
        num_batches)
    rank_batches = all_batches[dist.rank::dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        paddle.distributed.barrier()

    # Parse the inference input times
    if cfg.generation.times_range and times:
        raise ValueError(
            'Either times_range or times must be provided, but not both')
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg,
        times=times)
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if hasattr(cfg, 'training.hp.patch_shape_x'):
        patch_shape_x = cfg.training.hp.patch_shape_x
    else:
        patch_shape_x = None
    if hasattr(cfg, 'training.hp.patch_shape_y'):
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = patch_shape_y, patch_shape_x
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info('Patch-based training enabled')
    else:
        logger0.info('Patch-based training disabled')
    if cfg.generation.inference_mode == 'regression':
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == 'diffusion':
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == 'all':
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(
            f'Invalid inference mode {cfg.generation.inference_mode}')
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        with open('/workspace/workspace/modulus/examples/generative/corrdiff/corrdiff_inference_package/checkpoints/d/args.json', 'r') as f:
            args = json.load(f)
        net_res = EDMPrecondSR(**args['__args__'])
        model_dict = paddle.load(
            path='/workspace/workspace/modulus/examples/generative/corrdiff/corrdiff_inference_package/checkpoints/diffusion.pdparams')
        net_res.load_dict(model_dict)
        net_res.eval()
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
    else:
        net_res = None
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        with open('/workspace/workspace/modulus/examples/generative/corrdiff/corrdiff_inference_package/checkpoints/r/args.json', 'r') as f:
            args = json.load(f)
        net_reg = UNet(**args['__args__'])
        model_dict = paddle.load(
            path='/workspace/workspace/modulus/examples/generative/corrdiff/corrdiff_inference_package/checkpoints/regression.pdparams')
        net_reg.load_dict(model_dict)
        net_reg.eval()
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
    else:
        net_reg = None
    # if cfg.generation.perf.use_torch_compile:
    #     torch._dynamo.reset()
    #     if net_res:
    #         net_res = torch.compile(net_res, mode='reduce-overhead')
    if cfg.sampler.type == 'deterministic':
        if cfg.generation.hr_mean_conditioning:
            raise NotImplementedError(
                'High-res mean conditioning is not yet implemented for the deterministic sampler'
                )
        sampler_fn = partial(deterministic_sampler, num_steps=cfg.sampler.
            num_steps, solver=cfg.sampler.solver)
    elif cfg.sampler.type == 'stochastic':
        sampler_fn = partial(stochastic_sampler, img_shape=img_shape[1],
            patch_shape=patch_shape[1], boundary_pix=cfg.sampler.
            boundary_pix, overlap_pix=cfg.sampler.overlap_pix)
    else:
        raise ValueError(f'Unknown sampling method {cfg.sampling.type}')

    def generate_fn():
        img_shape_y, img_shape_x = img_shape
        with nvtx.annotate('generate_fn', color='green'):
            if cfg.generation.sample_res == 'full':
                image_lr_patch = image_lr
            else:
                paddle.framework.core.nvprof_nvtx_push('rearrange')
                image_lr_patch = rearrange(image_lr,
                    'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=img_shape_y //
                    patch_shape[0], w1=img_shape_x // patch_shape[1])
                paddle.framework.core.nvprof_nvtx_pop()
            #image_lr_patch = paddle.transpose(image_lr_patch, perm=[0, 3, 1, 2])
            if net_reg:
                with nvtx.annotate('regression_model', color='yellow'):
                    image_reg = regression_step(net=net_reg, img_lr=
                        image_lr_patch, latents_shape=(cfg.generation.
                        seed_batch_size, img_out_channels, img_shape[1],
                        img_shape[0]))
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate('diffusion model', color='purple'):
                    image_res = diffusion_step(net=net_res, sampler_fn=
                        sampler_fn, seed_batch_size=cfg.generation.
                        seed_batch_size, img_shape=img_shape,
                        img_out_channels=img_out_channels, rank_batches=
                        rank_batches, img_lr=image_lr_patch.expand(shape=[
                        cfg.generation.seed_batch_size, -1, -1, -1]),
                        rank=dist.rank, device=device, hr_mean=mean_hr)
            if cfg.generation.inference_mode == 'regression':
                image_out = image_reg
            elif cfg.generation.inference_mode == 'diffusion':
                image_out = image_res
            else:
                image_out = image_reg + image_res
            if cfg.generation.sample_res != 'full':
                image_out = rearrange(image_out,
                    '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=img_shape_y //
                    patch_shape[0], w1=img_shape_x // patch_shape[1])
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [paddle.zeros_like(x=image_out,
                        dtype=image_out.dtype) for _ in range(dist.world_size)]
                else:
                    gathered_tensors = None
                paddle.distributed.barrier()
                paddle.distributed.gather(tensor=image_out, gather_list=
                    gathered_tensors if dist.rank == 0 else None, dst=0)
                if dist.rank == 0:
                    return paddle.concat(x=gathered_tensors)
                else:
                    return None
            else:
                return image_out
    logger0.info('Generating images...')
    batch_size = 1
    warmup_steps = min(len(times), 2)
    with nc.Dataset(f'output_{dist.rank}.nc', 'w') as f:
        f.cfg = str(cfg)
        # with paddle.profiler.profiler():
        data_loader = paddle.io.DataLoader(dataset=dataset,
             batch_size=1)
        time_index = -1
        writer = NetCDFWriter(f, lat=dataset.latitude(), lon=
            dataset.longitude(), input_channels=dataset.
            input_channels(), output_channels=dataset.output_channels()
            )
        warmup_steps = 2
        start = paddle.device.Event(enable_timing=True)
        end = paddle.device.Event(enable_timing=True)
        writer_executor = ThreadPoolExecutor(max_workers=cfg.
            generation.perf.num_writer_workers)
        writer_threads = []
        times = dataset.time()

        for image_tar, image_lr, index in iter(data_loader):
            time_index += 1
            if dist.rank == 0:
                logger0.info(f'starting index: {time_index}')
            if time_index == warmup_steps:
                start.record()
            image_lr = paddle.to_tensor(image_lr, dtype='float32')
            image_tar = paddle.to_tensor(image_tar, dtype='float32')
            image_out = generate_fn()
            if dist.rank == 0:
                batch_size = tuple(image_out.shape)[0]
                writer_threads.append(writer_executor.submit(
                    save_images, writer, dataset, list(times),
                    image_out.cpu(), image_tar.cpu(), image_lr.cpu(
                    ), time_index, index[0]))
        end.record()
        end.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000.0
        timed_steps = time_index + 1 - warmup_steps
        if dist.rank == 0:
            average_time_per_batch_element = (elapsed_time /
                timed_steps / batch_size)
            logger.info(
                f'Total time to run {timed_steps} and {batch_size} ensembles = {elapsed_time} s'
                )
            logger.info(
                f'Average time per batch element = {average_time_per_batch_element} s'
                )
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
        writer_executor.shutdown()
    logger0.info('Generation Completed.')


if __name__ == '__main__':
    main()
