# Copyright 2024-2025 The Wan Team and The RealisDance-DiT Team. All rights reserved.
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
import gc
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from datetime import timedelta
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", 0))


def get_local_rank():
    if torch.cuda.device_count() == 0:
        print("WARNING: No available GPU.")
        return 0
    return get_rank() % torch.cuda.device_count()


def is_distributed():
    return get_world_size() > 1


def is_main_process():
    return not is_distributed() or get_rank() == 0


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    model_type="wan"
):
    model = model.to(torch.float32)
    if model_type == "wan":
        block_list = list(model.blocks)
    elif model_type == "t5":
        block_list = list(model.encoder.block)
    elif model_type == "clip":
        block_list = list(model.vision_model.encoder.layers)
    else:
        raise NotImplementedError(f"Unknown model type {model_type}")
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in block_list),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=(model_type == "wan"),
    )
    gc.collect()
    torch.cuda.empty_cache()
    return model


def init_dist():
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=24))
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=rank, world_size=world_size)
    initialize_model_parallel(sequence_parallel_degree=world_size, ulysses_degree=world_size)


def hook_for_multi_gpu_inference(pipe):
    local_rank = get_local_rank()
    pipe.transformer = shard_model(pipe.transformer, device_id=local_rank, model_type="wan")
    pipe.image_encoder = shard_model(pipe.image_encoder, device_id=local_rank, model_type="clip")
    pipe.text_encoder = shard_model(pipe.text_encoder, device_id=local_rank, model_type="t5")
    pipe.vae = pipe.vae.to(local_rank)
    sp_degree = get_world_size()
    pipe.transformer.set_sp_degree(sp_degree)
    return pipe
