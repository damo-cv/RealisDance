import os

import PIL.Image
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

from omegaconf.listconfig import ListConfig
from torch.utils.data.distributed import DistributedSampler
from src.data.image_dataset import ImageDataset
from src.data.video_dataset import VideoDataset


DATASET_REG_DICT = {
    "ImageDataset": ImageDataset,
    "VideoDataset": VideoDataset,
}


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, save_frame=False):
    videos = rearrange(videos, "b c f h w -> f b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = ((x + 1.0) / 2.0).clamp(0, 1)
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    if save_frame:
        os.makedirs(path, exist_ok=True)
        for idx, x in enumerate(outputs):
            path_i = os.path.join(path, f"{idx}.jpg")
            img = PIL.Image.fromarray(x)
            img.save(path_i)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, outputs, fps=fps,)


def get_distributed_dataloader(dataset_config, batch_size,
                               num_processes, num_workers, shuffle,
                               global_rank, seed):
    # Get the dataset
    if isinstance(dataset_config, ListConfig):
        dataset_list = []
        for dc in dataset_config:
            dc = dc.get("dataset")
            repeat_times = dc.get("repeat_times", 1)
            dataset_class = DATASET_REG_DICT[dc.get("dataset_class")]
            dataset_list += (
                dataset_class(**OmegaConf.to_container(dc.get("args"))),
            ) * repeat_times
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset_class = DATASET_REG_DICT[dataset_config.get("dataset_class")]
        dataset = dataset_class(**OmegaConf.to_container(dataset_config.get("args")))
    # Get dist sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=shuffle,
        seed=seed,
    )
    # DataLoaders creation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_dataloader(dataset_config, batch_size, num_workers, shuffle):
    # Get the dataset
    if isinstance(dataset_config, ListConfig):
        dataset_list = []
        for dc in dataset_config:
            dc = dc.get("dataset")
            repeat_times = dc.get("repeat_times", 1)
            dataset_class = DATASET_REG_DICT[dc.get("dataset_class")]
            dataset_list += (
                dataset_class(**OmegaConf.to_container(dc.get("args"))),
            ) * repeat_times
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset_class = DATASET_REG_DICT[dataset_config.get("dataset_class")]
        dataset = dataset_class(**OmegaConf.to_container(dataset_config.get("args")))

    # DataLoaders creation
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def sanity_check(batch, output_dir, image_finetune, global_rank):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and k != "data_key":
            for idx, (data_id, data_value) in enumerate(zip(batch["data_key"], v)):
                data_value = data_value[None, ...]
                if "image" in k:
                    data_value = (data_value / 2. + 0.5).clamp(0, 1)
                elif data_value.shape[1] == 6:
                    if image_finetune:
                        data_value = rearrange(data_value, 'b (e d) h w -> b d h (e w)', e=2, d=3)
                    else:
                        data_value = rearrange(data_value, 'b (e d) f h w -> b d f h (e w)', e=2, d=3)
                if image_finetune or "ref" in k:
                    torchvision.utils.save_image(
                        data_value, os.path.join(output_dir, f"{data_id}_{k}_{global_rank}.jpg"))
                else:
                    save_videos_grid(
                        data_value, os.path.join(output_dir, f"{data_id}_{k}_{global_rank}.gif"), rescale=False)
