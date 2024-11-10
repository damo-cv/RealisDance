import os
import torch
import random
import pickle
import traceback
import numpy as np
import decord

from decord import VideoReader
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from src.data.dwpose_utils.draw_pose import draw_pose


decord.bridge.set_bridge('torch')


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir, split,
        sample_size=(768, 576),
        clip_size=(320, 240),
        scale=(1.0, 1.0),
        sample_stride=4,
        sample_n_frames=16,
        ref_mode="random",
        image_finetune=False,
        start_index=-1,
        draw_face=False,
        fix_gap=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.load_dir = os.path.join(self.root_dir, self.split)
        assert os.path.exists(self.load_dir), f"the path {self.load_dir} of the dataset is wrong"

        self.sample_size = sample_size
        self.clip_size = clip_size
        assert sample_stride >= 1
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.at_least_n_frames = (self.sample_n_frames - 1) * self.sample_stride + 1
        # set where the reference frame comes from, which could be "first" or "random"
        assert ref_mode in ["first", "random"], \
            f"the ref_mode could only be \"first\" or \"random\". However \"ref_mode = {ref_mode}\" is given."
        self.ref_mode = ref_mode
        self.image_finetune = image_finetune
        self.start_index = start_index
        self.draw_face = draw_face
        self.fix_gap = fix_gap

        # build data info
        self.data_keys = sorted(os.listdir(os.path.join(self.load_dir, 'video')))
        self.length = len(self.data_keys)

        self.img_transform = transforms.Compose([
            # ratio is w/h
            transforms.RandomResizedCrop(
                sample_size, scale=scale,
                ratio=(sample_size[1]/sample_size[0], sample_size[1]/sample_size[0]), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.clip_transform = transforms.Compose([
            # ratio is w/h
            transforms.RandomResizedCrop(
                    clip_size, scale=scale,
                    ratio=(clip_size[1] / clip_size[0], clip_size[1] / clip_size[0]), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406],  # used for dino
                                 [0.229, 0.224, 0.225],  # used for dino
                                 inplace=True),
        ])
        self.pose_transform = transforms.Compose([
            # ratio is w/h
            transforms.RandomResizedCrop(
                sample_size, scale=scale,
                ratio=(sample_size[1]/sample_size[0], sample_size[1]/sample_size[0]), antialias=True),
        ])

    def __len__(self):
        return len(self.data_keys)

    def get_batch(self, data_id):
        video_path = os.path.join(self.load_dir, 'video', data_id)
        video_reader = VideoReader(video_path)
        hamer_path = os.path.join(self.load_dir, 'hamer', data_id)
        hamer_reader = VideoReader(hamer_path)
        smpl_path = os.path.join(self.load_dir, 'smpl', data_id)
        smpl_reader = VideoReader(smpl_path)
        pose_path = os.path.join(self.load_dir, 'dwpose', data_id.replace('.mp4', '.pkl'))
        with open(pose_path, 'rb') as pose_file:
            pose_list = pickle.load(pose_file)
        assert len(video_reader) == len(hamer_reader) == len(smpl_reader) == len(pose_list)
        video_length = len(video_reader)

        if self.image_finetune:
            if self.ref_mode == 'random':
                batch_index = random.sample(range(video_length), 2)
            else:
                batch_index = [0, random.randint(1, video_length - 1)]
        else:
            if self.ref_mode == 'random':
                ref_index = [random.randint(0, video_length - 1)]
            else:
                ref_index = [0]

            if self.fix_gap:
                if video_length < self.at_least_n_frames:
                    raise RuntimeError(f"The `video_length` ({video_length}) is less "
                                       f"than `at_least_n_frames`({self.at_least_n_frames}). "
                                       f"Thus, this video will be skiped.")
                clip_length = self.at_least_n_frames
                start_idx = random.randint(
                    0, video_length - clip_length
                ) if self.start_index < 0 else max(min(self.start_index, video_length - clip_length), 0)
                image_index = list(range(start_idx, clip_length + start_idx, self.sample_stride))
            else:
                clip_length = min(video_length, self.at_least_n_frames)
                start_idx = random.randint(
                    0, video_length - clip_length
                ) if self.start_index < 0 else max(min(self.start_index, video_length - clip_length), 0)
                image_index = list(
                    np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int))
            batch_index = ref_index + image_index

        image = video_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
        hamer = hamer_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
        smpl = smpl_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0

        pose = [draw_pose(pose_list[batch_index[idx]], image.shape[-2], image.shape[-1], draw_face=self.draw_face)
                for idx in range(len(batch_index))]

        pose = torch.from_numpy(
            np.stack(pose, axis=0)).permute(0, 3, 1, 2).contiguous() / 255.0
        if self.image_finetune:
            ref_image, image = image[0], image[1]
            ref_pose, ref_hamer, ref_smpl = pose[0], hamer[0], smpl[0]
            pose, hamer, smpl = pose[1], hamer[1], smpl[1]
        else:
            ref_image, image = image[0], image[1:]
            ref_pose, ref_hamer, ref_smpl = pose[0], hamer[0], smpl[0]
            pose, hamer, smpl = pose[1:], hamer[1:], smpl[1:]

        del video_reader
        del hamer_reader
        del smpl_reader
        return image, pose, hamer, smpl, ref_image, ref_pose, ref_hamer, ref_smpl

    @staticmethod
    def augmentation(frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __getitem__(self, idx):
        try_cnt = 0
        while True:
            try:
                try_cnt += 1
                if try_cnt > 10:
                    break
                data_id = self.data_keys[idx]
                image, pose, hamer, smpl, _ref_image, ref_pose, ref_hamer, ref_smpl = self.get_batch(data_id)
                state = torch.get_rng_state()
                ref_image = self.augmentation(_ref_image, self.img_transform, state)
                ref_image_clip = self.augmentation(_ref_image, self.clip_transform, state)
                ref_pose = self.augmentation(ref_pose, self.pose_transform, state)
                ref_hamer = self.augmentation(ref_hamer, self.pose_transform, state)
                ref_smpl = self.augmentation(ref_smpl, self.pose_transform, state)
                image = self.augmentation(image, self.img_transform, state)
                pose = self.augmentation(pose, self.pose_transform, state)
                hamer = self.augmentation(hamer, self.pose_transform, state)
                smpl = self.augmentation(smpl, self.pose_transform, state)
                if self.image_finetune:
                    return {"data_key": data_id, "image": image,
                            "pose": pose, "hamer": hamer, "smpl": smpl,
                            "ref_image": ref_image, "ref_image_clip": ref_image_clip,
                            "ref_pose": ref_pose, "ref_hamer": ref_hamer, "ref_smpl": ref_smpl}

                else:
                    return {"data_key": data_id, "image": image.permute(1, 0, 2, 3).contiguous(),
                            "pose": pose.permute(1, 0, 2, 3).contiguous(),
                            "hamer": hamer.permute(1, 0, 2, 3).contiguous(),
                            "smpl": smpl.permute(1, 0, 2, 3).contiguous(),
                            "ref_image": ref_image, "ref_image_clip": ref_image_clip,
                            "ref_pose": ref_pose, "ref_hamer": ref_hamer, "ref_smpl": ref_smpl}
            except Exception as e:
                print(f"read idx:{idx} error, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                idx = random.randint(0, self.length - 1)
