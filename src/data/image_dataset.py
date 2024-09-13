import json
import os.path as osp
import traceback

import cv2
import numpy as np
import torch
import random

from torchvision.transforms import transforms
from torch.utils.data import Dataset

from src.data.dwpose_utils.draw_pose import draw_pose


SMPL_ROOT = '~/dataset_smpl/'
HAMER_ROOT = '~/dataset_hamer/'


class ImageDataset(Dataset):
    def __init__(
        self, split, data_lst_file_path_lst,
        sample_size=(768, 576), clip_size=(320, 240),
        scale=(1.0, 1.0), version="v1", draw_face=False
    ):
        super().__init__()

        self.split = split
        self.sample_size = sample_size
        self.clip_size = clip_size
        self.data_lst = []
        for data_lst_file_path in data_lst_file_path_lst:
            self.data_lst += open(data_lst_file_path).readlines()
        self.length = len(self.data_lst)
        self.version = version
        self.draw_face = draw_face

        print(f"{self.split} dataset length is {self.length}")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # ratio is w/h
            transforms.RandomResizedCrop(
                sample_size, scale=scale,
                ratio=(sample_size[1]/sample_size[0], sample_size[1]/sample_size[0]), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.clip_transform = transforms.Compose([
            transforms.ToTensor(),
            # ratio is w/h
            transforms.RandomResizedCrop(
                clip_size, scale=scale,
                ratio=(clip_size[1]/clip_size[0], clip_size[1]/clip_size[0]), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406],     # used for dino
                                 [0.229, 0.224, 0.225],     # used for dino
                                 inplace=True),
        ])
        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            # ratio is w/h
            transforms.RandomResizedCrop(
                sample_size, scale=scale,
                ratio=(sample_size[1]/sample_size[0], sample_size[1]/sample_size[0]), antialias=True),
        ])

    def get_file_path(self, cur_img_dir, img_key):
        img_folder_key = '/img/'
        dwpose_folder_key = '/dwpose_json/'
        hamer_folder_key = '/hamer_img/'
        img_path = osp.join(cur_img_dir, img_key)
        ext = osp.splitext(img_path)[-1]
        dwpose_path = img_path.replace(img_folder_key, dwpose_folder_key).replace(ext, '.json')
        if self.version == "v1":
            hamer_path = osp.join(
                HAMER_ROOT, img_path.split(img_folder_key)[-1].replace(ext, '.png').replace(' ', ''))
        elif self.version == "v2":
            hamer_path = img_path.replace(img_folder_key, hamer_folder_key).replace(ext, '.png')
        else:
            raise ValueError(f"Invalid dy_dataset version {self.version}")
        smpl_path = osp.join(
            SMPL_ROOT, img_path.split(img_folder_key)[-1].replace(ext, '.png').replace(' ', ''))

        return img_path, dwpose_path, hamer_path, smpl_path

    def __len__(self):
        return len(self.data_lst)

    def open_json_and_draw_pose(self, path, h=None, w=None, canvas=None, draw_hands=True):
        with open(path) as jsonfile:
            pose = json.load(jsonfile)
        pose['bodies']['candidate'] = np.array(pose['bodies']['candidate'])
        pose['bodies']['subset'] = np.array(pose['bodies']['subset'])
        pose['hands'] = np.array(pose['hands'])
        if self.draw_face:
            pose['faces'] = np.array(pose['faces'])
        return draw_pose(pose, h, w, canvas, draw_hands=draw_hands, draw_face=self.draw_face)

    def get_metadata(self, idx):
        idx = idx % self.length

        data_info = self.data_lst[idx].rstrip().split(',')
        cur_img_dir = data_info[0]
        cur_vid_key = cur_img_dir.split('/')[-1]
        img_key_lst = data_info[1:]
        img_length = len(img_key_lst)
        ref_id, tgt_id = random.sample(range(img_length), 2)
        ref_img_key = img_key_lst[ref_id] \
            if self.split == 'train' else img_key_lst[0]
        tgt_img_key = img_key_lst[tgt_id] \
            if self.split == 'train' else img_key_lst[1:][len(img_key_lst[1:])//2]
        tgt_img_path, tgt_pose_path, tgt_hamer_path, tgt_smpl_path = self.get_file_path(
            cur_img_dir, tgt_img_key)
        ref_img_path, ref_pose_path, ref_hamer_path, ref_smpl_path = self.get_file_path(
            cur_img_dir, ref_img_key)

        ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
        ref_hamer_img = cv2.cvtColor(cv2.imread(ref_hamer_path), cv2.COLOR_BGR2RGB)
        ref_smpl_img = cv2.cvtColor(cv2.imread(ref_smpl_path), cv2.COLOR_BGR2RGB)
        ref_pose_img = self.open_json_and_draw_pose(ref_pose_path,  ref_img.shape[0], ref_img.shape[1])
        tgt_img = cv2.cvtColor(cv2.imread(tgt_img_path), cv2.COLOR_BGR2RGB)
        tgt_hamer_img = cv2.cvtColor(cv2.imread(tgt_hamer_path), cv2.COLOR_BGR2RGB)
        tgt_smpl_img = cv2.cvtColor(cv2.imread(tgt_smpl_path), cv2.COLOR_BGR2RGB)
        tgt_pose_img = self.open_json_and_draw_pose(tgt_pose_path,  tgt_img.shape[0], tgt_img.shape[1])

        # preparing outputs
        meta_data = {}
        meta_data['dataset_name'] = ref_img_path.split('/')[5]
        meta_data['img_key'] = f"{cur_vid_key}_{ref_img_key}_{tgt_img_key}"
        meta_data['ref_img'] = ref_img
        meta_data['ref_dino_img'] = ref_img
        meta_data['ref_pose_img'] = ref_pose_img
        meta_data['ref_hamer_img'] = ref_hamer_img
        meta_data['ref_smpl_img'] = ref_smpl_img
        meta_data['tgt_img'] = tgt_img
        meta_data['tgt_pose_img'] = tgt_pose_img
        meta_data['tgt_hamer_img'] = tgt_hamer_img
        meta_data['tgt_smpl_img'] = tgt_smpl_img

        return meta_data

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
                raw_data = self.get_metadata(idx)

                img_key = raw_data['img_key']
                ref_img = raw_data['ref_img']
                ref_dino_img = raw_data['ref_dino_img']
                ref_pose_img = raw_data['ref_pose_img']
                ref_hamer_img = raw_data['ref_hamer_img']
                ref_smpl_img = raw_data['ref_smpl_img']
                tgt_img = raw_data['tgt_img']
                tgt_pose_img = raw_data['tgt_pose_img']
                tgt_hamer_img = raw_data['tgt_hamer_img']
                tgt_smpl_img = raw_data['tgt_smpl_img']

                # DA
                state = torch.get_rng_state()
                ref_img = self.augmentation(ref_img, self.img_transform, state)
                ref_dino_img = self.augmentation(ref_dino_img, self.clip_transform, state)
                ref_pose_img = self.augmentation(ref_pose_img, self.pose_transform, state)
                ref_hamer_img = self.augmentation(ref_hamer_img, self.pose_transform, state)
                ref_smpl_img = self.augmentation(ref_smpl_img, self.pose_transform, state)
                tgt_img = self.augmentation(tgt_img, self.img_transform, state)
                tgt_pose_img = self.augmentation(tgt_pose_img, self.pose_transform, state)
                tgt_hamer_img = self.augmentation(tgt_hamer_img, self.pose_transform, state)
                tgt_smpl_img = self.augmentation(tgt_smpl_img, self.pose_transform, state)

                return {"data_key": img_key,
                        "image": tgt_img, "pose": tgt_pose_img, "hamer": tgt_hamer_img, "smpl": tgt_smpl_img,
                        "ref_image": ref_img, "ref_image_clip": ref_dino_img,
                        "ref_pose": ref_pose_img, "ref_hamer": ref_hamer_img, "ref_smpl": ref_smpl_img}
            except Exception as e:
                print(self.data_lst[idx])
                print(f"read idx: {idx} error, {type(e).__name__}: {e}")
                print(traceback.format_exc())
                idx = random.randint(0, self.length - 1)
