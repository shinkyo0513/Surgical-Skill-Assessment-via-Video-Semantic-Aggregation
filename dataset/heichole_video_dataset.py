import os

import json
import csv
from PIL import Image
from glob import glob
import numpy as np 
import math
from tqdm import tqdm
import bisect
import copy
import pandas as pd
import random
from skimage import transform, filters

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io

import sys
sys.path.append(".")
sys.path.append("..")

from utils import LongRangeSample

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class HeiCholeVideoDataset (Dataset):
    def __init__ (self, root_dir, is_train, task, split_index=1,
                  sampled_timestep_num=32, frames_per_timestep=1,
                  train_sample_augment=1, test_sample_augment=1,
                  train_horiz_flip=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.task = task
        self.split_index = split_index
        assert task in ['Calot', 'Dissection', 'Across']

        split_offset = 3 * (split_index - 1)
        print(split_index, split_offset)
        val_indices = [idx + split_offset for idx in [1, 2, 3, 13, 14, 15]]
        train_indices = [idx for idx in range(1, 25) if idx not in val_indices]
        self.indices = train_indices if self.is_train else val_indices

        self.video_info_dict = self.__read_video_info()
        
        self.sampled_timestep_num = sampled_timestep_num
        self.frames_per_timestep = frames_per_timestep
        self.sample_mode = 'random' if self.is_train else 'last'

        self.train_sample_augment = train_sample_augment
        self.test_sample_augment = test_sample_augment
        self.train_horiz_flip = train_horiz_flip

        self.sample_list = self.__sample_video()

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            # transforms.Resize((112, 150)),
            # transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])

    def __read_taskwise_video_info (self, task_name):
        frame_dir = os.path.join(self.root_dir, 'frames', '5fps_160w')
        annot_dir = os.path.join(self.root_dir, 'annotations', 'Skill')

        video_info_dict = {}
        for vidx in self.indices:
            vname = f'Hei-Chole{vidx}_{task_name}'

            video_frame_dir = os.path.join(frame_dir, vname)
            num_frame = len([f for f in os.listdir(video_frame_dir) if '.jpg' in f])
            # print(vname, num_frame)

            video_annot_dir = os.path.join(annot_dir, f'{vname}_Skill.csv')
            with open(video_annot_dir, 'r') as annot_csv:
                csv_reader = csv.reader(annot_csv, delimiter=',')
                annot_scores = [int(s) for s in list(csv_reader)[0]]
            sum_score = sum(annot_scores)

            video_info_dict[vname] = {'num_frame': num_frame, 'sum_score': sum_score, 'sub_scores': annot_scores}

        return video_info_dict

    def __read_video_info (self):
        if self.task != 'Across':
            return self.__read_taskwise_video_info(self.task)
        else:
            video_info_dict = {}
            for task_name in ['Calot', 'Dissection']:
                video_info_dict.update(self.__read_taskwise_video_info(task_name))
            return video_info_dict
        # return self.__read_taskwise_video_info(self.task)

    def __sample_video (self):
        sample_list = []

        for video_name, video_annots in self.video_info_dict.items():
            # print(video_name, video_annots)
            num_frame = int(video_annots['num_frame'])

            if self.is_train:
                sample_num = self.train_sample_augment
                for sample_idx in range(sample_num):
                    sampled_timesteps = LongRangeSample.long_range_sample(num_frame, self.sampled_timestep_num, self.sample_mode)
                    if self.train_horiz_flip:
                        sample_list.append([video_name, sample_idx, sampled_timesteps, False])
                        sample_list.append([video_name, sample_idx, sampled_timesteps, True])
                    else:
                        sample_list.append([video_name, sample_idx, sampled_timesteps, False])
            else:
                sample_num = self.test_sample_augment
                if sample_num == 1:
                    sampled_timesteps = LongRangeSample.long_range_sample(num_frame, self.sampled_timestep_num, self.sample_mode)
                    sample_list.append([video_name, 0, sampled_timesteps, False])
                elif sample_num == 2:
                    first_sampled_timesteps = LongRangeSample.long_range_sample(num_frame, self.sampled_timestep_num, 'first')
                    last_sampled_timesteps = LongRangeSample.long_range_sample(num_frame, self.sampled_timestep_num, 'last')
                    sample_list.append([video_name, 0, first_sampled_timesteps, False])
                    sample_list.append([video_name, 1, last_sampled_timesteps, False])
                else:
                    for sample_idx in range(sample_num):
                        sampled_timesteps = LongRangeSample.long_range_sample(num_frame, self.sampled_timestep_num, 'random')
                        sample_list.append([video_name, sample_idx, sampled_timesteps, False])
        return sample_list

    def __multi_frame_transform (self, multi_frame, resize=True, normlize=True):
        # Input multi_frame: T x H x W x C, torch.uint8, 0~255
        # Output frame_tensor: T x C x H x W, torch.float, 0~1, normlized 
        frame_tensor = multi_frame.permute(0, 3, 1, 2).contiguous() / 255.0 # T x C x H x W
        if resize:
            frame_tensor = F.interpolate(frame_tensor, self.transform_resize, mode='bilinear')
        if normlize:
            frame_tensor = (frame_tensor - self.transform_mean.view(-1,1,1)) / self.transform_std.view(-1,1,1)
        return frame_tensor

    def __read_frame_tensors (self, video_name, frame_idx_list, task_name, hflip):
        frame_images_dir = os.path.join(self.root_dir, 'frames', '5fps_160w', video_name)
        frame_images = [Image.open(os.path.join(frame_images_dir, f'img_{fidx:05d}.jpg')) for fidx in frame_idx_list]
        if hflip:
            frame_images = [transforms.functional.hflip(image) for image in frame_images]
        frame_tensors = [self.transform(image) for image in frame_images]
        frame_tensors = torch.stack(frame_tensors, dim=0)
        return frame_tensors

    def __len__ (self):
        return len(self.sample_list)

    def __getitem__ (self, idx):
        video_name, sample_idx, sampled_timesteps, hflip = self.sample_list[idx]
        score_annot = self.video_info_dict[video_name]['sum_score']
        num_frame = self.video_info_dict[video_name]['num_frame']

        sampled_fidxs = []
        for timestep in sampled_timesteps:
            sampled_fidx = timestep
            if self.frames_per_timestep == 1:
                sampled_fidxs.append(sampled_fidx)
            elif self.frames_per_timestep > 1:
                half = self.frames_per_timestep // 2
                sampled_fidx = max(half, sampled_fidx)
                sampled_fidx = min(num_frame-half, sampled_fidx)
                st_fidx = sampled_fidx - half
                ed_fidx = sampled_fidx + half
                sampled_fidxs += list(range(st_fidx, ed_fidx))
        video_sample_tensors = self.__read_frame_tensors(video_name, sampled_fidxs, self.task, hflip)
        if self.frames_per_timestep > 1:
            video_sample_tensors = torch.stack(video_sample_tensors.split(self.frames_per_timestep, dim=0), dim=0)
            video_sample_tensors = video_sample_tensors.transpose(1,2).contiguous()
        
        return video_sample_tensors, score_annot, video_name, sample_idx, hflip
