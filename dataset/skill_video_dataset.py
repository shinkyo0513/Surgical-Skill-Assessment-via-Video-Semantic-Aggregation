import os

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

class SkillVideoDataset (Dataset):
    def __init__ (self, root_dir, is_train, task, split_type='SuperTrialOut', split_index=1, 
                        sampled_timestep_num=32, frames_per_timestep=16, frame_extraction_downsample=3,
                        return_skill_label='global', debug=False, balanced_train_sample=False,
                        noised_train_label=False, train_sample_augment=1, test_sample_augment=1,
                        return_position_masks=False, score_norm_bias=15, score_norm_weight=25):
        self.root_dir = root_dir
        self.is_train = is_train
        self.debug = debug
        self.return_position_masks = return_position_masks

        self.frame_extraction_downsample = frame_extraction_downsample
        self.balanced_train_sample = balanced_train_sample

        assert task in ['Knot_Tying', 'Needle_Passing', 'Suturing', 'Across']
        self.task = task
        
        assert split_type in ['SuperTrialOut', 'UserOut', 'FourFolds']
        self.split_type = split_type
        self.split_index = split_index
        if self.split_type == 'SuperTrialOut':
            assert split_index >= 1 and split_index <= 5
        elif self.split_type == 'UserOut':
            assert split_index >= 1 and split_index <= 8
        elif self.split_type == 'FourFolds':
            assert split_index >= 1 and split_index <= 4

        self.video_info_dict = self.__read_video_info()
        
        self.sampled_timestep_num = sampled_timestep_num
        self.frames_per_timestep = frames_per_timestep
        self.sample_mode = 'random' if self.is_train else 'last'
        self.train_sample_augment = 1
        self.test_sample_augment = 1
        self.noised_train_label = noised_train_label

        self.score_norm_bias = score_norm_bias
        self.score_norm_weight = score_norm_weight

        self.sample_list = self.__sample_video()

        self.transform = transforms.Compose([
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        ])

        self.return_skill_label = return_skill_label
        assert self.return_skill_label in ['global', 'self_proclaimed']

    def __read_taskwise_video_info (self, task_name):
        video_info_dict = {}

        video_dir = os.path.join(self.root_dir, 'jigsaws_annot', task_name, 'video')
        seg_annot_dir = os.path.join(self.root_dir, 'jigsaws_annot', task_name, 'transcriptions')

        skill_annot_dir = os.path.join(self.root_dir, 'jigsaws_annot', task_name, f'meta_file_{task_name}.txt')
        skill_annot_file = open(skill_annot_dir, 'r')
        skill_annot_dict = {}
        for line in skill_annot_file.readlines():
            # spc: self-proclaimed, ['E', 'I', 'N']; grs: global rating score, sum of all element scores
            # ele_scores: (1) Respect for tissue; (2) Suture/needle handling; (3) Time and motion; (4) Flow of operation; 
            #             (5) Overall performance; (6) Quality of final product.
            video_name, sum_scores, ele_scores = line.strip().split('\t\t')
            spc, grs = sum_scores.split('\t') 
            skill_annot_dict[video_name] = [spc, int(grs)] + [int(s) for s in ele_scores.split('\t')]
        skill_annot_file.close()

        # Read all videos' kinematics xyz data into a dictionary
        if self.return_position_masks:
            kine_dir = os.path.join(self.root_dir, 'jigsaws_annot', task_name, 'kinematics/AllGestures')
            kine_dict = {}
            for kine_file in os.listdir(kine_dir):
                video_name = kine_file.replace('.txt', '')
                kine_xyzs = []
                with open(os.path.join(kine_dir, kine_file)) as f:
                    for line in f.readlines():
                        numbers = [float(item) for item in line.strip().split()]
                        left_xyz = numbers[57:60]
                        right_xyz = numbers[38:41]
                        kine_xyzs.append((left_xyz, right_xyz))
                kine_dict[video_name] = kine_xyzs

        split_filename = 'Train.txt' if self.is_train else 'Test.txt'
        if self.split_type == 'FourFolds':
            split_filedir = os.path.join(self.root_dir, 'Experimental_setup', task_name, \
                                self.split_type, f'{self.split_index}_Out', 'itr_1', split_filename)
        else:
            split_filedir = os.path.join(self.root_dir, 'Experimental_setup', task_name, 'unBalanced', \
                                'GestureRecognition', self.split_type, f'{self.split_index}_Out', 'itr_1', split_filename)
                                
        with open(split_filedir, 'r') as split_file:
            lines = tqdm(split_file.readlines()) if self.debug else split_file.readlines()
            for line in lines:
                str1, str2 = line.strip().split(' '*11)
                video_name = str2.replace('.txt', '')
                _, start_frame, end_frame = str1.replace('.txt', '').replace(video_name, '').split('_')
                start_frame = int(start_frame)
                end_frame = int(end_frame)

                skill_annot = skill_annot_dict[video_name]

                seg_annot = []
                with open(os.path.join(seg_annot_dir, f'{video_name}.txt'), 'r') as seg_annot_file:
                    for line in seg_annot_file.readlines():
                        seg_st, seg_ed, seg_label = line.strip().split(' ')
                        seg_annot.append([int(seg_st), int(seg_ed), seg_label])

                # if video_name != 'Suturing_D004':
                video_info_dict[video_name] = [start_frame, end_frame, skill_annot, seg_annot]
                if self.return_position_masks:
                    kine_xyzs = kine_dict[video_name]
                    video_info_dict[video_name].append(kine_xyzs)

        return video_info_dict

    def __read_video_info (self):
        if self.task != 'Across':
            return self.__read_taskwise_video_info(self.task)
        else:
            video_info_dict = {}
            for task_name in ['Knot_Tying', 'Needle_Passing', 'Suturing']:
                video_info_dict.update(self.__read_taskwise_video_info(task_name))
            return video_info_dict

    def __sample_video (self):
        sample_list = []
        if self.balanced_train_sample and self.is_train:
            skill_bin_stat = [0, ] * 4
            avg_skill_score = 0.0
            for video_name, video_info in self.video_info_dict.items():
                skill_score = video_info[2][1]
                idx = (skill_score - 10) // 5   # bin_size = 5
                idx = 3 if idx == 4 else idx
                skill_bin_stat[idx] += 1    # 10~14; 15~19; 20~24; 25~30
                avg_skill_score += skill_score
            print('skill bin stat: ', skill_bin_stat)

            avg_skill_score /= sum(skill_bin_stat)
            print('ori avg skill score: ', avg_skill_score)

            bin_sample_num = self.train_sample_augment * sum(skill_bin_stat) / (len(skill_bin_stat) * np.array(skill_bin_stat))
            bin_sample_num = [int(round(num)) for num in bin_sample_num]
            print('bin sample num: ', bin_sample_num) 

            video_sample_num = {}
            avg_skill_score = 0.0
            for video_name, video_info in self.video_info_dict.items():
                skill_score = video_info[2][1]
                idx = (skill_score - 10) // 5
                idx = 3 if idx == 4 else idx
                video_sample_num[video_name] =  bin_sample_num[idx]
                avg_skill_score += video_sample_num[video_name] * skill_score
            avg_skill_score /= sum(list(video_sample_num.values()))
            print(f'avg skill score: ', avg_skill_score)

        for video_name, video_info in self.video_info_dict.items():
            start_frame, end_frame = video_info[:2]

            if self.frames_per_timestep > 1:
                start_frame += self.frames_per_timestep * self.frame_extraction_downsample
                end_frame -= self.frames_per_timestep * self.frame_extraction_downsample

            frame_num = end_frame - start_frame + 1
            if self.is_train:
                sample_num = video_sample_num[video_name] if self.balanced_train_sample else self.train_sample_augment
                for sample_idx in range(sample_num):
                    sampled_timesteps = LongRangeSample.long_range_sample(frame_num, self.sampled_timestep_num, self.sample_mode)
                    sample_list.append([video_name, sample_idx, [start_frame+sampled_timestep for sampled_timestep in sampled_timesteps]])
            else:
                sample_num = self.test_sample_augment
                if sample_num == 1:
                    sampled_timesteps = LongRangeSample.long_range_sample(frame_num, self.sampled_timestep_num, self.sample_mode)
                    sample_list.append([video_name, 0, [start_frame+sampled_timestep for sampled_timestep in sampled_timesteps]])
                elif sample_num == 2:
                    first_sampled_timesteps = LongRangeSample.long_range_sample(frame_num, self.sampled_timestep_num, 'first')
                    last_sampled_timesteps = LongRangeSample.long_range_sample(frame_num, self.sampled_timestep_num, 'last')
                    sample_list.append([video_name, 0, [start_frame+sampled_timestep for sampled_timestep in first_sampled_timesteps]])
                    sample_list.append([video_name, 1, [start_frame+sampled_timestep for sampled_timestep in last_sampled_timesteps]])
                else:
                    for sample_idx in range(sample_num):
                        sampled_timesteps = LongRangeSample.long_range_sample(frame_num, self.sampled_timestep_num, 'random')
                        sample_list.append([video_name, sample_idx, [start_frame+sampled_timestep for sampled_timestep in sampled_timesteps]])
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

    def __read_frame_tensors (self, video_name, frame_idx_list, task_name):
        frame_images_dir = os.path.join(self.root_dir, task_name, f'{task_name}_160x120', 
                                        video_name.split('_')[-1], 'frame')
        frame_tensors = [self.transform(Image.open(os.path.join(frame_images_dir, 
                                        f'{fidx:05d}.jpg'))) for fidx in frame_idx_list]
        frame_tensors = torch.stack(frame_tensors, dim=0)
        return frame_tensors

    def __gen_postion_masks_from_xyzs (self, video_kine_xyzs, sampled_frame_indices):
        trans_matrix = np.array([[-1250, 0, 0, 165], [0, 1800, -356, -20]])

        mask_tensors = []
        for fidx in sampled_frame_indices:
            kine_xyz_idx = min(fidx * 3, len(video_kine_xyzs)-1)
            # Get image coordinates from the kinematics data
            kine_xyz = np.array(video_kine_xyzs[kine_xyz_idx])  # 2x3
            kine_xyz_expand = np.concatenate([kine_xyz, np.ones((2, 1))], axis=1)   # 2x4
            img_coords = np.round(np.dot(trans_matrix, kine_xyz_expand.T))     # 2x2 (2 x num_points)
            img_coords -= np.array([[24, 24], [4, 4]])  # coordinations correction because of central crop
            left_img_coord = img_coords[:, 0]
            right_img_coord = img_coords[:, 1]

            mask = np.zeros((14, 14))
            left_map_coord_x, left_map_coord_y = np.round(np.array(left_img_coord) / 8).astype(np.int)
            right_map_coord_x, right_map_coord_y = np.round(np.array(right_img_coord) / 8).astype(np.int)
            for x_offset in [-1, 0, 1]:
                for y_offset in [-1, 0, 1]:
                    if x_offset * y_offset == 0:
                        left_put_y = 13 - min(max(left_map_coord_y + y_offset, 0), 13)
                        left_put_x = min(max(left_map_coord_x + x_offset, 0), 13)
                        mask[(left_put_y, left_put_x)] = 1.0

                        right_put_y = 13 - min(max(right_map_coord_y + y_offset, 0), 13)
                        right_put_x = min(max(right_map_coord_x + x_offset, 0), 13)
                        mask[(right_put_y, right_put_x)] = 1.0
            # mask = transform.resize(mask, (112, 112), order=3)
            mask = filters.gaussian(mask, 0.04*max(mask.shape))
            mask -= mask.min()
            mask /= mask.max()

            mask_tensors.append(torch.from_numpy(mask))
        mask_tensors = torch.stack(mask_tensors, dim=0) # Lx14x14
        return mask_tensors

    def __len__ (self):
        return len(self.sample_list)

    def __getitem__ (self, idx):
        video_name, sample_idx, sampled_timesteps = self.sample_list[idx]
        # skill_annot, seg_annot, video_frames = self.video_info_dict[video_name][2:]
        skill_annot, seg_annot = self.video_info_dict[video_name][2:4]

        sampled_fidxs = []
        for timestep in sampled_timesteps:
            sampled_fidx = int(timestep/self.frame_extraction_downsample)
            if self.frames_per_timestep == 1:
                sampled_fidxs.append(sampled_fidx)
            elif self.frames_per_timestep > 1:
                st_fidx = sampled_fidx - self.frames_per_timestep // 2
                ed_fidx = sampled_fidx + self.frames_per_timestep // 2
                sampled_fidxs += list(range(st_fidx, ed_fidx))

        task_name = video_name[:-5]
        video_sample_tensors = self.__read_frame_tensors(video_name, sampled_fidxs, task_name)
        if self.frames_per_timestep > 1:
            video_sample_tensors = torch.stack(video_sample_tensors.chunk(self.frames_per_timestep, dim=0), dim=2)

        if self.return_skill_label == 'global':
            skill_score = skill_annot[1]

        if self.noised_train_label and self.is_train:
            noise = random.choice([-1, 0, 0, 1])
            skill_score = max(min(skill_score + noise, 30), 0)

        if self.score_norm_bias != 0 and self.score_norm_weight != 1:
            skill_score = (skill_score - self.score_norm_bias) / self.score_norm_weight

        if self.return_position_masks:
            video_kine_xyzs = self.video_info_dict[video_name][4]
            position_mask_tensors = self.__gen_postion_masks_from_xyzs(video_kine_xyzs, sampled_fidxs)
            return video_sample_tensors, skill_score, video_name, sample_idx, position_mask_tensors
        else:
            return video_sample_tensors, skill_score, video_name, sample_idx
