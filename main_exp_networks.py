import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import os
import glob
from os.path import join

import time
import copy
import random
from tqdm import tqdm
import numpy as np
from scipy import stats

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

from dataset.skill_video_dataset import SkillVideoDataset
from utils.ImageShow import *
from utils import PatchMatch
from utils import ShapingLoss
from utils import SmoothGrad
from utils import GradCAM
from group_visualize import plot_video_res

pt_save_root = os.path.join(proj_root, 'model_param')
exp_save_root = os.path.join(proj_root, 'group_exp_res')

def plot_video_exp_res (video_tensors, exp_results, title=None, save_path=None, save_separately=False):
    # video_tensors: Lx3x112x112
    # exp_results: Lx1x112x112
    num_timesteps = video_tensors.shape[0]
    assert num_timesteps == exp_results.shape[0]

    video_imgs = voxel_tensor_to_np(video_tensors.transpose(0, 1))  # np, 0~1, 3xLx112x112
    video_imgs_uint = np.uint8(video_imgs * 255)

    heatmaps = exp_results.squeeze(1).numpy()   # np, 0~1, Lx112x112
    overlaps = overlap_maps_on_voxel_np(video_imgs, heatmaps)   # np, 0~1, 3xLx112x112
    overlaps_uint = np.uint8(overlaps * 255)

    if save_separately and save_path != None:
        separate_save_dir = os.path.splitext(save_path)[0]
        os.makedirs(separate_save_dir, exist_ok=True)

    # save plot imgs, explanation heatmaps
    num_subline = 2
    num_row = num_subline * ( (num_timesteps-1) // 8 + 1 )
    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_timesteps):
        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+1)
        img_np_show(video_imgs_uint[:,i])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+8+1)
        img_np_show(overlaps_uint[:,i])

        if save_separately:
            video_img = Image.fromarray(video_imgs_uint[:,i].transpose(1,2,0))
            video_img.save(os.path.join(separate_save_dir, f'img_{i}.jpg'))
            exp_img = Image.fromarray(overlaps_uint[:,i].transpose(1,2,0))
            exp_img.save(os.path.join(separate_save_dir, f'exp_{i}.jpg'))

    if title != None:
        fig.suptitle(title, fontsize=14)

    if save_path != None:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)

        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    plt.close(fig)

def exp (args, split_index, device, save_label):
    print(f'--Task: {args.task}, Split Type: {args.val_split}, Split Index: {split_index} ...')
    frames_per_sample = 8 if '4layer' in args.extractor else 4
    num_samples = args.num_samples * frames_per_sample
    frames_per_timestep = 1

    from model_def.PartGroup_SkillNet3D import PartGroup_SkillNet3D
    model = PartGroup_SkillNet3D(args.num_parts, args.extractor, args.context, args.aggregate, 
                                args.avgpool_parts, args.scene_node, args.attention, args.multi_lstms, 
                                args.prepro, args.no_pastpro, args.simple_pastpro,
                                final_score_bias=15, final_score_weight=25).to(device)

    video_datasets = {x: SkillVideoDataset(ds_root, x=='train', task=args.task, debug=False,
                                            split_type=args.val_split, split_index=split_index,
                                            frames_per_timestep=frames_per_timestep,
                                            sampled_timestep_num=num_samples, 
                                            balanced_train_sample=args.balanced_train_sample,
                                            noised_train_label=args.noised_train_label,
                                            train_sample_augment=args.train_sample_augment,
                                            test_sample_augment=args.test_sample_augment,
                                            return_position_masks=False,
                                            score_norm_bias=0, score_norm_weight=1,
                        ) for x in ['val']}
    print({x: 'Num of clips:{}'.format(len(video_datasets[x])) for x in ['val']})
    dataloaders = {x: DataLoader(video_datasets[x], batch_size=args.batch_size, shuffle=(x=='train'),
                        num_workers=128) for x in ['val']}

    if multi_gpu:
        print('Use', num_devices, 'GPUs!')
        model = nn.DataParallel(model, device_ids=list(range(num_devices)))

    if args.read_checkpoint:
        checkpoint_dir = os.path.join(pt_save_root, f"{save_label}_{args.split_index}.pt")
        pretrain_wgt = torch.load(checkpoint_dir)
        model_wgt = model.state_dict()
        for pname in model_wgt.keys():
            if 'extractor' in pname:
                model_wgt[pname] = pretrain_wgt[pname]
        model.load_state_dict(model_wgt)

    exp_save_dir = os.path.join(exp_save_root, f"{save_label}_{args.split_index}")

    since = time.time()
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
            if args.context == 'lstm':
                model.rnn.train()

        # Iterate over data.
        for samples in tqdm(dataloaders[phase]):
            inputs = samples[0].to(device)  # BxLx3x112x112
            gt_scores = samples[1].to(device, dtype=torch.float) # B
            batch_video_names = samples[2]

            exp_labels = torch.ones_like(gt_scores).to(device, dtype=torch.float)   # B
            # exp_outputs = SmoothGrad.smooth_grad(inputs, exp_labels, model, device, variant='square')  # BxLx1x112x112
            exp_outputs = GradCAM.grad_cam(inputs, exp_labels, model, device)  # BxLx1x112x112

            for bidx in range(inputs.shape[0]):
                plot_video_exp_res(
                        inputs[bidx].detach().cpu(), 
                        exp_outputs[bidx].detach().cpu(), 
                        title = f'{batch_video_names[bidx]}',
                        save_path = os.path.join(exp_save_dir, 
                                    f'{batch_video_names[bidx]}.jpg'),
                        save_separately = args.save_separately
                )
    print()

    time_elapsed = time.time() - since
    return 

if __name__ == '__main__':

    randomseed = 0
    random.seed(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    torch.cuda.manual_seed_all(randomseed)

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, choices=['resnet_lstm', 'r2p1d_lstm_dist', 
    #                     'r3d_lstm_dist', 'r3d_bilstm_dist', 'r3d_gcn_dist'])
    parser.add_argument("--extractor", type=str, choices=['r3d', 'r2p1d', 'r3d_4layer'])
    parser.add_argument("--context", type=str, choices=['lstm', 'bilstm', 'gcn', 'none'])
    parser.add_argument("--aggregate", type=str, choices=['mean', 'avgpool', 'final', 'lstm'])

    parser.add_argument("--val_split", type=str, default='SuperTrialOut', 
                        choices=['SuperTrialOut', 'UserOut', 'FourFolds'])
    parser.add_argument("--task", type=str, default='Suturing', 
                        choices=['Suturing', 'Knot_Tying', 'Needle_Passing', 'Across'])
    parser.add_argument("--split_index", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--multi_gpu", action='store_true')
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--schedule_step", type=int, default=20)
    parser.add_argument("--scene_node", action='store_true')
    parser.add_argument("--num_parts", type=int, default=3)
    parser.add_argument("--no_pastpro", action='store_true')
    parser.add_argument("--tconsist_weight", type=float, default=0)
    parser.add_argument("--shaping_weight", type=float, default=0)
    parser.add_argument("--position_regu_weight", type=float, default=0)
    parser.add_argument("--heatmap_regu_weight", type=float, default=0)
    parser.add_argument("--assign_supp_weight", type=float, default=0)

    # Unfrequently used arguments
    parser.add_argument("--attention", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--avgpool_parts", action='store_true')
    parser.add_argument("--multi_lstms", action='store_true')
    parser.add_argument("--prepro", action='store_true')
    parser.add_argument("--simple_pastpro", action='store_true')
    parser.add_argument("--rolling_train", action='store_true')
    parser.add_argument("--freeze_extractor", action='store_true')
    parser.add_argument("--freeze_half_extractor", action='store_true')
    parser.add_argument("--freeze_central", action='store_true')
    parser.add_argument("--init_extractor", action='store_true')
    parser.add_argument("--tconsist_start_from", type=int, default=0)
    parser.add_argument("--train_sample_augment", type=int, default=1)
    parser.add_argument("--test_sample_augment", type=int, default=1)
    parser.add_argument("--balanced_train_sample", action='store_true')
    parser.add_argument("--noised_train_label", action='store_true')

    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--save_separately", action='store_true')
    parser.add_argument("--extra_label", type=str, default=None)

    parser.add_argument("--read_checkpoint", action='store_true')
    args = parser.parse_args()

    model_type = f'{args.extractor}_{args.context}_{args.aggregate}'
    save_label = f"Skill{args.task}_{args.val_split}_{model_type}_{args.num_parts}parts"
    if args.no_pastpro:
        save_label += "_np"
    if args.simple_pastpro:
        save_label += "_sp"
    if args.multi_lstms:
        save_label += "_ml"
    if args.attention:
        save_label += "_att"
    if args.rolling_train:
        save_label += "_rt"
    if args.init_extractor:
        save_label += "_ie"
    if args.freeze_extractor:
        save_label += "_fe"
    if args.freeze_half_extractor:
        save_label += "_fhe"
    if args.freeze_central:
        save_label += "_fc"
    if args.scene_node:
        save_label += "_sn"
    if args.avgpool_parts:
        save_label += "_ap"

    if args.shaping_weight > 0:
        save_label += f"_shape{args.shaping_weight}"
    if args.tconsist_weight > 0:
        save_label += f"_tcons{args.tconsist_weight}"
    if args.position_regu_weight > 0:
        save_label += f"_posregu{args.position_regu_weight}"
    if args.heatmap_regu_weight > 0:
        save_label += f"_htmpregu{args.heatmap_regu_weight}"
    if args.assign_supp_weight > 0:
        save_label += f"_assgsup{args.assign_supp_weight}"
    if args.tconsist_start_from > 0:
        save_label += f"_ts{args.tconsist_start_from}"
    save_label += f"_lr{args.learning_rate}"
    if args.extra_label != None:
        save_label += f"_{args.extra_label}"
    print(save_label)

    multi_gpu = args.multi_gpu
    num_devices = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hist = exp(args, args.split_index, device, save_label)

