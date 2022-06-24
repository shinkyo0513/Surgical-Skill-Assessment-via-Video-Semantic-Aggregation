import os
from os.path import join, isdir, isfile

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ImageShow import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import colorsys

def generate_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors)*255.
    return colors_np

def show_assign_on_image(img, assign_hard, color_map):
    # img: 3x112x112, np.uint8, 0~255
    # assign_hard: 7x7
    # generate the numpy array for colors
    # colors = generate_colors(num_parts)

    # coefficient for blending
    coeff = 0.4

    img_h, img_w = img.shape[1:]
    assign_h, assign_w = assign_hard.shape
    if assign_h != img_h or assign_w != img_w:
        resized_assign_hard = cv2.resize(assign_hard, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # blending by each pixel
    res = np.ones_like(img)
    for i in range(img_h):
        for j in range(img_w):
            assign_ij = resized_assign_hard[i][j]
            res[:, i, j] = (1-coeff) * img[:, i, j] + coeff * color_map[assign_ij]

    # save the resulting img
    res_uint = np.uint8(res)
    return res_uint

def plot_video_res (video_tensors, part_assigns, part_att, show_att=True, title=None, save_path=None, save_separately=False):
    # video_tensors: Lx3x112x112
    # part_assigns: Lx3x7x7, part_att: L x 3
    num_timesteps = part_assigns.shape[0]
    num_parts = part_assigns.shape[1]
    color_map = generate_colors(num_parts)
    assign_h, assign_w = part_assigns.shape[2:]

    hard_assigns_np = part_assigns.argmax(dim=1).numpy()    # np, Lx7x7

    img_t = video_tensors.shape[0]
    if img_t != num_timesteps:
        t_times = max(round(img_t / num_timesteps), 1)
        video_imgs = torch.stack([video_tensors[t,...] for t in range(0, img_t, t_times)], dim=1).contiguous()  # 3xL'x112x112
        video_imgs = voxel_tensor_to_np(video_imgs)  # np, 0~1, 3xLx112x112
    else:
        video_imgs = voxel_tensor_to_np(video_tensors.transpose(0, 1))  # np, 0~1, 3xLx112x112
    video_imgs_uint = np.uint8(video_imgs * 255)

    assign_cover_imgs_uint = np.stack([show_assign_on_image(video_imgs_uint[:,t], hard_assigns_np[t], color_map) for t in range(num_timesteps)], axis=0)
    
    if show_att:
        # part_att_rltv = part_att / part_att.sum(dim=1).max().item()   # Lx3
        part_att_rltv = part_att
        part_att_rltv_np = part_att_rltv.unsqueeze(-1).expand(-1, -1, assign_w).numpy() # np, Lx3x7
        att_maps_np = np.take_along_axis(part_att_rltv_np, hard_assigns_np, axis=1)  # np, Lx7x7

        att_maps_np -= att_maps_np.min()
        if att_maps_np.max() > 0:
            att_maps_np /= att_maps_np.max()
        else:
            print(f'The maximum of the attention maps is less than 0.')

        att_cover_imgs_np = overlap_maps_on_voxel_np(video_imgs, att_maps_np).swapaxes(0,1)  # np, Lx3x7x7, no frame-wise norm
        att_cover_imgs_uint = np.uint8(att_cover_imgs_np * 255)

    if save_separately and save_path != None:
        separate_save_dir = os.path.splitext(save_path)[0]
        os.makedirs(separate_save_dir, exist_ok=True)

    # save plot imgs, assign_cover, att_cover
    num_subline = 3 if show_att else 2
    num_row = num_subline * ( (num_timesteps-1) // 8 + 1 )
    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_timesteps):
        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+1)
        img_np_show(video_imgs_uint[:,i])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*8*num_subline+i%8+8+1)
        img_np_show(assign_cover_imgs_uint[i])

        if save_separately:
            # print(video_imgs_uint[:,i].shape, assign_cover_imgs_uint[i].shape)
            video_img = Image.fromarray(video_imgs_uint[:,i].transpose(1,2,0))
            video_img.save(os.path.join(separate_save_dir, f'img_{i}.jpg'))
            assign_img = Image.fromarray(assign_cover_imgs_uint[i].transpose(1,2,0))
            assign_img.save(os.path.join(separate_save_dir, f'assign_{i}.jpg'))

        if show_att:
            plt.subplot(num_row, 8, (i//8)*8*3+i%8+16+1)
            img_np_show(att_cover_imgs_uint[i])

    if title != None:
        fig.suptitle(title, fontsize=14)

    if save_path != None:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)

        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    plt.close(fig)
