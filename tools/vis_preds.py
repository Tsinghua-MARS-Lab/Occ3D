import os
import pickle as pkl
import numpy as np
import torch
import argparse
from tools import vis_occ

VOXEL_SIZE=[0.1, 0.1, 0.2]
POINT_CLOUD_RANGE=[-80, -80, -5, 80, 80, 7.8]
SPTIAL_SHAPE=[1600, 1600, 64]
TGT_VOXEL_SIZE=[0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE=[-40, -40, -1, 40, 40, 5.4]

FREE_LABEL = 23


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='path to saved results')
    args = parser.parse_args()
    interval = 10
    voxel_size = TGT_VOXEL_SIZE
    with open(args.file_path ,'rb') as f:
        results = pkl.load(f)
    
    for result in results[::interval]:
        voxel_semantics = result['voxel_semantics']
        voxel_semantics_preds = result['voxel_semantics_preds']
        mask_infov = result['mask_infov']
        mask_lidar = result['mask_lidar']
        mask_camera = result['mask_camera']

        voxel_semantics = np.squeeze(voxel_semantics, axis=0)
        voxel_semantics_preds = np.squeeze(voxel_semantics_preds, axis=0)
        mask_infov = np.squeeze(mask_infov, axis=0)
        mask_lidar = np.squeeze(mask_lidar, axis=0)
        mask_camera = np.squeeze(mask_camera, axis=0)

        voxel_label_vis = voxel_semantics
        voxel_show = np.logical_and(mask_camera, voxel_semantics!=FREE_LABEL)
        vis = vis_occ.main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=None,
                            offset=[voxel_label_vis.shape[0] * voxel_size[0] * 1.2 * 0, 0, 0])

        voxel_label_vis = voxel_semantics_preds
        voxel_show = np.logical_and(mask_camera, voxel_semantics_preds!=FREE_LABEL)
        vis = vis_occ.main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[voxel_label_vis.shape[0] * voxel_size[0] * 1.2 * 1, 0, 0])

        voxel_label_vis = voxel_semantics
        voxel_show = np.logical_and(mask_camera, voxel_semantics_preds!=voxel_semantics)
        vis = vis_occ.main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[voxel_label_vis.shape[0] * voxel_size[0] * 1.2 * 2, 0, 0])
        vis.run()
        vis.poll_events()
        vis.update_renderer()

        del vis