import mmcv
import numpy as np
import os
import tempfile
import torch
import pickle
from functools import reduce
import time
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet3d.datasets import DATASETS 
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import copy
from mmcv.parallel import DataContainer as DC
import random
from mmdet3d.core.bbox import get_box_type

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)


from .waymo_dataset import CustomWaymoDataset
from .occ_metrics import Metric_FScore,Metric_mIoU
from tqdm import tqdm


@DATASETS.register_module()
class CustomWaymoDataset_T(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                 history_len = 1, skip_len = 0,
                 withimage=True,
                 pose_file=None,
                 **kwargs):
        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all

        self.length = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.skip_len = skip_len
        self.withimage = withimage
        self.load_interval=load_interval
        super().__init__(*args, **kwargs)
        #assert self.length == len(self.data_infos_full)

    def __len__(self):
        return self.length // self.load_interval

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        # [T, T-1]
        index = index * self.load_interval
        if self.history_len == 0:
            idx_list = [index]
        else:
            idx_list = list(range(index - self.history_len, index))
            random.shuffle(idx_list)
            idx_list = sorted(idx_list[1:])
            idx_list.append(index)
        data_queue = []
        scene_id = None
        for i in idx_list:
            i = max(0, i)
            if i == idx_list[-1]:
                input_dict = self.get_data_info(i)
            else:
                input_dict = self.get_data_info(i)
            if scene_id == None: scene_id = input_dict['sample_idx'] // 1000
            if input_dict is None: return None

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue.append(example)
            
        if self.filter_empty_gt and \
                (data_queue[0] is None):  
            return None
        if self.withimage:

            return self.union2one(data_queue)
        else:
            return data_queue[-1]


    def union2one(self, queue):
        """
        input: queue: dict of [T-len+1, T], containing data_info
        convert sample queue into one single sample.
        calculate transformation from ego_now to image_old
        note that we dont gather gt objects of previous frames
        """
        
        prev_scene_token=None
        prev_pos = None
        prev_angle = None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['sample_idx']//1000 != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['sample_idx'] // 1000
                metas_map[i]['scene_token']= prev_scene_token
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0

            else:
                metas_map[i]['scene_token'] = prev_scene_token
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        
        return queue


    def get_data_info(self, index):

        info = self.data_infos_full[index]
        
        sample_idx = info['image']['image_idx']
        scene_idx = sample_idx % 1000000 // 1000
        frame_idx = sample_idx % 1000000 % 1000
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c


        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []

        

            for idx_img in range(self.num_views):
                
                pose = self.pose_all[scene_idx][frame_idx][idx_img]
                lidar2img = pose['intrinsics'] @ np.linalg.inv(pose['sensor2ego'])
                intrinsics = pose['intrinsics']
                sensor2ego = pose['sensor2ego']

                if idx_img == 2: 
                    image_paths.append(img_filename.replace('image_0', f'image_3'))
                elif idx_img == 3: 
                    image_paths.append(img_filename.replace('image_0', f'image_2'))
                else:
                    image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))

                lidar2img_rts.append(lidar2img)
                intrinsics_rts.append(intrinsics)
                sensor2ego_rts.append(sensor2ego)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
        )
        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
            input_dict['cam_intrinsic'] = intrinsics_rts
            input_dict['sensor2ego'] = sensor2ego_rts

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        can_bus=np.zeros(9)
        input_dict['can_bus'] = can_bus


        return input_dict


    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)


        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))


        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def evaluate(self, occ_results, metric='mIoU', runner=None, **eval_kwargs):
        def eval(occ_eval_metrics):
            print('\nStarting Evaluation...')
            for index, occ_result in enumerate(tqdm(occ_results)):
                voxel_semantics = occ_result['voxel_semantics']
                voxel_semantics_preds = occ_result['voxel_semantics_preds']
                mask_infov = occ_result.get("mask_infov", None)
                mask_lidar = occ_result.get("mask_lidar", None)
                mask_camera = occ_result.get("mask_camera", None)

                occ_eval_metrics.add_batch(voxel_semantics_preds, voxel_semantics, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera)
            occ_eval_metrics.print()

        if "mIoU" in metric:
            occ_eval_metrics = Metric_mIoU()
            eval(occ_eval_metrics)
        elif "FScore" in metric:
            occ_eval_metrics = Metric_FScore()
            eval(occ_eval_metrics)
        else:
            raise NotImplementedError        



