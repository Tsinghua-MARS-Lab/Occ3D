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
from mmdet3d.datasets import DATASETS #really fucked up for not adding '3d'
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import copy
from mmcv.parallel import DataContainer as DC
import random
from mmdet3d.core.bbox import get_box_type

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
# from .zltvis import save_temporal_frame

from .zltwaymo import CustomWaymoDataset
from .occ_metrics import Metric_FScore,Metric_mIoU
from tqdm import tqdm


@DATASETS.register_module()
class CustomWaymoDataset_T(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                #  use_interval=5,
                #  length=3000,
                 history_len = 1, skip_len = 0,
                 withimage=True,
                 pose_file=None,
                 **kwargs):
        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all
        # wtf need set attr first?
        self.length_wtf = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.skip_len = skip_len
        self.withimage = withimage
        self.load_interval_wtf=load_interval
        self.length = self.length_wtf
        super().__init__(*args, **kwargs)
        #assert self.length == len(self.data_infos_full)

    def __len__(self):
        return self.length_wtf // self.load_interval_wtf

    def __getitem__(self, idx):
        # idx = 80 # DEBUG_TMP
        if self.test_mode:
            return self.prepare_test_data(idx)
        # import time
        # _ = time.time()
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # print('dataloading cost: {} ms'.format(time.time()-_))
            return data

    def prepare_train_data(self, index):
        # [T, T-1]
        index = index * self.load_interval_wtf
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
            # data_queue.insert(0, copy.deepcopy(example))
        if self.filter_empty_gt and \
                (data_queue[0] is None):  # or ~(data_queue[-1]['gt_labels_3d']._data != -1).any()):
            return None
        if self.withimage:
            # data_queue: T-len+1, T
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
        # oldname='queue'
        # np.save('debug/debug_temporal1/'+oldname, queue)
        prev_scene_token=None
        prev_pos = None
        prev_angle = None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        # ego2global = queue[-1]['img_metas'].data['pose']
        # print('ego2global',type(ego2global.data),ego2global.data.shape)
        # print()
        # print('len',len(queue))
        # print()
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
            # global2ego_old = np.linalg.inv(metas_map[i]['pose'].data)
            # ego2img_old_rts = []
            # for ego_old2img_old in metas_map[i]['lidar2img']:
            #     ego2img_old = ego_old2img_old @ global2ego_old @ ego2global.data.numpy() #@pt_ego
            #     ego2img_old_rts.append(ego2img_old)
            # metas_map[i]['lidar2img'] = ego2img_old_rts
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        # breakpoint()
        # save_temporal_frame(queue)
        # name = 'queue_union'
        # np.save('debug/debug_temporal1/'+name, queue)
        # breakpoint()
        return queue


    def get_data_info(self, index):
        # index = 248
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

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []

            # # load calibration for all 5 images.
            # calib_path = img_filename.replace('image_0', 'calib').replace('.png', '.txt')
            # Tr_velo_to_cam_list = []
            # with open(calib_path, 'r') as f:
            #     lines = f.readlines()
            # for line_num in range(6, 6 + self.num_views):
            #     trans = np.array([float(info) for info in lines[line_num].split(' ')[1:13]]).reshape(3, 4)
            #     trans = np.concatenate([trans, np.array([[0., 0., 0., 1.]])], axis=0).astype(np.float32)
            #     Tr_velo_to_cam_list.append(trans)
            # assert np.allclose(Tr_velo_to_cam_list[0], info['calib']['Tr_velo_to_cam'].astype(np.float32))

            for idx_img in range(self.num_views):
                # rect = info['calib']['R0_rect'].astype(np.float32)
                # # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
                # Trv2c = Tr_velo_to_cam_list[idx_img]
                # P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
                # lidar2img = P0 @ rect @ Trv2c
                pose = self.pose_all[scene_idx][frame_idx][idx_img]
                lidar2img = pose['intrinsics'] @ np.linalg.inv(pose['sensor2ego'])
                intrinsics = pose['intrinsics']
                sensor2ego = pose['sensor2ego']

                # wtf attention
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

        # input_dict['pose'] = info['pose']
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos


        # rotation_matrix = input_dict['pose'][:3,:3]
        # translation = input_dict['pose'][:3,3]

        # rotation = Quaternion(matrix=rotation_matrix)

        # print('shape test' * 1000)
        # print()
        # print(rotation_matrix,translation,rotation,type(rotation_matrix))

        can_bus=np.zeros(9)

        # can_bus[:3] = translation
        # can_bus[3:7] = rotation
        # patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # if patch_angle < 0:
        #     patch_angle += 360
        # can_bus[-2] = patch_angle / 180 * np.pi
        # can_bus[-1] = patch_angle

        input_dict['can_bus'] = can_bus

        # # load occ info
        # input_dict['gt_masks'] = np.fromfile(self.gt_path + f'{str(index).zfill(7)}/' + 'gt_masks.bin', dtype=np.int32).reshape((-1, 376,276, 16))
        # input_dict['gt_semantic_seg'] = np.fromfile(self.gt_path + f'{str(index).zfill(7)}/' + 'gt_semantic_seg.bin', dtype=np.int32).reshape((-1, 376,276, 16))
        # with open(f'{self.gt_path}' + f'{str(index).zfill(7)}/' + 'bboxes.pkl', 'rb') as f:
        #     input_dict['gt_bboxes_occ'] = pickle.load(f)
        #
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
        # print('bbox test'*100)
        # print('gt_bboxes_3d',gt_bboxes_3d.shape)
        # convert gt_bboxes_3d to velodyne coordinates

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))


        gt_bboxes = annos['bbox']

        # TODO:  generate gt_masks gt_labels, gt_semantic_labels, bbox
        # time_start = time.time()
        # pc = torch.load('/home/txy/ylf/mmdetection3d/data/waymo_subset/gt_test/points_0000049.pt')
        # pc_labels = torch.load('/home/txy/ylf/mmdetection3d/data/waymo_subset/gt_test/semantic_0000049.pt')
        # ins_id = torch.load('/home/txy/ylf/mmdetection3d/data/waymo_subset/gt_test/instanceid_0000049.pt')
        # with open('/home/txy/ylf/mmdetection3d/data/waymo_subset/gt_test/file2ins.pkl', 'rb') as f:
        #     ins_id_mapping = pickle.load(f)
        # voxel_size = 0.4
        # voxel = torch.div(pc, voxel_size, rounding_mode='floor').int()
        # occ_voxel, idx = np.unique(voxel.numpy(), axis=0, return_index=True)
        # voxel_labels = pc_labels[idx]
        # voxel_ins_labels = ins_id[idx]
        # #panop_labels = panop_labels[:, idx]
        # # filter voxel by range
        # pc_range = [(-35, 75), (-75, 75), (-2.2, 4.2)]# -2.2, 4.2
        # voxel_range = [(int(np.ceil(x / 0.4)), int(np.ceil(y / 0.4))) for x, y in pc_range]
        # filter_voxel = reduce(np.logical_and,
        #                       [np.logical_and(occ_voxel[:, i] >= voxel_range[i][0], occ_voxel[:, i] < voxel_range[i][1])
        #                        for i in range(0, 3)])
        # filter_voxel_idx = np.where(filter_voxel == False)[0]
        # voxel = np.delete(occ_voxel, filter_voxel_idx, axis=0)
        # voxel_labels = np.delete(voxel_labels.numpy(), filter_voxel_idx)
        # voxel_ins_labels = np.delete(voxel_ins_labels.numpy(), filter_voxel_idx)
        # # gt_semantic_seg
        # voxel_labels = voxel_labels - 1
        # gt_semantic_seg = np.full((voxel_range[0][1] - voxel_range[0][0], voxel_range[1][1] - voxel_range[1][0], voxel_range[2][1] - voxel_range[2][0]), 16, dtype=np.int32)
        # gt_semantic_seg[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = voxel_labels
        # # gt_masks for each instances
        # gt_masks = np.zeros((0, voxel_range[0][1] - voxel_range[0][0], voxel_range[1][1] - voxel_range[1][0], voxel_range[2][1] - voxel_range[2][0]), dtype=np.int32)
        # gt_labels = []
        # gt_boxes_occ = []
        # for idx, id in enumerate(annos['id']):
        #     ins_label = ins_id_mapping[id]
        #     ins_idx = np.where(voxel_ins_labels==ins_label)[0]
        #     if ins_idx.shape[0] == 0: continue
        #     seman_label = np.argmax(np.bincount(voxel_labels[ins_idx]))
        #     if seman_label >= 13: continue
        #     gt_mask = np.empty((1, voxel_range[0][1] - voxel_range[0][0], voxel_range[1][1] - voxel_range[1][0], voxel_range[2][1] - voxel_range[2][0]), dtype=np.int32)
        #     gt_mask.fill(16000)
        #     gt_mask[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = ins_label + seman_label * 1000
        #     gt_masks = np.concatenate((gt_masks, gt_mask), axis=0)
        #     gt_labels.append(seman_label)
        #     gt_boxes_occ.append(gt_bboxes_3d[idx])
        #
        # gt_semantic_seg.tofile('/home/txy/occ_test/gt_semantic_seg.bin')
        # gt_masks.tofile('/home/txy/occ_test/gt_masks.bin')
        # with open('/home/txy/occ_test/bboxes.pkl', 'wb') as f:
        #     bboxes = {}
        #     bboxes['gt_boxes_occ'] = gt_boxes_occ
        #     bboxes['gt_labels'] = gt_labels
        #     pickle.dump(bboxes, f)
        #
        # time_end = time.time()
        # print('time cost', time_end - time_start, 's')
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
                # info = self.data_infos[index]
                # filename=info['pts_filename'].split('/')[-1].split('.')[0]
                # occ_gt = np.load(os.path.join(self.data_root, filename,))
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



