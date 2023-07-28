import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                    save_dir='.',
                    use_infov_mask=True,
                    use_lidar_mask=False,
                    use_image_mask=True,
                    FREE_LABEL=23,
                    point_cloud_range = [0, -40, -5., 40, 40, 7.8],
                    voxel_size = [0.4, 0.4, 0.4],
                    CLASS_NAMES = [
                        'GO',
                        'TYPE_VEHICLE', "TYPE_BICYCLIST", "TYPE_PEDESTRIAN", "TYPE_SIGN",
                        'TYPE_TRAFFIC_LIGHT', 'TYPE_POLE', 'TYPE_CONSTRUCTION_CONE', 'TYPE_BICYCLE', 'TYPE_MOTORCYCLE',
                        'TYPE_BUILDING', 'TYPE_VEGETATION', 'TYPE_TREE_TRUNK', 
                        'TYPE_ROAD', 'TYPE_WALKABLE',
                        'TYPE_FREE',
                    ],

                 ):
        self.save_dir = save_dir
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = len(CLASS_NAMES)
        self.FREE_LABEL = FREE_LABEL
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.CLASS_NAMES = CLASS_NAMES
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.class_voxel_count_pred = {}
        self.class_voxel_count_gt = {}

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample) 
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten()) 
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self,semantics_pred,semantics_gt,mask_infov=None,mask_lidar=None,mask_camera=None):
        self.cnt += 1
        if mask_infov is not None: mask_infov = mask_infov.astype(bool)
        if mask_lidar is not None: mask_lidar = mask_lidar.astype(bool)
        if mask_camera is not None: mask_camera = mask_camera.astype(bool)
        semantics_gt[semantics_gt==self.FREE_LABEL] = self.num_classes - 1
        semantics_pred[semantics_pred==self.FREE_LABEL] = self.num_classes - 1
        mask = np.ones_like(semantics_gt, dtype=np.bool)
        if self.use_infov_mask:
            mask = np.logical_and(mask, mask_infov)                  
        if self.use_lidar_mask:
            mask = np.logical_and(mask, mask_lidar)
        if self.use_image_mask:
            mask = np.logical_and(mask, mask_camera)
        masked_semantics_gt = semantics_gt[mask]
        masked_semantics_pred = semantics_pred[mask]

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist
        for idx in np.unique(masked_semantics_gt):
            if idx not in self.class_voxel_count_gt:
                self.class_voxel_count_gt[idx] = 0
            self.class_voxel_count_gt[idx] += (masked_semantics_gt==idx).sum()
        for idx in np.unique(masked_semantics_pred):
            if idx not in self.class_voxel_count_pred:
                self.class_voxel_count_pred[idx] = 0
            self.class_voxel_count_pred[idx] += (masked_semantics_pred==idx).sum()

    def print(self,):
        # mIoU_avg = _mIoU / num_samples
        hist = self.hist
        cnt = self.cnt
        CLASS_NAMES = self.CLASS_NAMES
        class_voxel_count_gt = self.class_voxel_count_gt
        class_voxel_count_pred = self.class_voxel_count_pred

        mIoU = self.per_class_iu(hist)
        print(f'===> per class IoU of {cnt} samples:')
        for ind_class in range(self.num_classes):
            class_name = CLASS_NAMES[ind_class]
            print(f'===> class {class_name} IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            
        print(f'===> mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU[:-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))
        print(f'===> gt/pred voxel count of {cnt} samples: ')
        keysList = list(class_voxel_count_gt.keys())
        keysList.sort()
        for key in keysList:
            class_name = CLASS_NAMES[key]
            if key not in class_voxel_count_gt: class_voxel_count_gt[key] = 0
            if key not in class_voxel_count_pred: class_voxel_count_pred[key] = 0
            print('     class: {},      gt cout: {},    pred cout: {}'.format(class_name, class_voxel_count_gt[key], class_voxel_count_pred[key]))
            
    def eval_from_file(self, pred_path, gt_path, load_interval=1):
        gts_dict = {}
        for scene in os.listdir(gt_path):
            for frame in os.listdir(os.path.join(gt_path, scene)):
                scene_token = frame
                gts_dict[scene_token] = os.path.join(gt_path, scene, frame, 'labels.npz')
        print('number of gt samples = {}'.format(len(gts_dict)))

        dirs_list = [
            "work_dirs/bevformer_base_occ_conv3d_waymo_allgift/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_ambiguous/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_noohem/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_no_cross_atten/results_epoch8/",
        ]
        pred_path = dirs_list[2] # "work_dirs/bevformer_base_occ_conv3d_waymo_ambiguous/results_epoch6/"
        union_files = set(os.listdir(dirs_list[0]))
        print(pred_path)
        for _dir in dirs_list:
            union_files = union_files.intersection(set(os.listdir(_dir)))

        preds_dict = {}
        for file in os.listdir(pred_path)[::load_interval]:
            if file not in union_files: continue
            if '.npz' not in file: continue

            scene_token = file.split('.npz')[0]
            preds_dict[scene_token] = os.path.join(pred_path, file)
        print('number of pred samples = {}'.format(len(preds_dict)))
        return gts_dict, preds_dict
    
    def __call__(self):
        gts_dict, preds_dict = self.eval_from_file()
        # _mIoU = 0.        
        for scene_token in tqdm(preds_dict.keys()):
            cnt += 1
            # gt = np.load(gts_dict[scene_token])
            # bs,H,W,Z
            self.add_batch(semantics_pred, semantics_gt, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera)
            # _mIoU += _miou

        results = self.print()
        return results


class Metric_FScore():
    def __init__(self,
                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8
        raise NotImplementedError

    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))


