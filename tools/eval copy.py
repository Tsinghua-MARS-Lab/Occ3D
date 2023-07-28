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
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False, bboxes=None, linesets=None,scene_idx=None,frame_idx=None) -> None:
    import open3d as o3d
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='f1 score')

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    #opt.background_color = np.asarray([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    vis.add_geometry(pcd)
    vis.run()

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
                 gt_path,
                 pred_path,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):

        self.gt_path = gt_path
        self.pred_path = pred_path
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
    
    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        empty classes:0
        non-empty class: 1-16
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
    
    def get_dicts(self, ):
        gts_dict = {}
        for scene in os.listdir(self.gt_path):
            for frame in os.listdir(os.path.join(self.gt_path, scene)):
                scene_token = frame
                gts_dict[scene_token] = os.path.join(self.gt_path, scene, frame, 'labels.npz')
        print('number of gt samples = {}'.format(len(gts_dict)))

        preds_dict = {}
        for file in os.listdir(pred_path):
            scene_token = file.split('.npz')[0]
            preds_dict[scene_token] = os.path.join(self.pred_path, file)
        print('number of pred samples = {}'.format(len(preds_dict)))
        return gts_dict, preds_dict
    
    def __call__(self):
        gts_dict, preds_dict = self.get_dicts()

        # assert set(gts_dict.keys()) == set(preds_dict.keys()) # DEBUG_TMP
        num_samples = len(preds_dict.keys())
        # _mIoU = 0.
        cnt = 0
        hist = np.zeros((self.num_classes, self.num_classes))
        
        for scene_token in tqdm(preds_dict.keys()):
            cnt += 1
            gt = np.load(gts_dict[scene_token])
            pred = np.load(preds_dict[scene_token])
            semantics_gt, mask_lidar, mask_camera = gt['semantics'], gt['mask_lidar'], gt['mask_camera']
            semantics_pred = pred['semantics']
            mask_lidar = mask_lidar.astype(bool)
            mask_camera = mask_camera.astype(bool)
            if self.use_image_mask:
                masked_semantics_gt = semantics_gt[mask_camera]
                masked_semantics_pred = semantics_pred[mask_camera]
            elif self.use_lidar_mask:
                masked_semantics_gt = semantics_gt[mask_lidar]
                masked_semantics_pred = semantics_pred[mask_lidar]
            else:
                masked_semantics_gt = semantics_gt
                masked_semantics_pred = semantics_pred

            # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

            _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
            hist += _hist
            # _mIoU += _miou
        # mIoU_avg = _mIoU / num_samples
        mIoU = self.per_class_iu(hist)
        assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> class {ind_class} IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            
        print(f'===> mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))
        return mIoU

class Metric_F1Score():
    def __init__(self,
                gt_path,
                pred_path,
                leaf_size=10,
                threshold_acc=0.6,
                threshold_complete=0.6,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=False,) -> None:
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
    

    def get_dicts(self, ):
        gts_dict = {}
        for scene in os.listdir(self.gt_path):
            for frame in os.listdir(os.path.join(self.gt_path, scene)):
                scene_token = frame
                gts_dict[scene_token] = os.path.join(self.gt_path, scene, frame, 'labels.npz')
        print('number of gt samples = {}'.format(len(gts_dict)))

        preds_dict = {}
        for file in os.listdir(pred_path):
            scene_token = file.split('.npz')[0]
            preds_dict[scene_token] = os.path.join(self.pred_path, file)
        print('number of pred samples = {}'.format(len(preds_dict)))
        return gts_dict, preds_dict

    def voxel2points(self, voxel):
        #occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        #mask = reduce(torch.logical_or, [voxel == state for state in args.void])
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        #mask = np.logical_not(np.logical_or(voxel == self.void[0], voxel == self.void[1]))
        occIdx = np.where(mask)
        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                            occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                            occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]), axis=1)
        return points

    def __call__(self, ):
        #data_folder = Path('/home/txy/occ_nus_out')
        # data_folder = Path('./occ_nus_out')
        # sample_lists = list(data_folder.glob('*.npz'))
        gts_dict, preds_dict = self.get_dicts()
        cnt = 0
        tot_acc, tot_cmpl, tot_f1_mean = 0., 0., 0.
        for scene_token in tqdm(preds_dict.keys()):
            cnt += 1
            gt = np.load(gts_dict[scene_token])
            pred = np.load(preds_dict[scene_token])
            semantics_gt, mask_lidar, mask_camera = gt['semantics'], gt['mask_lidar'], gt['mask_camera']
            semantics_pred = pred['semantics']
            mask_lidar = mask_lidar.astype(bool)
            mask_camera = mask_camera.astype(bool)

            if self.use_image_mask:
                # masked_semantics_gt = semantics_gt[mask_camera]
                # masked_semantics_pred = semantics_pred[mask_camera]
                semantics_gt[mask_camera == False] = 255
                semantics_pred[mask_camera == False] = 255
            elif self.use_lidar_mask:
                semantics_gt[mask_lidar == False] = 255
                semantics_pred[mask_lidar == False] = 255
            else:
                pass
            
            DEBUG = False
            if DEBUG:
                semantics_gt = np.full((200, 200, 16), 255)
                semantics_gt[2, 2, 2] = 1
                semantics_gt[2, 1, 2] = semantics_gt[2, 3, 2] = semantics_gt[1, 2, 2] = semantics_gt[3, 3, 2] = 1
                semantics_pred = np.full((200, 200, 16), 255)
                semantics_pred[2, 2, 2] = 1
            ground_truth = self.voxel2points(semantics_gt)
            prediction = self.voxel2points(semantics_pred)
            # Voxelize to evaluate the accuracy/completeness independent to point cloud density
            # groundTruthCells = gt
            # predictionCells = pred

            # if args.vis:
            #     pcd = deepcopy(groundTruth)
            #     labels = np.zeros_like(pcd[:, 0], dtype=np.int8)
            #     pcd = np.concatenate((groundTruth, prediction))
            #     labels = np.concatenate((labels, np.ones_like(prediction[:, 0], dtype=np.int8)))
            #     pcd_colors = args.colormap[labels]
            #     show_point_cloud(pcd, True, pcd_colors)
            # groundTruthCells = getCellCoordinates(groundTruthNp, args.voxelSize)
            # predictionCells = getCellCoordinates(predictionNp[observedMask], args.voxelSize)

            # Evaluate chamfer distance
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            # print("groundTruth: ", ground_truth.shape)
            # print("prediction: ", prediction.shape) 
            complete_distance, _ = prediction_tree.query(ground_truth) # 每个gt找pred最近
            complete_distance = complete_distance.flatten()
            #print("Computed chamfer distance groundTruth to predictions: ", time.time() - start_time)
            accuracy_distance, _ = ground_truth_tree.query(prediction) # 每个pred找gt最近
            accuracy_distance = accuracy_distance.flatten()
            #print("Computed chamfer distance predictions to groundTruth: ", time.time() - start_time)

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            # completeGroundTruthPoints = groundTruth[completeMask]
            # completeGroundTruthCells = getCellCoordinates(completeGroundTruthPoints, args.voxelSize)
            # nCompleteGroundTruthCells = getNumUniqueCells(completeGroundTruthCells)
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            # accuratePredictionPoints = prediction[np.logical_and(accuracyMask, observedMask)]
            # accuratePredictionPoints = prediction[accuracyMask]
            # accuratePredictionCells = getCellCoordinates(accuratePredictionPoints, args.voxelSize)
            # nAccuratePredictionCells = getNumUniqueCells(accuratePredictionCells)
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / accuracy + 1 / completeness)

            tot_acc += accuracy
            tot_cmpl += completeness
            tot_f1_mean += fmean

            if cnt == 10: break

        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('######## completeness: {} #######'.format(tot_cmpl / cnt), base_color, attrs=attrs))
        print(pcolor('######## accuracy:     {} #######'.format(tot_acc / cnt), base_color, attrs=attrs))
        print(pcolor('######## F1 score:     {} #######'.format(tot_f1_mean / cnt), base_color, attrs=attrs))

def eval_nuscene(gt_path, pred_path):
    miou = Metric_mIoU(
        gt_path=gt_path,
        pred_path=pred_path,
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True,
    )
    metric = miou()

def eval_nuscenes_f1_score(gt_path, pred_path):
    f1_score = Metric_F1Score(
        gt_path=gt_path, 
        pred_path=pred_path,
        leaf_size=10,
        threshold_acc=0.4,
        threshold_complete=0.4,
        voxel_size=[0.4, 0.4, 0.4],
        range=[-40, -40, -1, 40, 40, 5.4],
        void=[17, 255],
        use_lidar_mask=False,
        use_image_mask=True,
    )
    metric = f1_score()

if __name__ == '__main__':
    gt_path="/home/txy/bev-occ/data/occ-trainval/gts/"
    pred_path="work_dirs/bevformer_base_occ_conv3d_3dvolume/results/"
    # eval_nuscene(gt_path, pred_path)
    eval_nuscenes_f1_score(gt_path, pred_path)
    