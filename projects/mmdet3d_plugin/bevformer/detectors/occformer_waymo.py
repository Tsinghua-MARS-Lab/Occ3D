# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time

from san import tools as san_tools
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

@DETECTORS.register_module()
class OccFormerWaymo(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 clip_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(OccFormerWaymo,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        if clip_backbone is not None:
            assert img_backbone is None
            clip_config = clip_backbone['config_file']
            clip_model_path = clip_backbone['model_path']
            clip_model_path = clip_backbone['model_path']

            clip_cfg = san_tools.setup(clip_config)
            clip_backbone = DefaultTrainer.build_model(clip_cfg)
            if clip_model_path.startswith("huggingface:"):
                clip_model_path = san_tools.download_model(clip_model_path)
            print("Loading model from: ", clip_model_path)
            DetectionCheckpointer(clip_backbone, save_dir=clip_cfg.OUTPUT_DIR).resume_or_load(
                clip_model_path
            )
            clip_backbone.eval()
            self.img_backbone = clip_backbone

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics,
                          mask_infov,
                          mask_lidar,
                          mask_camera,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        #breakpoint()
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, 
            voxel_semantics=voxel_semantics, mask_infov=mask_infov,mask_lidar=mask_lidar, mask_camera=mask_camera
        )
        loss_inputs = [voxel_semantics, mask_infov, mask_lidar, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        #breakpoint()
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape #[1, 2, 5, 3, 640, 960]
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue, img_metas=img_metas_list)  # [80,128]=>[10,15]
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            #breakpoint()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      voxel_semantics=None,
                      mask_infov=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        #breakpoint()
        len_queue = img.size(1)  #[1, 5, 3, 640, 960]
        prev_img = img[:, :-1, ...] # [1, 2, 5, 3, 640, 960] BEVFormer保留了3帧，这里是2帧
        img = img[:, -1, ...]    #[1, 5, 3, 640, 960] waymo是5帧

        # # DEBUG_TMP
        # import pickle as pkl
        # with open('work_dirs/metas_waymo.pkl', 'wb') as f:
        #     pkl.dump({
        #         "imgs": img,
        #         "voxel_semantics": voxel_semantics,
        #         "mask_camera": mask_camera,

        #     }, f)
        #DEBUG_TMP
        # prev_bev = None
        # #breakpoint()
        prev_img_metas = copy.deepcopy(img_metas)
        for prev_img_meta in prev_img_metas: del prev_img_meta[len_queue-1]
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        #breakpoint()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, voxel_semantics, mask_infov, mask_lidar, mask_camera, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self,  img_metas,
                            img=None,
                            voxel_semantics=None,
                            mask_infov=None,
                            mask_lidar=None,
                            mask_camera=None,
                            **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        
        try:
            scene_token = img_metas[0][0]['sample_idx'] // 1000 # waymo
        except Exception as e:
            scene_token = img_metas[0][0]['sample_idx'] # nuscene

        if scene_token != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = scene_token

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        if voxel_semantics is not None: voxel_semantics = voxel_semantics[0]
        if mask_infov is not None: mask_infov = mask_infov[0]
        if mask_lidar is not None: mask_lidar = mask_lidar[0]
        if mask_camera is not None: mask_camera = mask_camera[0]

        new_prev_bev, voxel_semantics_preds = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], voxel_semantics=voxel_semantics, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        occ_results = {
            "voxel_semantics": voxel_semantics.to(torch.uint8).cpu().numpy(),
            "voxel_semantics_preds": voxel_semantics_preds.to(torch.uint8).cpu().numpy(),
            "mask_infov": mask_infov.cpu().numpy() if mask_infov is not None else None,
            "mask_lidar": mask_lidar.cpu().numpy() if mask_lidar is not None else None,
            "mask_camera": mask_camera.cpu().numpy() if mask_camera is not None else None,
        }
        return occ_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, voxel_semantics=None, mask_infov=None, mask_lidar=None, mask_camera=None):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,test=True, voxel_semantics=voxel_semantics, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return outs['bev_embed'], occ


    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, voxel_semantics=None, mask_infov=None, mask_lidar=None, mask_camera=None):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale, voxel_semantics=voxel_semantics, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, occ
