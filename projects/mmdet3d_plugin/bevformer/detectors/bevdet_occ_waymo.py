# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import numpy as np

@DETECTORS.register_module()
class BEVDetOccWaymo(MVXTwoStageDetector):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs):
        super(BEVDetOccWaymo, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = \
            builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img, pose_metas, **kwargs):
        """Extract features of images."""
        # print('img',img.shape)
        x = self.image_encoder(img)
        x, depth = self.img_view_transformer([x] ,pose_metas)
        x = self.bev_encoder(x)
        return x, depth

    def extract_feat(self, points, img,pose_metas, img_metas=None, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, pose_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
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
        # print('img',img.shape)
        len_queue = img.size(1)
        if len_queue==1:
            img=img.squeeze(dim=1)
        img_metas = [each[len_queue - 1] for each in img_metas]

        pose_metas=self.get_pose_metas(img,img_metas)

        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img, pose_metas=pose_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d,voxel_semantics,mask_camera, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def get_pose_metas(self,img,img_metas):
        cam_intrinsic = []
        sensor2ego = []
        lidar2img = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            cam_intrinsic.append(img_meta['cam_intrinsic'])
            sensor2ego.append(img_meta['sensor2ego'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = img.new_tensor(lidar2img)  # (B, N, 4, 4)

        cam_intrinsic = np.asarray(cam_intrinsic)
        cam_intrinsic = img.new_tensor(cam_intrinsic)  # (B, N, 4, 4)

        sensor2ego = np.asarray(sensor2ego)
        sensor2ego = img.new_tensor(sensor2ego)  # (B, N, 4, 4)
        # print(lidar2cam.shape,cam_intrinsic.shape,ego2lidar.shape,)
        pose_metas = [lidar2img, sensor2ego, cam_intrinsic]
        return pose_metas

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics,
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

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)

        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses


    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,

                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img, 'img'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_metas)))

        if not isinstance(img[0][0], list):
            img_inputs = [img] if img is None else img
            points = [points] if points is None else points
            pose_metas = self.get_pose_metas(img_inputs[0], img_metas[0])
            occ_results=self.simple_test(points[0], img_metas[0], img_inputs[0],pose_metas=pose_metas,
                             **kwargs)

            DEBUG = True
            if DEBUG:
                # print('output', type(occ_results), type(voxel_semantics[0]), type(img_metas[0]))
                save_root = 'work_dirs/bevdet-r101-waymo/results_epoch24/'
                sample_idx = img_metas[0][0]['sample_idx']
                sample_idx = str(sample_idx)
                np.savez_compressed(save_root+str(sample_idx),output=occ_results.to(torch.uint8).cpu().numpy(),
                                    voxel_semantics=voxel_semantics[0].to(torch.uint8).cpu().numpy(), mask_lidar=mask_lidar[0].to(torch.uint8).cpu().numpy(), mask_camera=mask_camera[0].to(torch.uint8).cpu().numpy())
                print('saved',save_root+str(sample_idx))

            return occ_results
        else:
            return self.aug_test(None, img_metas[0], img[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    pose_metas=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas,pose_metas=pose_metas, **kwargs)

        occ = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        return occ

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,test=True)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return occ


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


# @DETECTORS.register_module()
# class BEVDetTRT(BEVDet):
#
#     def result_serialize(self, outs):
#         outs_ = []
#         for out in outs:
#             for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
#                 outs_.append(out[0][key])
#         return outs_
#
#     def result_deserialize(self, outs):
#         outs_ = []
#         keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
#         for head_id in range(len(outs) // 6):
#             outs_head = [dict()]
#             for kid, key in enumerate(keys):
#                 outs_head[0][key] = outs[head_id * 6 + kid]
#             outs_.append(outs_head)
#         return outs_
#
#     def forward(
#         self,
#         img,
#         ranks_depth,
#         ranks_feat,
#         ranks_bev,
#         interval_starts,
#         interval_lengths,
#     ):
#         x = self.img_backbone(img)
#         x = self.img_neck(x)
#         x = self.img_view_transformer.depth_net(x)
#         depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
#         tran_feat = x[:, self.img_view_transformer.D:(
#             self.img_view_transformer.D +
#             self.img_view_transformer.out_channels)]
#         tran_feat = tran_feat.permute(0, 2, 3, 1)
#         x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
#                                ranks_depth, ranks_feat, ranks_bev,
#                                interval_starts, interval_lengths)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         bev_feat = self.bev_encoder(x)
#         outs = self.pts_bbox_head([bev_feat])
#         outs = self.result_serialize(outs)
#         return outs
#
#     def get_bev_pool_input(self, input):
#         coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
#         return self.img_view_transformer.voxel_pooling_prepare_v2(coor)
#
#
# @DETECTORS.register_module()
# class BEVDet4D(BEVDet):
#     r"""BEVDet4D paradigm for multi-camera 3D object detection.
#
#     Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_
#
#     Args:
#         pre_process (dict | None): Configuration dict of BEV pre-process net.
#         align_after_view_transfromation (bool): Whether to align the BEV
#             Feature after view transformation. By default, the BEV feature of
#             the previous frame is aligned during the view transformation.
#         num_adj (int): Number of adjacent frames.
#         with_prev (bool): Whether to set the BEV feature of previous frame as
#             all zero. By default, False.
#     """
#     def __init__(self,
#                  pre_process=None,
#                  align_after_view_transfromation=False,
#                  num_adj=1,
#                  with_prev=True,
#                  **kwargs):
#         super(BEVDet4D, self).__init__(**kwargs)
#         self.pre_process = pre_process is not None
#         if self.pre_process:
#             self.pre_process_net = builder.build_backbone(pre_process)
#         self.align_after_view_transfromation = align_after_view_transfromation
#         self.num_frame = num_adj + 1
#
#         self.with_prev = with_prev
#
#     @force_fp32()
#     def shift_feature(self, input, trans, rots, bda, bda_adj=None):
#         n, c, h, w = input.shape
#         _, v, _ = trans[0].shape
#
#         # generate grid
#         xs = torch.linspace(
#             0, w - 1, w, dtype=input.dtype,
#             device=input.device).view(1, w).expand(h, w)
#         ys = torch.linspace(
#             0, h - 1, h, dtype=input.dtype,
#             device=input.device).view(h, 1).expand(h, w)
#         grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
#         grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)
#
#         # get transformation from current ego frame to adjacent ego frame
#         # transformation from current camera frame to current ego frame
#         c02l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
#         c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
#         c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
#         c02l0[:, :, 3, 3] = 1
#
#         # transformation from adjacent camera frame to current ego frame
#         c12l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
#         c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
#         c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
#         c12l0[:, :, 3, 3] = 1
#
#         # add bev data augmentation
#         bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
#         bda_[:, :, :3, :3] = bda.unsqueeze(1)
#         bda_[:, :, 3, 3] = 1
#         c02l0 = bda_.matmul(c02l0)
#         if bda_adj is not None:
#             bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
#             bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
#             bda_[:, :, 3, 3] = 1
#         c12l0 = bda_.matmul(c12l0)
#
#         # transformation from current ego frame to adjacent ego frame
#         l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
#             n, 1, 1, 4, 4)
#         '''
#           c02l0 * inv(c12l0)
#         = c02l0 * inv(l12l0 * c12l1)
#         = c02l0 * inv(c12l1) * inv(l12l0)
#         = l02l1 # c02l0==c12l1
#         '''
#
#         l02l1 = l02l1[:, :, :,
#                       [True, True, False, True], :][:, :, :, :,
#                                                     [True, True, False, True]]
#
#         feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
#         feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
#         feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
#         feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
#         feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
#         feat2bev[2, 2] = 1
#         feat2bev = feat2bev.view(1, 3, 3)
#         tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)
#
#         # transform and normalize
#         grid = tf.matmul(grid)
#         normalize_factor = torch.tensor([w - 1.0, h - 1.0],
#                                         dtype=input.dtype,
#                                         device=input.device)
#         grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
#                                                             2) * 2.0 - 1.0
#         output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
#         return output
#
#     def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
#                          bda, mlp_input):
#         x = self.image_encoder(img)
#         bev_feat, depth = self.img_view_transformer(
#             [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
#         if self.pre_process:
#             bev_feat = self.pre_process_net(bev_feat)[0]
#         return bev_feat, depth
#
#     def extract_img_feat_sequential(self, inputs, feat_prev):
#         imgs, rots_curr, trans_curr, intrins = inputs[:4]
#         rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
#         bev_feat_list = []
#         mlp_input = self.img_view_transformer.get_mlp_input(
#             rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
#             post_trans, bda[0:1, ...])
#         inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
#                        intrins, post_rots, post_trans, bda[0:1,
#                                                            ...], mlp_input)
#         bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
#         bev_feat_list.append(bev_feat)
#
#         # align the feat_prev
#         _, C, H, W = feat_prev.shape
#         feat_prev = \
#             self.shift_feature(feat_prev,
#                                [trans_curr, trans_prev],
#                                [rots_curr, rots_prev],
#                                bda)
#         bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))
#
#         bev_feat = torch.cat(bev_feat_list, dim=1)
#         x = self.bev_encoder(bev_feat)
#         return [x], depth
#
#     def prepare_inputs(self, inputs):
#         # split the inputs into each frame
#         B, N, _, H, W = inputs[0].shape
#         N = N // self.num_frame
#         imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
#         imgs = torch.split(imgs, 1, 2)
#         imgs = [t.squeeze(2) for t in imgs]
#         rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
#         extra = [
#             rots.view(B, self.num_frame, N, 3, 3),
#             trans.view(B, self.num_frame, N, 3),
#             intrins.view(B, self.num_frame, N, 3, 3),
#             post_rots.view(B, self.num_frame, N, 3, 3),
#             post_trans.view(B, self.num_frame, N, 3)
#         ]
#         extra = [torch.split(t, 1, 1) for t in extra]
#         extra = [[p.squeeze(1) for p in t] for t in extra]
#         rots, trans, intrins, post_rots, post_trans = extra
#         return imgs, rots, trans, intrins, post_rots, post_trans, bda
#
#     def extract_img_feat(self,
#                          img,
#                          img_metas,
#                          pred_prev=False,
#                          sequential=False,
#                          **kwargs):
#         if sequential:
#             return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
#         imgs, rots, trans, intrins, post_rots, post_trans, bda = \
#             self.prepare_inputs(img)
#         """Extract features of images."""
#         bev_feat_list = []
#         depth_list = []
#         key_frame = True  # back propagation for key frame only
#         for img, rot, tran, intrin, post_rot, post_tran in zip(
#                 imgs, rots, trans, intrins, post_rots, post_trans):
#             if key_frame or self.with_prev:
#                 if self.align_after_view_transfromation:
#                     rot, tran = rots[0], trans[0]
#                 mlp_input = self.img_view_transformer.get_mlp_input(
#                     rots[0], trans[0], intrin, post_rot, post_tran, bda)
#                 inputs_curr = (img, rot, tran, intrin, post_rot,
#                                post_tran, bda, mlp_input)
#                 if key_frame:
#                     bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
#                 else:
#                     with torch.no_grad():
#                         bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
#             else:
#                 bev_feat = torch.zeros_like(bev_feat_list[0])
#                 depth = None
#             bev_feat_list.append(bev_feat)
#             depth_list.append(depth)
#             key_frame = False
#         if pred_prev:
#             assert self.align_after_view_transfromation
#             assert rots[0].shape[0] == 1
#             feat_prev = torch.cat(bev_feat_list[1:], dim=0)
#             trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
#             rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
#             trans_prev = torch.cat(trans[1:], dim=0)
#             rots_prev = torch.cat(rots[1:], dim=0)
#             bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
#             return feat_prev, [
#                 imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
#                 trans_prev, post_rots[0], post_trans[0], bda_curr
#             ]
#         if self.align_after_view_transfromation:
#             for adj_id in range(1, self.num_frame):
#                 bev_feat_list[adj_id] = \
#                     self.shift_feature(bev_feat_list[adj_id],
#                                        [trans[0], trans[adj_id]],
#                                        [rots[0], rots[adj_id]],
#                                        bda)
#         bev_feat = torch.cat(bev_feat_list, dim=1)
#         x = self.bev_encoder(bev_feat)
#         return [x], depth_list[0]
#
#
# @DETECTORS.register_module()
# class BEVDepth4D(BEVDet4D):
#
#     def forward_train(self,
#                       points=None,
#                       img_metas=None,
#                       gt_bboxes_3d=None,
#                       gt_labels_3d=None,
#                       gt_labels=None,
#                       gt_bboxes=None,
#                       img_inputs=None,
#                       proposals=None,
#                       gt_bboxes_ignore=None,
#                       **kwargs):
#         """Forward training function.
#
#         Args:
#             points (list[torch.Tensor], optional): Points of each sample.
#                 Defaults to None.
#             img_metas (list[dict], optional): Meta information of each sample.
#                 Defaults to None.
#             gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
#                 Ground truth 3D boxes. Defaults to None.
#             gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
#                 of 3D boxes. Defaults to None.
#             gt_labels (list[torch.Tensor], optional): Ground truth labels
#                 of 2D boxes in images. Defaults to None.
#             gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
#                 images. Defaults to None.
#             img (torch.Tensor optional): Images of each sample with shape
#                 (N, C, H, W). Defaults to None.
#             proposals ([list[torch.Tensor], optional): Predicted proposals
#                 used for training Fast RCNN. Defaults to None.
#             gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
#                 2D boxes in images to be ignored. Defaults to None.
#
#         Returns:
#             dict: Losses of different branches.
#         """
#         img_feats, pts_feats, depth = self.extract_feat(
#             points, img=img_inputs, img_metas=img_metas, **kwargs)
#         gt_depth = kwargs['gt_depth']
#         loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
#         losses = dict(loss_depth=loss_depth)
#         losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
#                                             gt_labels_3d, img_metas,
#                                             gt_bboxes_ignore)
#         losses.update(losses_pts)
#         return losses
