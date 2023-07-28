# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
from mmseg.models import LOSSES as LOSSES_SEG


@HEADS.register_module()
class OccFormerHeadWaymo(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 volume_flag=True, # volume or bev 
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 bev_z=8,
                 loss_occ=None,
                 loss_binary_occ=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=False,
                 positional_encoding=None,
                 FREE_LABEL=None,
                 **kwargs):
        if not volume_flag: assert bev_z == 1
        self.volume_flag = volume_flag

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        self.FREE_LABEL = FREE_LABEL
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]



        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(OccFormerHeadWaymo, self).__init__()

        # self.loss_occ = build_loss(loss_occ)
        self.loss_occ_fun = dict()
        for name, loss_dict in loss_occ.items():
            if LOSSES_SEG.get(loss_dict['type']) is not None:
                self.loss_occ_fun['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
            else:
                _type = loss_dict['type']
                raise KeyError(f'{_type} not in LOSSES_SEG registry')

        # self.loss_occ = build_loss(loss_occ)
        if loss_binary_occ is not None:
            self.loss_binary_occ_func = dict()
            for name, loss_dict in loss_binary_occ.items():
                if LOSSES_SEG.get(loss_dict['type']) is not None:
                    self.loss_binary_occ_func['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
                else:
                    _type = loss_dict['type']
                    raise KeyError(f'{_type} not in LOSSES_SEG registry')
            
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_z * self.bev_h * self.bev_w, self.embed_dims)
    # def _init_layers(self):
    #     """Initialize classification branch and regression branch of head."""
    #     cls_branch = []
    #     for _ in range(self.num_reg_fcs):
    #         cls_branch.append(Linear(self.embed_dims, self.embed_dims))
    #         cls_branch.append(nn.LayerNorm(self.embed_dims))
    #         cls_branch.append(nn.ReLU(inplace=True))
    #     cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
    #     fc_cls = nn.Sequential(*cls_branch)
    #
    #     reg_branch = []
    #     for _ in range(self.num_reg_fcs):
    #         reg_branch.append(Linear(self.embed_dims, self.embed_dims))
    #         reg_branch.append(nn.ReLU())
    #     reg_branch.append(Linear(self.embed_dims, self.code_size))
    #     reg_branch = nn.Sequential(*reg_branch)
    #
    #     def _get_clones(module, N):
    #         return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    #
    #     # last reg_branch is used to generate proposal from
    #     # encode feature map when as_two_stage is True.
    #     num_pred = (self.transformer.decoder.num_layers + 1) if \
    #         self.as_two_stage else self.transformer.decoder.num_layers
    #
    #     if self.with_box_refine:
    #         self.cls_branches = _get_clones(fc_cls, num_pred)
    #         self.reg_branches = _get_clones(reg_branch, num_pred)
    #     else:
    #         self.cls_branches = nn.ModuleList(
    #             [fc_cls for _ in range(num_pred)])
    #         self.reg_branches = nn.ModuleList(
    #             [reg_branch for _ in range(num_pred)])
    #
    #     if not self.as_two_stage:
    #         self.bev_embedding = nn.Embedding(
    #             self.bev_h * self.bev_w, self.embed_dims)
    #         self.query_embedding = nn.Embedding(self.num_query,
    #                                             self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, test=False, voxel_semantics=None, mask_infov=None, mask_lidar=None, mask_camera=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        #breakpoint()
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = None
        bev_queries = self.bev_embedding.weight.to(dtype)  #[20000, 256]
        if self.volume_flag:
            bev_mask = torch.zeros((bs,self.bev_z, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
        else:
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)  #[1, 200, 100]
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  #[1, 256, 1, 200, 100]
        grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w)  #(0.4, 0.4)
        #breakpoint()
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            _bev_embed, _feat_flatten, _bev_pos, _bev_queries, _spatial_shapes, _level_start_index, _shift =  self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            return _bev_embed

        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_z,
                self.bev_h,
                self.bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                reg_branches=None,  # noqa:E501
                cls_branches=None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                voxel_semantics=voxel_semantics, mask_infov=mask_infov, mask_lidar=mask_lidar, mask_camera=mask_camera
            )
        bev_embed, occ_outs, extra = outputs
        # bev_embed, hs, init_reference, inter_references = outputs
        #
        #
        # outs = {
        #     'bev_embed': bev_embed,
        #     'all_cls_scores': outputs_classes,
        #     'all_bbox_preds': outputs_coords,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None,
        # }

        # if test:
        #     return bev_embed, occ_outs
        # else:
        #     return occ_outs
        outs = {
            'bev_embed': bev_embed,
            'occ':occ_outs,
            'extra': extra,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics_list,
             mask_infov_list,
             mask_lidar_list,
             mask_camera_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        voxel_semantics_list[voxel_semantics_list==self.FREE_LABEL] = self.num_classes-1 # wft forget convert FREE_LABEL from 23 to 15
        # assert voxel_semantics_list.min()>=0 and voxel_semantics_list.max()<=17
        occ=preds_dicts['occ']
        loss_dict = self.loss_single(voxel_semantics_list,mask_infov_list,mask_lidar_list,mask_camera_list,occ)
        if 'extra' in preds_dicts and bool(preds_dicts['extra']):
            extra = preds_dicts['extra']
            pred_list = extra['outputs_list']
            for iter_i, preds in enumerate(pred_list):
                losses = self.loss_single(voxel_semantics_list,mask_infov_list,mask_lidar_list,mask_camera_list,preds, binary_loss=True)
                # loss_dict['loss_occ_iter{}'.format(iter_i)] = losses
                for k,v in losses.items():
                    loss_dict['loss_occ_iter{}_{}'.format(iter_i, k)] = v

        return loss_dict

    def loss_single(self,voxel_semantics,mask_infov,mask_lidar,mask_camera,preds_dicts, binary_loss=False):
        mask=torch.ones_like(voxel_semantics)
        if self.use_infov_mask:
            mask = torch.logical_and(mask_infov, mask)
        if self.use_lidar_mask:
            mask = torch.logical_and(mask_lidar, mask)
        if self.use_camera_mask:
            mask = torch.logical_and(mask_camera, mask)
        mask = mask.bool()
        preds = preds_dicts

        def get_loss(loss_occ_fun, cls_score, labels, weight=None):
            loss_occ = dict()
            for loss_name in sorted(list(loss_occ_fun.keys())):
                if 'focal' in loss_name:
                    avg_factor = mask.sum()
                else:
                    avg_factor = None
                if 'lovasz' in loss_name:
                    cls_score = cls_score.reshape(*cls_score.shape, 1, 1)
                    labels = labels.reshape(*labels.shape, 1, 1)
                _loss = loss_occ_fun[loss_name](
                    cls_score, labels, weight, avg_factor=avg_factor)
                loss_occ[loss_name] = _loss
            return loss_occ

        # smart convert gt
        if binary_loss:
            assert preds.shape[-1] == 2
            binary_gt = voxel_semantics != self.num_classes-1
            bs, W, H, D = voxel_semantics.shape
            _bs, _W, _H, _D, _ = preds.shape
            assert W % _W == 0 and H % _H == 0 and D % _D == 0
            scale_W, scale_H, scale_D = W//_W, H//_H, D//_D
            
            _scale = 1
            while _scale != scale_W:
                binary_gt = binary_gt.reshape(bs, -1, 2, H,  D)
                binary_gt = torch.logical_or(binary_gt[:, :, 0, :, :], binary_gt[:, :, 1, :, :, :])
                _scale *= 2
            _scale = 1
            while _scale != scale_H:
                binary_gt = binary_gt.reshape(bs, _W,  -1,  2, D)
                binary_gt = torch.logical_or(binary_gt[:, :, :, 0, :], binary_gt[:, :, :, 1, :])
                _scale *= 2
            _scale = 1
            while _scale != scale_D:
                binary_gt = binary_gt.reshape(bs, _W,  _H,  -1, 2)
                binary_gt = torch.logical_or(binary_gt[:, :, :, :, 0], binary_gt[:, :, :, :, 1])
                _scale *= 2
            binary_gt = binary_gt.long()
            binary_gt=binary_gt.reshape(-1)
            preds=preds.reshape(-1, 2)
            mask=torch.ones_like(binary_gt, dtype=torch.bool)
            # num_total_samples=mask.sum()
            # print("#############", binary_gt.sum())
            # loss_occ=self.loss_occ(preds, binary_gt, mask, avg_factor=num_total_samples)
            loss_occ = get_loss(self.loss_binary_occ_func, preds[mask], binary_gt[mask], weight=None)
        else:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask=mask.reshape(-1)
            # num_total_samples=mask.sum()
            # loss_occ=self.loss_occ(preds, voxel_semantics, mask, avg_factor=num_total_samples)
            loss_occ = get_loss(self.loss_occ_fun, preds[mask], voxel_semantics[mask], weight=None)

        return loss_occ

    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_label=occ_score.argmax(-1)

        occ_label[occ_label==self.num_classes-1] = self.FREE_LABEL

        return occ_label
