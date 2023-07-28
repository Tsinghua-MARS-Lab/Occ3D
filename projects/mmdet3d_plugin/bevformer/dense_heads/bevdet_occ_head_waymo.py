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
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule, caffe2_xavier_init
from ..loss.lovasz_losses import lovasz_softmax
from mmseg.models import LOSSES as LOSSES_SEG

@HEADS.register_module()
class BEVDetOccHeadWaymo(BaseModule):
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
                 more_conv=False,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 input_bev_z = None,
                 input_bev_h = None,
                 input_bev_w = None,
                 middle_convs_num=1,
                 bev_z=16,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 occ_thr=0.3,
                 use_free_mask=False,
                 use_focal_loss=False,
                 # num_classes=18,
                 act_cfg=None,
                 pillar_h=16,
                 norm_cfg = dict(type='SyncBN', requires_grad=True),
                 norm_cfg_3d=dict(type='SyncBN', requires_grad=True),
                 loss_occ=None,
                 use_mask=False,
                 FREE_LABEL=None,
                 positional_encoding=None,
                 **kwargs):
        self.FREE_LABEL = FREE_LABEL
        self.more_conv=more_conv
        self.use_free_mask = use_free_mask
        self.use_focal_loss = use_focal_loss
        self.pillar_h=pillar_h
        self.bev_z = bev_z
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.input_bev_z = input_bev_z
        self.input_bev_h = input_bev_h
        self.input_bev_w = input_bev_w
        self.occ_thr = occ_thr
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask
        self.embed_dims=embed_dims
        if use_free_mask:
            # assert use_focal_loss, \
            #     'you should use focal loss if you ommit the free class!'
            self.num_classes = self.num_classes-1

        use_bias_3d = norm_cfg_3d is None
    
        super(BEVDetOccHeadWaymo, self).__init__()
        self.loss_occ_fun = dict()
        for name, loss_dict in loss_occ.items():
            if LOSSES_SEG.get(loss_dict['type']) is not None:
                self.loss_occ_fun['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
            else:
                _type = loss_dict['type']
                raise KeyError(f'{_type} not in LOSSES_SEG registry')

        # self.loss_occ = build_loss(loss_occ)


        self.middle_dims = self.embed_dims // pillar_h
        self.out_dim = self.middle_dims*2
        self.decoder = nn.Sequential(
            ConvModule(
                self.middle_dims,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias_3d,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=act_cfg),
            ConvModule(
                self.out_dim,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias_3d,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg_3d,
                act_cfg=act_cfg),
        )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2,self.num_classes),
        )

        # if self.more_conv:
        #     decoder = []
        #     decoder.append(
        #         ConvModule(
        #             self.embed_dims,
        #             self.embed_dims,
        #             kernel_size=[2, 3, 3],
        #             stride=[2, 1, 1],
        #             padding=[0, 1, 1],
        #             # bias=use_bias_3d,
        #             conv_cfg=dict(type='ConvTranspose3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),)
        #     for i in range(middle_convs_num):
        #         decoder.append(
        #             ConvModule(
        #                 self.embed_dims,
        #                 self.embed_dims,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=2,
        #                 dilation=2,
        #                 # bias=use_bias_3d,
        #                 conv_cfg=dict(type='Conv3d'),
        #                 norm_cfg=norm_cfg_3d,
        #                 act_cfg=act_cfg),
        #         )
        #     decoder.append(
        #         ConvModule(
        #             self.embed_dims,
        #             self.embed_dims * 2,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #             # bias=use_bias_3d,
        #             conv_cfg=dict(type='Conv3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),
        #     )
        #     for i in range(middle_convs_num-1):
        #         decoder.append(
        #             ConvModule(
        #                 self.embed_dims * 2,
        #                 self.embed_dims * 2,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 # bias=use_bias_3d,
        #                 conv_cfg=dict(type='Conv3d'),
        #                 norm_cfg=norm_cfg_3d,
        #                 act_cfg=act_cfg),
        #         )
        #     self.decoder=nn.Sequential(*decoder)
        # else:
        #     self.decoder = nn.Sequential(
        #         ConvModule(
        #             self.embed_dims,
        #             self.embed_dims,
        #             kernel_size=[2, 3, 3],
        #             stride=[2, 1, 1],
        #             padding=[0, 1, 1],
        #             # bias=use_bias_3d,
        #             conv_cfg=dict(type='ConvTranspose3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),
        #         ConvModule(
        #             self.embed_dims,
        #             self.embed_dims * 2,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #             # bias=use_bias_3d,
        #             conv_cfg=dict(type='Conv3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),
        #     )
        #
        # self.predicter = nn.Sequential(
        #     nn.Linear(self.embed_dims * 2, self.embed_dims),
        #     nn.Linear(self.embed_dims, self.num_classes),
        # )

        # self.loss_occ = build_loss(loss_occ)
        # self.positional_encoding = build_positional_encoding(
        #     positional_encoding)
        self.embed_dims = self.embed_dims


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, bev_feats, img_metas, prev_bev=None, only_bev=False, test=False):
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

        bs=bev_feats.size(0)
        occ_out = self.decoder(bev_feats.view(bs, -1, self.pillar_h, self.bev_h, self.bev_w))
        # occ_out=self.decoder(bev_feats)
        occ_out=occ_out.permute(0,4,3,2,1)
        occ_out = self.predicter(occ_out)


        outs = {
            'occ':occ_out,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics_list,
             mask_camera_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        occ=preds_dicts['occ']
        loss_dict = self.loss_single(voxel_semantics_list,mask_camera_list,occ)
        # loss_dict['loss_ce'] = loss_ce

        return loss_dict

    def get_loss(self,loss_occ_fun, cls_score, labels, weight=None):
        loss_occ = dict()
        for loss_name in sorted(list(loss_occ_fun.keys())):
            if 'lovasz' in loss_name:
                cls_score = cls_score.reshape(*cls_score.shape, 1, 1)
                labels = labels.reshape(*labels.shape, 1, 1)
            _loss = loss_occ_fun[loss_name](
                cls_score, labels, weight)
            loss_occ[loss_name] = _loss
        return loss_occ

    def loss_single(self,voxel_semantics,mask_camera,preds_dicts):
        voxel_semantics[voxel_semantics == self.FREE_LABEL] = self.num_classes - 1
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds_dicts=preds_dicts.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_ce=self.loss_occ(preds_dicts,voxel_semantics,mask_camera, avg_factor=num_total_samples)

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1, self.num_classes)
            if self.use_free_mask:
                free_mask=voxel_semantics<self.num_classes
                voxel_semantics=voxel_semantics[free_mask]
                preds_dicts=preds_dicts[free_mask]
                pos_num=voxel_semantics.shape[0]

            else:
                pos_num=voxel_semantics.shape[0]
            loss_ce = self.get_loss(self.loss_occ_fun, preds_dicts,voxel_semantics, weight=None)
            #loss_ce = self.loss_occ(preds_dicts, voxel_semantics.long(),avg_factor=pos_num)
        # loss_lovasz=lovasz_softmax(preds_dicts.softmax(-1),voxel_semantics,ignore=self.num_classes-1)
        return loss_ce #,loss_lovasz

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
        occ_out = preds_dicts['occ']
        if self.use_focal_loss:
            occ_out = occ_out.sigmoid()

        if self.use_free_mask:
            bs, h, w, z, c = occ_out.shape
            occ_out = occ_out.reshape(bs, -1, self.num_classes)
            occ_out = torch.cat((occ_out, torch.ones_like(occ_out)[:, :, :1] * self.occ_thr), dim=-1)
            occ_out = occ_out.reshape(bs, h, w, z, -1)
        else:
            occ_out = occ_out.softmax(-1)
        occ_out = occ_out.argmax(-1)

        return occ_out

