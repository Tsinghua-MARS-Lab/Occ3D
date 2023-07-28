
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule, caffe2_xavier_init
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder3DConv(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args,embed_dims=256, pc_range=None, num_points_in_voxel=2,num_voxel=8,bev_h=100,bev_w=100,return_intermediate=False, dataset_type='nuscenes',
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):

        super(BEVFormerEncoder3DConv, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.embed_dims=embed_dims
        layer_convs=[]
        use_bias_3d = norm_cfg_3d is None
        for i in range(self.num_layers):
            layer_convs.append(
                ConvModule(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
            )
        self.layer_convs=nn.Sequential(*layer_convs)
        self.n_p_in_voxel=num_points_in_voxel
        self.n_voxel=num_voxel
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, n_p_in_voxel=2,n_voxel=8, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.

            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        n_p_in_pillar=n_voxel*n_p_in_voxel
        # reference points in 3D space, used in spatial cross-attention (SCA)

        if dim == '3d':
            zs = torch.linspace(0.2, n_voxel - 0.2, n_p_in_pillar, dtype=dtype,
                                device=device).view(n_voxel,n_p_in_voxel, 1, 1).permute(1,0,2,3).expand(n_p_in_voxel,n_voxel, H, W)  / n_voxel

            xs = torch.linspace(0.2, W - 0.2, W, dtype=dtype,
                                device=device).view(1,1, 1, W).expand(n_p_in_voxel,n_voxel, H, W) / W
            ys = torch.linspace(0.2, H - 0.2, H, dtype=dtype,
                                device=device).view(1,1, H, 1).expand(n_p_in_voxel,n_voxel, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)  #shape: (num_points_in_pillar,h,w,3)
            ref_3d = ref_3d.permute(0, 4, 1, 2, 3).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  #shape: (bs,num_points_in_pillar,h*w,3)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.2,
                               n_voxel - 0.2,
                               n_voxel,
                               dtype=dtype,
                               device=device),
                torch.linspace(0.2,
                               H - 0.2,
                               H,
                               dtype=dtype,
                               device=device),
                torch.linspace(0.2,
                               W - 0.2,
                               W,
                               dtype=dtype,
                               device=device)
            )  # shape: (bev_z, bev_h, bev_w)
            ref_z = ref_z.reshape(-1)[None] / n_voxel
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y, ref_z), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  # (bs, num_query, 1, 3)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        ego2lidar=img_metas[0]['ego2lidar']
        lidar2img = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        ego2lidar = reference_points.new_tensor(ego2lidar)
        # ego2lidar = ego2lidar.unsqueeze(dim=0).repeat(num_imgs,1,1).unsqueeze(0)

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3) #shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3] # D=num_points_in_pillar , num_query=h*w
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  #shape: (num_points_in_pillar,bs,num_cam,h*w,4)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar=ego2lidar.view(1,1,1,1,4,4).repeat(D,1,num_cam,num_query,1,1)
        reference_points_cam = torch.matmul(torch.matmul(lidar2img.to(torch.float32),ego2lidar.to(torch.float32)),reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) #shape: (num_cam,bs,h*w,num_points_in_pillar,2)


        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_z=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2],self.n_p_in_voxel,self.n_voxel, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        # print('ref_3d',ref_3d.shape)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        shift3d = shift.new_zeros(1, 3)
        shift3d[:, :2] = shift
        shift_ref_2d += shift3d[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 3)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 3)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_z=bev_z,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bs, _, c = output.shape
            output = output.permute(0, 2, 1).reshape(bs, c, bev_z, bev_h, bev_w)
            output = self.layer_convs[lid](output)
            output = output.flatten(2).permute(0, 2, 1)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

