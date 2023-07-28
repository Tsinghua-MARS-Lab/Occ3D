# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule, caffe2_xavier_init



@TRANSFORMER.register_module()
class HybridTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 act_cfg=None,
                 norm_cfg_3d=dict(type='SyncBN', requires_grad=True),
                 position=None,  # positional embedding of query point
                 encoder_embed_dims=[256, 256, 128, 64],
                 feature_map_z=[1, 4, 8, 16],
                 dilations=[2,2,2,2],
                 paddings=[2,2,2,2],

                 embed_dims=256,
                 more_conv=False,
                 use_conv=False,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 decoder_on_bev=True,
                 bev_z=16,
                 **kwargs):
        super(HybridTransformer, self).__init__(**kwargs)
        self.more_conv=more_conv
        self.use_conv=use_conv
        self.encoders = []
        self.positional_encodings = [] 
        self.encoder_block_num = len(encoder)
        self.encoder_keys = []
        self.feature_map_z = feature_map_z
        self.encoder_embed_dims = encoder_embed_dims
        self.dilations = dilations
        self.paddings=paddings
        self.norm_cfg_3d=norm_cfg_3d
        self.act_cfg=act_cfg
        for encoder_key in encoder:
            self.encoder_keys.append(encoder_key)
            self.encoders.append(build_transformer_layer_sequence(encoder[encoder_key]))
            self.positional_encodings.append(build_positional_encoding(position[encoder_key]))
        
        # register model
        for i, layer in enumerate(self.encoders):
           self.add_module('encoder_{}'.format(i), layer)
        for i, layer in enumerate(self.positional_encodings):
           self.add_module('pos_{}'.format(i), layer)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.embed_dim_ratios=[ embed_dims//dim for dim in encoder_embed_dims]

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.decoder_on_bev = decoder_on_bev
        self.bev_z = bev_z
        self.two_stage_num_proposals = two_stage_num_proposals 
        
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        
        if self.decoder_on_bev:
            voxel2bev = []
            last_feature = self.feature_map_z[-1]*self.encoder_embed_dims[-1]
            # mid_num = last_feature//2  # exp7, exp8
            # voxel2bev.append(nn.Linear(last_feature, mid_num))
            # voxel2bev.append(nn.ReLU(inplace=True))
            # voxel2bev.append(nn.Linear(mid_num, self.embed_dims))
            # voxel2bev.append(nn.ReLU(inplace=True))
            # voxel2bev.append(nn.LayerNorm(self.embed_dims))
            # self.voxel2bev = nn.Sequential(*voxel2bev)
            
            mid_num = last_feature
            voxel2bev.append(nn.Linear(last_feature, mid_num))
            voxel2bev.append(nn.ReLU(inplace=True))
            voxel2bev.append(nn.Linear(mid_num, self.embed_dims))
            voxel2bev.append(nn.ReLU(inplace=True))
            voxel2bev.append(nn.LayerNorm(self.embed_dims))
            self.voxel2bev = nn.Sequential(*voxel2bev)
        
        # mid-stage bev->voxe->voxel-> voxel
        if self.use_conv:
            self.convs=[]
            for i in range(self.encoder_block_num-1):
                if self.more_conv:
                    conv = nn.Sequential(
                        ConvModule(
                            self.encoder_embed_dims[i],
                            self.encoder_embed_dims[i],
                            kernel_size=3,
                            stride=1,
                            padding=self.paddings[i],
                            dilation=self.dilations[i],
                            # bias=use_bias_3d,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=self.norm_cfg_3d,
                            act_cfg=self.act_cfg),
                        ConvModule(
                            self.encoder_embed_dims[i],
                            self.encoder_embed_dims[i + 1],
                            kernel_size=[2, 3, 3],
                            stride=[2, 1, 1],
                            padding=[0, 1, 1],
                            # bias=use_bias_3d,
                            conv_cfg=dict(type='ConvTranspose3d'),
                            norm_cfg=self.norm_cfg_3d,
                            act_cfg=self.act_cfg),
                        ConvModule(
                            self.encoder_embed_dims[i + 1],
                            self.encoder_embed_dims[i + 1],
                            kernel_size=3,
                            stride=1,
                            padding=self.paddings[i+1],
                            dilation=self.dilations[i+1],
                            # bias=use_bias_3d,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=self.norm_cfg_3d,
                            act_cfg=self.act_cfg),
                    )
                else:
                    conv=nn.Sequential(
                        ConvModule(
                            self.encoder_embed_dims[i],
                            self.encoder_embed_dims[i+1],
                            kernel_size=[2, 3, 3],
                            stride=[2, 1, 1],
                            padding=[0, 1, 1],
                            # bias=use_bias_3d,
                            conv_cfg=dict(type='ConvTranspose3d'),
                            norm_cfg=self.norm_cfg_3d,
                            act_cfg=self.act_cfg),
                        ConvModule(
                            self.encoder_embed_dims[i+1],
                            self.encoder_embed_dims[i+1],
                            kernel_size=3,
                            stride=1,
                            padding=self.paddings[i],
                            dilation=self.dilations[i],
                            # bias=use_bias_3d,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=self.norm_cfg_3d,
                            act_cfg=self.act_cfg),
                    )

                self.convs.append(conv)
                self.add_module('pyramid_convs_{}'.format(i), conv)

        else:
            self.bev_voxel_transfers = []
            for i in range(self.encoder_block_num-1):
                fc1 = self.encoder_embed_dims[i]*self.feature_map_z[i]
                fc2 = self.encoder_embed_dims[i+1]*self.feature_map_z[i+1]
                block = nn.Sequential(
                    nn.Linear(fc1, fc2),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(fc2),
                )
                self.bev_voxel_transfers.append(block)
                self.add_module('bev_voxel_transfers_{}'.format(i), block)
        
        self.image_feature_map_1_2 = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.ReLU(inplace=True),
        )
        self.image_feature_map_1_4 = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//4),
            nn.ReLU(inplace=True),
        )
        if 8 in self.embed_dim_ratios:
            self.image_feature_map_1_8 = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//8),
                nn.ReLU(inplace=True),
            )
        else:
            self.image_feature_map_1_8 = None

        if 16 in self.embed_dim_ratios:
            self.image_feature_map_1_16 = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//16),
                nn.ReLU(inplace=True),
            )
        else:
            self.image_feature_map_1_16 = None
        
            
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        if self.decoder_on_bev:
            xavier_init(self.voxel2bev, distribution='uniform', bias=0.)
        if not self.use_conv:
            for block in self.bev_voxel_transfers:
                xavier_init(block, distribution='uniform', bias=0.)
        xavier_init(self.image_feature_map_1_2, distribution='uniform', bias=0.)
        xavier_init(self.image_feature_map_1_4, distribution='uniform', bias=0.)
        if self.image_feature_map_1_8 is not None:
            xavier_init(self.image_feature_map_1_8, distribution='uniform', bias=0.)
        if self.image_feature_map_1_16 is not None:
            xavier_init(self.image_feature_map_1_16, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_voxel_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_z,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # (num_query, bs, embed_dims)
        bev_pos = None
        # bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # (num_query, bs, embed_dims)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # (2, bs) -> (bs, 2)

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]  
        bev_queries = bev_queries + can_bus * self.use_can_bus  # (query_num, bs, embed_dims)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten_original = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        feat_flatten_map1_2 = self.image_feature_map_1_2(feat_flatten_original)
        feat_flatten_map1_4 = self.image_feature_map_1_4(feat_flatten_original)
        if self.image_feature_map_1_8 is not None:
            feat_flatten_map1_8 = self.image_feature_map_1_8(feat_flatten_original)
        else:
            feat_flatten_map1_8 = None
        if self.image_feature_map_1_16 is not None:
            feat_flatten_map1_16 = self.image_feature_map_1_16(feat_flatten_original)
        else:
            feat_flatten_map1_16 = None
        
        block_features = []
        for block_index in range(self.encoder_block_num):
            # encoder： BEV -> Voxeli -> Voxelj -> Voxelk
            # print('bev_query.shape:', block_index, bev_queries.shape)
            block_bev_z = self.feature_map_z[block_index]
            block_embed_dims = self.encoder_embed_dims[block_index]
            if block_bev_z == 1:
                bev_mask = torch.zeros((bs, bev_h, bev_w),
                            device=bev_queries.device).to(bev_queries.dtype)
            else:
                bev_mask = torch.zeros((bs, block_bev_z, bev_h, bev_w),
                            device=bev_queries.device).to(bev_queries.dtype)
            pos = self.positional_encodings[block_index](bev_mask).to(bev_queries.dtype)  # (bs, embed_dims, h, w)
            pos = pos.flatten(2).permute(2, 0, 1)  # (query_num, bs, embed_dims)
            
            if block_embed_dims == self.embed_dims:
                feat_flatten = feat_flatten_original
            elif block_embed_dims*2 == self.embed_dims:
                feat_flatten = feat_flatten_map1_2
            elif block_embed_dims*4 == self.embed_dims:
                feat_flatten = feat_flatten_map1_4
            elif block_embed_dims*8 == self.embed_dims:
                feat_flatten = feat_flatten_map1_8
            elif block_embed_dims*16 == self.embed_dims:
                feat_flatten = feat_flatten_map1_16
            
            if prev_bev is not None:  # (bs, num_query, embed_dims)
                stage_prev_bev = prev_bev[block_index]
                if block_bev_z == 1:  # 2D BEV
                    if stage_prev_bev.shape[1] == bev_h * bev_w:
                        stage_prev_bev = stage_prev_bev.permute(1, 0, 2)  # (num_query, bs, embed_dims)
                    if self.rotate_prev_bev:
                        for i in range(bs):
                            # num_prev_bev = prev_bev.size(1)
                            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                            tmp_prev_bev = stage_prev_bev[:, i].reshape(
                                bev_h, bev_w, -1).permute(2, 0, 1)  # (embed_dims, bev_h, bev_w)
                            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                                center=self.rotate_center)  # TODO: for 3D voxel
                            tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                                bev_h * bev_w, 1, -1)
                            stage_prev_bev[:, i] = tmp_prev_bev[:, 0]
                        
                else:  # 3D Voxel
                    if stage_prev_bev.shape[1] == block_bev_z* bev_h * bev_w:
                        stage_prev_bev = stage_prev_bev.permute(1, 0, 2)  # (num_query, bs, embed_dims)
                    if self.rotate_prev_bev:  # revise for 3D feature map
                        for i in range(bs):
                            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                            tmp_prev_bev = stage_prev_bev[:, i].reshape(block_bev_z, bev_h, bev_w, -1).permute(3, 0, 1, 2)  # (embed_dims, bev_z, bev_h, bev_w)
                            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                            tmp_prev_bev = tmp_prev_bev.permute(1, 2, 3, 0).reshape(block_bev_z * bev_h * bev_w, 1, -1)
                            stage_prev_bev[:, i] = tmp_prev_bev[:, 0] 
            else:
                stage_prev_bev = None

            # print()
            # print('bev_queries',bev_queries.shape)
            # print()

            output = self.encoders[block_index](
                bev_queries,
                feat_flatten,
                feat_flatten,
                bev_z=block_bev_z,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=stage_prev_bev,
                shift=shift,
                **kwargs
            )
            
            block_features.append(output)
            if self.use_conv:
                if block_index < self.encoder_block_num - 1:  # bev-> voxel or voxel_i -> voxel_j
                    bev_queries = output.view(block_bev_z, bev_h, bev_w, bs, self.encoder_embed_dims[block_index])
                    bev_queries = bev_queries.permute(3,4,0,1,2)
                    # bev_queries = bev_queries.flatten(3)  # (bev_h, bev_w, bs, embed_dims1*z1)
                    bev_queries = self.convs[block_index](bev_queries)
                    bev_queries = bev_queries.view(bs,self.encoder_embed_dims[block_index + 1],
                                                   self.feature_map_z[block_index + 1],bev_h, bev_w,
                                                   )
                    bev_queries = bev_queries.permute(2,3,4,0,1)
                    bev_queries = bev_queries.reshape(-1, bs, self.encoder_embed_dims[block_index + 1])  # (num_query, bs, embed_dims)
            else:
                if block_index < self.encoder_block_num-1:  # bev-> voxel or voxel_i -> voxel_j
                    bev_queries = output.view(block_bev_z, bev_h, bev_w, bs, self.encoder_embed_dims[block_index])
                    bev_queries = bev_queries.permute(1, 2, 3, 0, 4)
                    bev_queries = bev_queries.flatten(3)  # (bev_h, bev_w, bs, embed_dims1*z1)
                    bev_queries = self.bev_voxel_transfers[block_index](bev_queries)  # (bev_h, bev_w, bs, embed_dims2*z2)
                    bev_queries = bev_queries.view(bev_h, bev_w, bs, self.feature_map_z[block_index+1], self.encoder_embed_dims[block_index+1])
                    bev_queries = bev_queries.permute(3, 0, 1, 2, 4)
                    bev_queries = bev_queries.reshape(-1, bs, self.encoder_embed_dims[block_index+1])  # (num_query, bs, embed_dims)
                
        return block_features  # is a list 

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_z,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        block_features = self.get_voxel_features(
            mlvl_feats,
            bev_queries,
            bev_z,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # voxel_embed shape: (bs, num_query, embed_dims)

        return block_features


