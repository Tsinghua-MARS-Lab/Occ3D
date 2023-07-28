# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
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
from mmcv.cnn import PLUGIN_LAYERS, Conv2d, Conv3d, ConvModule, caffe2_xavier_init
from .unet import MYASPPHead

@TRANSFORMER.register_module()
class OccTransformerHeavyDecoder(BaseModule):
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
                 volume_flag=False,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_3d=False,
                 use_conv=True,
                 rotate_center=[100, 100],

                 num_classes=18,
                 out_dim=32,
                 pillar_h=16,
                 total_z=32,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):
        self.volume_flag = volume_flag
        super(OccTransformerHeavyDecoder, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_3d = use_3d
        self.use_conv = use_conv
        self.pillar_h = pillar_h
        self.out_dim = out_dim
        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder = []
                conv_cfg = dict(type='Conv2d')
                conv_num = 5

                # conv module
                decoder_layers = []
                for ii in range(conv_num):
                    decoder_layers.append(
                        ConvModule(
                            self.embed_dims,
                            self.embed_dims,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=use_bias,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg)
                    )
                # aspp
                decoder_layers.append(
                    MYASPPHead(
                        is_volume=False,
                        in_channels=self.embed_dims,
                        in_index=3,
                        channels=self.embed_dims,
                        dilations=(1, 3, 6, 9),
                        dropout_ratio=0.1,
                        num_classes=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    )
                )
                # deconv to origin shape
                _out_dim = self.embed_dims * self.pillar_h
                decoder_layers.append(
                    ConvModule(
                        self.embed_dims,
                        _out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
                self.decoder = nn.Sequential(*decoder_layers)
                self.predicter = nn.Sequential(
                    nn.Linear(_out_dim // total_z, self.embed_dims // 2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims // 2, num_classes),
                )

            else:
                _out_dim = self.embed_dims
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims // total_z, self.embed_dims),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims, _out_dim),
                )
                raise NotImplementedError
        else:
            use_bias_3d = norm_cfg_3d is None
            middle_dims = 32  # decrease memory cost
            decoder_layers = []
            conv_cfg = dict(type='Conv3d')
            decoder_layers.append(
                ConvModule(
                    self.embed_dims,
                    middle_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=use_bias_3d,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg)
            )
            decoder_layers.append(
                MYASPPHead(
                    in_channels=middle_dims,
                    in_index=3,
                    channels=middle_dims,
                    dilations=(1, 3, 6, 9),
                    dropout_ratio=0.1,
                    num_classes=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg_3d,
                    align_corners=False,
                    # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                )
            )
            decoder_layers.append(
                ConvModule(
                    middle_dims,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=use_bias_3d,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg)
            )

            _out_dim = self.embed_dims

            # kk = total_z//pillar_h
            # for i in range(kk):
            #     _out_dim = self.embed_dims
            #     decoder_layers.append(
            #         ConvModule(
            #             self.embed_dims,
            #             self.embed_dims,
            #             kernel_size=[1, 3, 3],
            #             stride=[1, 1, 1],
            #             padding=[0, 1, 1],
            #             bias=use_bias_3d,
            #             conv_cfg=dict(type='ConvTranspose3d'),
            #             norm_cfg=norm_cfg_3d,
            #             act_cfg=act_cfg)
            #     )
            #     decoder_layers.append(
            #         ConvModule(
            #             self.embed_dims,
            #             self.embed_dims,
            #             kernel_size=[3, 1, 1],
            #             stride=[1, 1, 1],
            #             padding=[1, 0, 0],
            #             bias=use_bias_3d,
            #             conv_cfg=dict(type='ConvTranspose3d'),
            #             norm_cfg=norm_cfg_3d,
            #             act_cfg=act_cfg)
            #     )
            #     decoder_layers.append(
            #         ConvModule(
            #             self.embed_dims,
            #             _out_dim if i == kk -1 else self.embed_dims,
            #             kernel_size=[2, 1, 1],
            #             stride=[2, 1, 1],
            #             padding=[0, 0, 0],
            #             bias=use_bias_3d,
            #             conv_cfg=dict(type='ConvTranspose3d'),
            #             norm_cfg=norm_cfg_3d,
            #             act_cfg=act_cfg)
            #     )

            self.decoder = nn.Sequential(*decoder_layers)
            self.predicter = nn.Sequential(
                nn.Linear(_out_dim, self.embed_dims // 2),
                nn.Softplus(),
                nn.Linear(self.embed_dims // 2, num_classes),
            )
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        self.total_z = total_z

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

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,

            bev_h,
            bev_w,
            bev_z=None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        if not self.volume_flag: assert bev_z == 1 or bev_z is None

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        # obtain rotation angle and shift with ego motion

        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]

        if self.use_can_bus:
            delta_x = np.array([each['can_bus'][0]
                                for each in kwargs['img_metas']])
            delta_y = np.array([each['can_bus'][1]
                                for each in kwargs['img_metas']])
            ego_angle = np.array(
                [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])

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
                [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy
        else:
            shift = bev_queries.new_zeros((1, 2))

        if prev_bev is not None:
            if self.volume_flag:
                if prev_bev.shape[1] == bev_h * bev_w * bev_z:
                    prev_bev = prev_bev.permute(1, 0, 2)
                elif len(prev_bev.shape) == 4:
                    prev_bev = prev_bev.view(bs, -1, bev_h * bev_w).permute(2, 0, 1)
                elif len(prev_bev.shape) == 5:
                    prev_bev = prev_bev.view(bs, -1, bev_z * bev_h * bev_w).permute(2, 0, 1)
            else:
                if prev_bev.shape[1] == bev_h * bev_w:
                    prev_bev = prev_bev.permute(1, 0, 2)
                elif len(prev_bev.shape) == 4:
                    prev_bev = prev_bev.view(bs, -1, bev_h * bev_w).permute(2, 0, 1)

            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    if self.volume_flag:
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            bev_z, bev_h, bev_w, -1).permute(3, 0, 1, 2)
                        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                              center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.permute(1, 2, 3, 0).reshape(
                            bev_z * bev_h * bev_w, 1, -1)
                    else:
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            bev_h, bev_w, -1).permute(2, 0, 1)
                        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                              center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                            bev_h * bev_w, 1, -1)
                    # print()
                    # print('prev_bev',prev_bev.shape,tmp_prev_bev.shape)
                    # print()
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # TODO JT add can bus signals
        if self.use_can_bus:
            can_bus = bev_queries.new_tensor(
                [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

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

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_z=bev_z,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,

                bev_h,
                bev_w,
                bev_z=None,
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
        if not self.volume_flag: assert bev_z == 1 or bev_z is None
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,

            bev_h,
            bev_w,
            bev_z=bev_z,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        if self.use_3d:
            zz = bev_z if self.volume_flag else self.pillar_h
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, zz, bev_h, bev_w)
            res = self.decoder(bev_embed)
            bev_embed = bev_embed + res
            outputs = bev_embed.permute(0, 4, 3, 2, 1)
            outputs = self.predicter(outputs)
        elif self.use_conv:
            total_z = self.total_z
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w)
            outputs = self.decoder(bev_embed)
            outputs = outputs.view(bs, -1, self.total_z, bev_h, bev_w).permute(0, 4, 3, 2, 1).contiguous()
            outputs = outputs.reshape(bs * bev_w * bev_h * total_z, -1)
            outputs = self.predicter(outputs)
            outputs = outputs.view(bs, bev_w, bev_h, total_z, -1)
        else:
            total_z = self.total_z
            C = bev_embed.shape[-1] // total_z
            bev_embed = bev_embed.reshape(bs, bev_h, bev_w, total_z, C)
            bev_embed = bev_embed.reshape(-1, C)
            outputs = self.decoder(bev_embed)
            outputs = self.predicter(outputs)
            outputs = outputs.view(bs, bev_h, bev_w, total_z, self.out_dim)
            # outputs = outputs.premute()

        # print('outputs',type(outputs))
        return bev_embed, outputs
