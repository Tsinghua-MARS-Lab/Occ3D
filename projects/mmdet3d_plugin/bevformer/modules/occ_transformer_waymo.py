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
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule, caffe2_xavier_init
from .unet import MYASPPHead


@TRANSFORMER.register_module()
class OccTransformerWaymo(BaseModule):
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
                 volume_flag=True,
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
                 use_conv=False,
                 rotate_center=[100, 100],
                 num_classes=18,
                 out_dim=32,
                 pillar_h=16,
                 total_z=32,
                 iter_encoders=None,
                 topK_method='foreground',
                 act_cfg=dict(type='ReLU',inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):
        self.volume_flag = volume_flag
        super(OccTransformerWaymo, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        if iter_encoders is not None: self.iter_encoders = torch.nn.ModuleList([build_transformer_layer_sequence(encoder) for encoder in iter_encoders])
        self.topK_method = topK_method

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_3d=use_3d
        self.use_conv=use_conv
        self.pillar_h = pillar_h
        self.out_dim=out_dim
        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder = []
                conv_cfg = dict(type='Conv2d')
                conv_num = 3
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
                _out_dim = self.embed_dims*self.pillar_h
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
                    nn.Linear(_out_dim//total_z, self.embed_dims//2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims//2,num_classes),
                )

            else:
                _out_dim = self.embed_dims
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims//total_z, self.embed_dims),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims, _out_dim),
                )
                raise NotImplementedError
        else:
            use_bias_3d = norm_cfg_3d is None
            middle_dims=32 # decrease memory cost
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
            self.decoder = nn.Sequential(*decoder_layers)
            self.predicter = nn.Sequential(
                nn.Linear(_out_dim, self.embed_dims//2),
                nn.Softplus(),
                nn.Linear(self.embed_dims//2, 2), # binary classify
            )
            if iter_encoders is not None:
                iter_decoders = []
                iter_predicters = []
                for iter_i in range(len(self.iter_encoders)):
                    middle_dims=32 # decrease memory cost
                    decoder_layers = []
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
                    for kk in range(2):
                        decoder_layers += [
                            ConvModule(
                                middle_dims,
                                middle_dims,
                                kernel_size=[1, 3, 3],
                                stride=1,
                                padding=[0, 1, 1],
                                bias=use_bias_3d,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg_3d,
                                act_cfg=act_cfg),
                            ConvModule(
                                middle_dims,
                                middle_dims,
                                kernel_size=[3, 1, 3],
                                stride=1,
                                padding=[1, 0, 1],
                                bias=use_bias_3d,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg_3d,
                                act_cfg=act_cfg),
                            ConvModule(
                                middle_dims,
                                middle_dims,
                                kernel_size=[3, 3, 1],
                                stride=1,
                                padding=[1, 1, 0],
                                bias=use_bias_3d,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg_3d,
                                act_cfg=act_cfg),
                        ]
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
                    decoder = nn.Sequential(*decoder_layers)
                    predicter = nn.Sequential(
                        nn.Linear(_out_dim, self.embed_dims//2),
                        nn.Softplus(),
                        nn.Linear(self.embed_dims//2, num_classes if iter_i==len(iter_encoders)-1 else 2), # binary classify or semantic seg
                    )
                    iter_decoders.append(decoder)
                    iter_predicters.append(predicter)
                self.iter_decoders = nn.ModuleList(iter_decoders)
                self.iter_predicters = nn.ModuleList(iter_predicters)  

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
        #breakpoint()
        if not self.volume_flag: assert bev_z == 1

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)   #[20000, 1, 256]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)   #[20000, 1, 256]
        # obtain rotation angle and shift with ego motion

        grid_length_y = grid_length[0]  #0.4
        grid_length_x = grid_length[1]  #0.4

        # assert self.rotate_prev_bev==False and self.use_can_bus==False, "TODO JT add can bus signals"
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
                    prev_bev = prev_bev.view(bs,-1,bev_h * bev_w).permute(2, 0, 1)
                elif len(prev_bev.shape) == 5:
                    prev_bev = prev_bev.view(bs, -1,bev_z* bev_h * bev_w).permute(2, 0, 1)
            else:
                if prev_bev.shape[1] == bev_h * bev_w:
                    prev_bev = prev_bev.permute(1, 0, 2)
                elif len(prev_bev.shape) == 4:
                    prev_bev = prev_bev.view(bs,-1,bev_h * bev_w).permute(2, 0, 1)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    if self.volume_flag:
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            bev_z, bev_h, bev_w, -1).permute(3, 0, 1, 2)
                        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                            center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.permute(1, 2,3, 0).reshape(
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
        #breakpoint()
        feat_flatten = torch.cat(feat_flatten, 2)   #[5, 1, 12750, 256]
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

        return bev_embed, feat_flatten, bev_pos, bev_queries, spatial_shapes, level_start_index, shift

    @staticmethod
    def convert_gt(binary_gt, preds):
        bs, W, H, D = binary_gt.shape
        _bs, _W, _H, _D = preds.shape
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
        return binary_gt
    

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
                voxel_semantics=None, mask_lidar=None, mask_camera=None,
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
        #breakpoint()
        if not self.volume_flag: assert bev_z == 1
        _bev_embed, _feat_flatten, _bev_pos, _bev_queries, _spatial_shapes, _level_start_index, _shift = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_z,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,   #[1, 20000, 256]
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        # !!!!!!!!!! assert bev_embed in [bs, DHW, C] order 
        #breakpoint()
        previous_bev_embed_bs_DHW_C = _bev_embed # used for history bev in test 
        bev_embed_bs_DHW_C  = _bev_embed   #[1, 20000, 256]
        feat_flatten = _feat_flatten    #[5, 12750, 1, 256]
        spatial_shapes = _spatial_shapes    #[ 80, 120]=>[10,15]
        level_start_index = _level_start_index
        shift = _shift
        bs = mlvl_feats[0].size(0)
        extra = {}
        if self.use_3d:
            zz = bev_z if self.volume_flag else self.pillar_h
            bev_embed_bs_C_D_H_W = bev_embed_bs_DHW_C.permute(0, 2, 1).view(bs, -1, zz, bev_h, bev_w)
            res_bs_C_D_H_W = self.decoder(bev_embed_bs_C_D_H_W)
            bev_embed_bs_C_D_H_W = bev_embed_bs_C_D_H_W + res_bs_C_D_H_W
            bev_embed_bs_W_H_D_C = bev_embed_bs_C_D_H_W.permute(0,4,3,2,1)
            outputs_bs_W_H_D_C = self.predicter(bev_embed_bs_W_H_D_C)

            bev_embed_list = [bev_embed_bs_W_H_D_C]
            outputs_list = [outputs_bs_W_H_D_C]
            topk_dim = 1 # 1 for foreground
            for iter_i, iter_encoder in enumerate(self.iter_encoders):
                # topk voxel
                topk_ratio = iter_encoder.topk_ratio
                if self.topK_method == 'foreground' or self.topK_method == 'no_cross_atten' or self.topK_method == 'no_conv':
                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(outputs_onedim_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio)
                    indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True
                elif self.topK_method == 'ambiguous':
                    scores_bs_W_H_D = outputs_bs_W_H_D_C.softmax(dim=-1)[:, :, :, :, topk_dim]
                    ambiguous_bs_W_H_D = 1 - torch.abs(0.5 - scores_bs_W_H_D)
                    ambiguous_squeeze_bsWHD = ambiguous_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(ambiguous_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(ambiguous_squeeze_bsWHD.shape[0] * topk_ratio)
                    indices = torch.topk(ambiguous_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True                    
                elif self.topK_method == 'mixed':
                    scores_bs_W_H_D = outputs_bs_W_H_D_C.softmax(dim=-1)[:, :, :, :, topk_dim]
                    ambiguous_bs_W_H_D = 1 - torch.abs(0.5 - scores_bs_W_H_D)
                    ambiguous_squeeze_bsWHD = ambiguous_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(ambiguous_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(ambiguous_squeeze_bsWHD.shape[0] * topk_ratio * 0.5)
                    indices = torch.topk(ambiguous_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True

                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio * 0.5)
                    indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True
                elif self.topK_method == 'random':
                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(outputs_onedim_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio)
                    # indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    indices = torch.randint(low=0, high=outputs_squeeze_bsWHD.shape[0], size=(topk,)).to(topk_mask_squeeze_bsWHD.device)
                    topk_mask_squeeze_bsWHD[indices] = True 
                else:
                    raise NotImplementedError
                # DEBUG_TMP
                DEBUG_TMP=False
                if DEBUG_TMP:
                    binary_gt_bs_W_H_D = voxel_semantics != 23
                    binary_gt_bs_W_H_D = self.convert_gt(binary_gt_bs_W_H_D, topk_mask_bs_W_H_D)
                    binary_gt_bsWHD = binary_gt_bs_W_H_D.reshape(-1)
                    topk_mask_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    recall = torch.logical_and(binary_gt_bsWHD, topk_mask_bsWHD).sum() / binary_gt_bsWHD.sum()
                    percision = torch.logical_and(binary_gt_bsWHD, topk_mask_bsWHD).sum() / topk_mask_bsWHD.sum()
                    print("### HERE ### {} {} {}".format(iter_i, recall, percision))

                # upsample
                bs, C, D, H, W = bev_embed_bs_C_D_H_W.shape
                tg_D, tg_H, tg_W = iter_encoder.DHW
                topk_mask_bs_D_H_W = topk_mask_bs_W_H_D.permute(0, 3, 2, 1)
                topk_mask_bs_C_D_H_W = topk_mask_bs_D_H_W.unsqueeze(dim=1) # => bs,1,D,H,W 
                update_bev_embed_bs_C_D_H_W  = torch.nn.functional.interpolate(bev_embed_bs_C_D_H_W, size=(tg_D, tg_H, tg_W), mode='trilinear', align_corners=True)
                update_topk_bs_C_D_H_W  = torch.nn.functional.interpolate(topk_mask_bs_C_D_H_W.float(), size=(tg_D, tg_H, tg_W), mode='trilinear', align_corners=True)
                update_topk_bs_C_D_H_W = update_topk_bs_C_D_H_W > 0
                update_topk_bs_D_H_W = update_topk_bs_C_D_H_W.squeeze(dim=1)
                update_bev_embed_bs_C_DHW = update_bev_embed_bs_C_D_H_W.reshape(bs, C, tg_D*tg_H*tg_W)
                update_bev_embed_DHW_bs_C = update_bev_embed_bs_C_DHW.permute(2, 0, 1) # => (DHW, bs, C)
                update_topk_bs_DHW = update_topk_bs_D_H_W.reshape(bs, tg_D*tg_H*tg_W)
                bev_embed_bs_DHW_C = iter_encoder(
                    update_bev_embed_DHW_bs_C,
                    feat_flatten,
                    feat_flatten,
                    bev_z=tg_D,
                    bev_h=tg_H,
                    bev_w=tg_W,
                    bev_pos=None, # bev_pos, # TODO JT try update bev_pos ?
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    prev_bev=None, # prev_bev, # TODO JT try update prev_bev ?
                    shift=shift,
                    topk_mask=update_topk_bs_DHW,
                    **kwargs
                )
                update_bev_embed_bs_DHW_C = update_bev_embed_bs_C_DHW.permute(0, 2, 1)
                if self.topK_method != 'no_cross_atten':
                    bev_embed_bs_DHW_C = bev_embed_bs_DHW_C + update_bev_embed_bs_DHW_C
                else:
                    bev_embed_bs_DHW_C = update_bev_embed_bs_DHW_C
                bev_embed_bs_C_D_H_W = bev_embed_bs_DHW_C.permute(0, 2, 1).view(bs, -1, tg_D, tg_H, tg_W)
                if self.topK_method != 'no_conv':
                    res_bs_C_D_H_W = self.iter_decoders[iter_i](bev_embed_bs_C_D_H_W)
                    bev_embed_bs_C_D_H_W = bev_embed_bs_C_D_H_W + res_bs_C_D_H_W
                bev_embed_bs_W_H_D_C = bev_embed_bs_C_D_H_W.permute(0,4,3,2,1)
                outputs_bs_W_H_D_C = self.iter_predicters[iter_i](bev_embed_bs_W_H_D_C)
                outputs = outputs_bs_W_H_D_C
                # previous binay seg, last semantic seg
                if iter_i != len(self.iter_encoders)-1:
                    bev_embed_list.append(bev_embed_bs_W_H_D_C)
                    outputs_list.append(outputs_bs_W_H_D_C)

            extra['bev_embed_list'] = bev_embed_list
            extra['outputs_list'] = outputs_list
        
        elif self.use_conv:
            #breakpoint()
            bev_embed = bev_embed_bs_DHW_C
            total_z = self.total_z
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w)  #[1, 20000, 256]
            outputs = self.decoder(bev_embed)   #[1, 1024, 200, 100]
            outputs = outputs.view(bs, -1, self.total_z, bev_h, bev_w).permute(0,4,3,2,1).contiguous()  #[1, 100, 200, 32, 32]
            outputs = outputs.reshape(bs*bev_w*bev_h*total_z, -1)    #([640000, 32]
            outputs = self.predicter(outputs)    #[640000, 16]
            outputs = outputs.view(bs, bev_w, bev_h, total_z, -1)   #[1, 100, 200, 32, 16]
        else:
            #breakpoint()
            bev_embed = bev_embed_bs_DHW_C
            total_z = self.total_z
            C = bev_embed.shape[-1]//total_z
            bev_embed = bev_embed.reshape(bs, bev_h, bev_w, total_z, C)
            bev_embed = bev_embed.reshape(-1, C)
            outputs = self.decoder(bev_embed)
            outputs = self.predicter(outputs)
            outputs = outputs.view(bs, bev_h, bev_w, total_z, self.out_dim)
            # outputs = outputs.premute()
        #breakpoint()
        # print('outputs',type(outputs))
        return previous_bev_embed_bs_DHW_C, outputs, extra