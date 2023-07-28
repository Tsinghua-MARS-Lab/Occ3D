
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
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderWaymo(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, volume_flag=True, pc_range=None, num_points_in_voxel=-1, num_voxel=-1, num_points_in_pillar=-1, return_intermediate=False, dataset_type='waymo',
                 **kwargs):

        super(BEVFormerEncoderWaymo, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if volume_flag:
            assert num_voxel != -1 and num_points_in_voxel != -1
        else:
            assert num_points_in_pillar != -1
        self.n_p_in_voxel=num_points_in_voxel
        self.n_voxel=num_voxel        
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.volume_flag = volume_flag
        self.dataset_type = dataset_type

    @staticmethod
    def get_reference_points(volume_flag, H, W, Z, n_p_in_voxel, n_voxel, num_points_in_pillar, dim='3d', bs=1, device='cuda', dtype=torch.float):
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
        #breakpoint()
        if volume_flag:
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
                return ref_2d   #[1, 4, 20000, 3]
        else:
            # reference points in 3D space, used in spatial cross-attention (SCA)
            if dim == '3d':
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                    device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                    device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                    device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
                ref_3d = torch.stack((xs, ys, zs), -1)
                ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  #shape: (bs,num_points_in_pillar,h*w,3)
                return ref_3d

            # reference points on 2D bev plane, used in temporal self-attention (TSA).
            elif dim == '2d':
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=dtype, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=dtype, device=device)
                )
                ref_y = ref_y.reshape(-1)[None] / H
                ref_x = ref_x.reshape(-1)[None] / W
                ref_2d = torch.stack((ref_x, ref_y), -1)
                ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
                return ref_2d   #[1, 20000, 1, 2]

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas, dataset_type='waymo'):
        lidar2img = []
        #breakpoint()
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

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

        if dataset_type=='waymo':
            reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)
        elif dataset_type=='nuscenes':
            ego2lidar=img_metas[0]['ego2lidar']
            ego2lidar = reference_points.new_tensor(ego2lidar)
            ego2lidar=ego2lidar.view(1,1,1,1,4,4).repeat(D,1,num_cam,num_query,1,1)
            reference_points_cam = torch.matmul(torch.matmul(lidar2img.to(torch.float32),ego2lidar.to(torch.float32)),reference_points.to(torch.float32)).squeeze(-1)    
        else:
            raise NotImplementedError
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
                topk_mask=None,
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
        #breakpoint()
        if self.volume_flag: _dim = 3
        else: _dim = 2
        output = bev_query  #[20000, 1, 256]
        intermediate = []

        ref_3d = self.get_reference_points(
            self.volume_flag, bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.n_p_in_voxel, self.n_voxel, self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            self.volume_flag, bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.n_p_in_voxel, self.n_voxel, self.num_points_in_pillar, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'], self.dataset_type)  #[5, 1, 20000, 4, 2], [5, 1, 20000, 4]
        if topk_mask is not None:
            # bev_mask: [cams, bs, DHW, n_p_in_voxel]
            # topk_mask: [bs, DHW]
            bs, DHW = topk_mask.shape
            num_cam = bev_mask.shape[0]
            topk_mask = topk_mask.reshape(1, bs, DHW, 1).repeat(num_cam, 1, 1, self.n_p_in_voxel)
            bev_mask_update = torch.logical_and(bev_mask, topk_mask)
            # print(bev_mask.sum(), topk_mask.sum(), bev_mask_update.sum())
            bev_mask = bev_mask_update

        # # DEBUG_TMP
        # import pickle as pkl
        # with open('work_dirs/ref_waymo.pkl', 'wb') as f:
        #     pkl.dump(
        #         {"ref_3d":ref_3d, "ref_2d": ref_2d, "reference_points_cam": reference_points_cam, "bev_mask": bev_mask},
        #         f
        #     )
        # raise Exception


        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        if self.volume_flag:
            shift_ref_2d = ref_2d  # .clone()
            shift3d = shift.new_zeros(1, 3)
            shift3d[:, :2] = shift
            shift_ref_2d += shift3d[:, None, None, :]
        else:
            shift_ref_2d = ref_2d  # .clone() [1, 20000, 1, 2]
            shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        if bev_pos is not None: bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, _dim)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, _dim)  #[2, 20000, 1, 2]
        #breakpoint()
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

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderTopKWaymo(BEVFormerEncoderWaymo):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, DHW=[16, 200, 200], topk_ratio=0.05, **kwargs):
        super(BEVFormerEncoderTopKWaymo, self).__init__(*args, **kwargs)
        self.topk_ratio = topk_ratio
        self.DHW = DHW
    

@TRANSFORMER_LAYER.register_module()
class OccFormerLayerWaymo(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 volume_flag=True,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        self.volume_flag = volume_flag
        super(OccFormerLayerWaymo, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #     ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_z=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        #breakpoint()
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                if self.volume_flag:
                    _spatial_shapes=torch.tensor([[bev_z,bev_h, bev_w]], device=query.device)
                else:
                    _spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device)
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=_spatial_shapes,   #[200, 100]
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query
                # #breakpoint()
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
        
            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
