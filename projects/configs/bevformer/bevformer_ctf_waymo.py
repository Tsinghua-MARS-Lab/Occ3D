_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
point_cloud_range = [0, -40, -5., 40, 40, 7.8]
voxel_size=[0.4, 0.4, 0.4]
num_classes = 16
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
class_weight_binary = [5.314075572339673, 1]
class_weight_multiclass = [
    21.996729830048952,
    7.504469780801267, 10.597629961083673, 12.18107968968811, 15.143940258446506, 13.035521328502758, 
    9.861234292376812, 13.64431851057796, 15.121236434460473, 21.996729830048952, 6.201671013759701, 
    5.7420517938838325, 9.768712859518626, 3.4607400626606317, 4.152268220983671, 1.000000000000000,
]
pose_file = 'dataset//waymo_v1.4.0/cam_infos.pkl'
data_root='dataset//waymo_v1.3.1_untar/kitti_format/'
occ_gt_data_root='dataset//waymo_v1.4.0/voxel_go/'
val_pose_file='/public/MARS/datasets/waymo_v1.4.0/cam_infos_vali.pkl'
occ_val_gt_data_root='/public/MARS/datasets/waymo_v1.4.0/voxel_go_vali'

# pose_file='/mnt/petrelfs/zhaohang.p/waymo_occ_gt/cam_infos.pkl'
# data_root='/mnt/petrelfs/zhaohang.p/mmdetection/data/waymo/kitti_format/'
# occ_gt_data_root='/mnt/petrelfs/zhaohang.p/waymo_occ_gt/voxel_gt'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
num_feats = [_dim_//3, _dim_//3 , _dim_ - _dim_//3 - _dim_//3]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 100
total_z = 32
bev_z_ = 4
pillar_h = 4
num_points_in_voxel = 4 # total_z // bev_z_
queue_length = 3 # each sequence contains `queue_length` frames.
TemporalSelfAttention3D_num_points = 4
SpatialCrossAttention_num_points = 4
num_views = 5
FREE_LABEL = 23
load_interval = 5
model = dict(
    type='OccFormerWaymo',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccFormerHeadWaymo',
        FREE_LABEL=FREE_LABEL,
        bev_z=bev_z_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        # loss_occ=dict(
        #     type='FocalLoss',
        #     use_sigmoid=False,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=10.0),
        use_lidar_mask=False,
        use_camera_mask=False,
        # loss_occ= dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=1.0),
        loss_occ=dict(
            ceohem=dict(
                type='CrossEntropyOHEMLoss',
                class_weight=class_weight_multiclass,
                use_sigmoid=False,
                use_mask=False,
                loss_weight=1.0, 
                top_ratio=0.2,
                top_weight=4.0),
            # lovasz=dict(
            #     type='LovaszLoss',
            #     class_weight=class_weight_multiclass,
            #     loss_type='multi_class',
            #     classes='present',
            #     per_image=False,
            #     reduction='none',
            #     loss_weight=1.0)
        ),
        loss_binary_occ=dict(
            ceohem=dict(
                type='CrossEntropyOHEMLoss',
                class_weight=class_weight_binary,
                use_sigmoid=False,
                use_mask=False,
                loss_weight=1.0, 
                top_ratio=0.2,
                top_weight=4.0),
            # lovasz=dict(
            #     type='LovaszLoss',
            #     class_weight=class_weight_binary,
            #     loss_type='multi_class',
            #     classes='present',
            #     per_image=False,
            #     reduction='none',
            #     loss_weight=1.0)
        ),
        transformer=dict(
            type='OccTransformerWaymo',
            num_cams=num_views,
            pillar_h=bev_z_,
            total_z=total_z,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', ),
            norm_cfg_3d=dict(type='BN3d', ),
            use_3d=True,
            use_conv=False,
            rotate_prev_bev=False,
            use_shift=True,
            use_can_bus=False,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoderWaymo',
                num_layers=4,
                num_points_in_voxel=num_points_in_voxel,
                num_voxel=bev_z_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccFormerLayerWaymo',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention3D',
                            embed_dims=_dim_,
                            num_points=TemporalSelfAttention3D_num_points,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=num_views,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention4D',
                                embed_dims=_dim_,
                                num_points=SpatialCrossAttention_num_points,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_dim_*4,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            iter_encoders=[
                dict(
                    type='BEVFormerEncoderTopKWaymo',
                    DHW=[8, 200, 100],
                    topk_ratio=0.2,
                    num_layers=1,
                    num_points_in_voxel=4,
                    num_voxel=8,
                    pc_range=point_cloud_range,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='OccFormerLayerWaymo',
                        attn_cfgs=[
                            # dict(
                            #     type='TemporalSelfAttention3D',
                            #     embed_dims=_dim_,
                            #     num_points=4,
                            #     num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                num_cams=num_views,
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention4D',
                                    embed_dims=_dim_,
                                    num_points=4,
                                    num_levels=_num_levels_),
                                embed_dims=_dim_,
                            )
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_dim_*4,
                            num_fcs=2,
                            ffn_drop=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=('cross_attn', 'norm',
                                        'ffn', 'norm'))),

                dict(
                    type='BEVFormerEncoderTopKWaymo',
                    DHW=[16, 200, 100],
                    topk_ratio=0.2,
                    num_layers=1,
                    num_points_in_voxel=4,
                    num_voxel=16,
                    pc_range=point_cloud_range,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='OccFormerLayerWaymo',
                        attn_cfgs=[
                            # dict(
                            #     type='TemporalSelfAttention3D',
                            #     embed_dims=_dim_,
                            #     num_points=4,
                            #     num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                num_cams=num_views,
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention4D',
                                    embed_dims=_dim_,
                                    num_points=4,
                                    num_levels=_num_levels_),
                                embed_dims=_dim_,
                            )
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_dim_*4,
                            num_fcs=2,
                            ffn_drop=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=('cross_attn', 'norm',
                                        'ffn', 'norm'))),

                dict(
                    type='BEVFormerEncoderTopKWaymo',
                    DHW=[32, 200, 100],
                    topk_ratio=0.2,
                    num_layers=1,
                    num_points_in_voxel=4,
                    num_voxel=32,
                    pc_range=point_cloud_range,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='OccFormerLayerWaymo',
                        attn_cfgs=[
                            # dict(
                            #     type='TemporalSelfAttention3D',
                            #     embed_dims=_dim_,
                            #     num_points=4,
                            #     num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                num_cams=num_views,
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention4D',
                                    embed_dims=_dim_,
                                    num_points=4,
                                    num_levels=_num_levels_),
                                embed_dims=_dim_,
                            )
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=_dim_,
                            feedforward_channels=_dim_*4,
                            num_fcs=2,
                            ffn_drop=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        operation_order=('cross_attn', 'norm',
                                        'ffn', 'norm'))),
                ]
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding3D',
            num_feats=num_feats,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            height_num_embed=bev_z_,
        ),

        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,      
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

file_client_args = dict(backend='disk')
dataset_type = 'CustomWaymoDataset_T'


train_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),
    dict(type='LoadOccGTFromFileWaymo',data_root=occ_gt_data_root,use_larger=True, crop_x=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3DWaymo', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img','voxel_semantics','mask_lidar','mask_camera'] )
]

# test_pipeline = [
#     dict(type='LoadMultiViewImageFromFiles', to_float32=True),
#     dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
#     dict(type='PhotoMetricDistortionMultiViewImage'),
#     dict(type='NormalizeMultiviewImage', **img_norm_cfg),
#     dict(type='PadMultiViewImage', size_divisor=32),
#     dict(type='CustomCollect3D', keys=[ 'img', 'voxel_semantics','mask_camera'])
# ]


test_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),
    dict(type='LoadOccGTFromFileWaymo',data_root=occ_val_gt_data_root,use_larger=True, crop_x=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3DWaymo', keys=['img','voxel_semantics','mask_lidar','mask_camera'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=load_interval,
        num_views=num_views,
        split='training',
        ann_file=data_root + 'waymo_infos_train.pkl',
        pose_file=pose_file,       
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # use_valid_flag=True,
        history_len=queue_length,
        # bev_size=(bev_h_, bev_w_),
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
            split='training',
             ann_file=data_root + 'waymo_infos_val.pkl',
            pose_file=val_pose_file,       
            num_views=num_views,
             pipeline=test_pipeline,  #bev_size=(bev_h_, bev_w_),
             test_mode=True,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
            data_root=data_root,
            split='training',
            num_views=num_views,
            ann_file=data_root + 'waymo_infos_val.pkl',
            pose_file=val_pose_file, 
            pipeline=test_pipeline, #bev_size=(bev_h_, bev_w_),
            classes=class_names, modality=input_modality,
            test_mode=False,
            history_len=queue_length,
            box_type_3d='LiDAR'
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=10000, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
find_unused_parameters=True