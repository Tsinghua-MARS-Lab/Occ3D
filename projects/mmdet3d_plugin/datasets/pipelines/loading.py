import operator
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os
from PIL import Image



@PIPELINES.register_module()
class MyLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged'):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            padded = np.zeros((self.img_scale[0],self.img_scale[1],3))
            padded[0:img.shape[0], 0:img.shape[1], :] = img
            img = padded
        return img

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # import time
        # _=time.time()
        filename = results['img_filename']
        filename = filename# DEBUG_TMP
        # img = [mmcv.imread(name, self.color_type) for name in filename]
        img = [np.asarray(Image.open(name))[...,::-1] for name in filename]
        # breakpoint()
        results['ori_shape'] = [img_i.shape for img_i in img]
        if self.img_scale is not None:
            img = [self.pad(img_i) for img_i in img]
        results['img_shape'] = [img_i.shape for img_i in img]
        results['pad_shape'] = [tuple(map(lambda x, y: y - x, s, t)) for s,t in zip(results['ori_shape'], results['img_shape'])]
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        # Set initial values for default meta_keys
        # results['scale_factor'] = [1.0, 1.0]####
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        # for i in range(img.shape[-1]):
        #     mmcv.imwrite(img[..., i], 'debug_image/new_loaded_{}.png'.format(i))
        # open('debug_image/new_load_results.txt','w').write(str(results)+'\n')
        # print('read img cost: {} ms'.format(time.time()-_))
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module()
class LoadOccGTFromFileWaymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            use_larger=True,
            crop_x=False,
        ):
        self.use_larger=use_larger
        self.data_root = data_root
        self.crop_x = crop_x

    def __call__(self, results):
        pts_filename = results['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path=os.path.join(self.data_root, seq_name,  '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov']
        mask_lidar = occ_labels['origin_voxel_state']
        mask_camera = occ_labels['final_voxel_state']
        if self.crop_x:
            w, h, d = semantics.shape
            semantics = semantics[w//2:, :, :]
            mask_infov = mask_infov[w//2:, :, :]
            mask_lidar = mask_lidar[w//2:, :, :]
            mask_camera = mask_camera[w//2:, :, :]

        results['voxel_semantics'] = semantics
        results['mask_infov'] = mask_infov
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(self.data_root,occ_gt_path)

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)