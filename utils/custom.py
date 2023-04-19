from typing import List, Tuple
import numpy as np
import pickle as pkl
import torch
# from mmdet3d.core.bbox import LiDARInstance3DBoxes

import numpy as np
import torch
import open3d as o3d # not known why, must import torch before open3d, othewise GPU matmul fail
from matplotlib.cm import get_cmap
cm = get_cmap("afmhot")

def to_stride(shape: np.ndarray):
    stride = np.ones_like(shape)
    stride[:shape.shape[0] - 1] = np.cumprod(shape[::-1])[::-1][1:]
    return stride

def _indice_to_scalar(indices: torch.Tensor, shape: List[int]):
    assert indices.shape[1] == len(shape)
    stride = to_stride(np.array(shape, dtype=np.int64))
    scalar_inds = indices[:, -1].clone()
    for i in range(len(shape) - 1):
        scalar_inds += stride[i] * indices[:, i]
    return scalar_inds.contiguous()

def sparse2dense(
    points: torch.Tensor, # [N, 3+]
    points_labels: torch.Tensor, # [N, ]
    bbox, # [M, 7+],
    bbox_labels: torch.Tensor, # [M, ]
    voxel_size = (0.1, 0.1, 0.25), # x/y/z order
    point_cloud_range = [-20, -20, -2, 20, 20, 6], # x/y/z order
    sparse_shape=[32, 400, 400], # z/y/x order，
    DEBUG=False,
):
    voxel_size_numpy = np.asarray(voxel_size)
    point_cloud_range_numpy = np.asarray(point_cloud_range)
    sparse_shape_numpy = np.asarray(sparse_shape)
    assert np.alltrue(voxel_size_numpy * sparse_shape_numpy[::-1] == point_cloud_range_numpy[3:6] - point_cloud_range_numpy[:3])
    _device = points.device
    voxel_size_device = torch.tensor(voxel_size).to(_device)
    point_cloud_range_device = torch.tensor(point_cloud_range).to(_device)
    
    # cubic = torch.full(sparse_shape, fill_value=-1, dtype=torch.long, device=_device)
    # cubic_squeeze = cubic.reshape(-1)
    inrange = (points[:,0] > point_cloud_range[0]) & (points[:,1] > point_cloud_range[1]) & (points[:,2] > point_cloud_range[2]) & \
              (points[:,0] < point_cloud_range[3]) & (points[:,1] < point_cloud_range[4]) & (points[:,2] < point_cloud_range[5])
    points = points[inrange]
    points_labels = points_labels[inrange]
    
    points = points - point_cloud_range_device[:3]
    pcds_voxel = torch.div(points, voxel_size_device, rounding_mode='floor').long()
    pcds_voxel = torch.flip(pcds_voxel, dims=[1]) # must in z/y/x ATTENTSION
    scalar = _indice_to_scalar(pcds_voxel, sparse_shape) # TODO
    
    voxel_state = torch.full(sparse_shape, fill_value=0, dtype=torch.int, device=_device)
    occ_squeeze = voxel_state.reshape(-1)
    occ_squeeze[scalar] = 1

    semantic_label = torch.full(sparse_shape, fill_value=0, dtype=points_labels.dtype, device=_device)
    semantic_label_sequeeze = semantic_label.reshape(-1)
    semantic_label_sequeeze[scalar] = points_labels
    
    # voxel_state: Z/Y/X order, 1 for voxel_state
    # semantic_label Z/Y/X order,  
    return voxel_state, semantic_label

if __name__ == '__main__':
    file = 'data/fakedata/10.pkl'
    with open(file, 'rb') as f:
        data = pkl.load(f)
    points = torch.tensor(data['points']).cuda()
    labels = torch.tensor(data['labels']).cuda().long()
    
    voxel_state, semantic_label = sparse2dense(points, labels, None, None)
    voxel_state = voxel_state.cpu().numpy()
    semantic_label = semantic_label.cpu().numpy()
    
    # vis check
    _shape = voxel_state.shape
    _scalar = np.arange(0, _shape[0]*_shape[1]*_shape[2])
    fake_pcds = np.zeros((_shape[0]*_shape[1]*_shape[2], 3))
    fake_pcds[:, 0] = _scalar // _shape[1] // _shape[2]
    fake_pcds[:, 1] = _scalar // _shape[1] % _shape[2]
    fake_pcds[:, 2] = _scalar % _shape[1] % _shape[2]
    
    mask = voxel_state.reshape(-1) == 1
    fake_pcds = fake_pcds[mask]
    semantic_label = semantic_label.reshape(-1)[mask]

    open3d_color_list = [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 0),
    ]
    colors = np.zeros_like(fake_pcds)
    for idx in np.unique(semantic_label):
        colors[semantic_label==idx] = open3d_color_list[idx%len(open3d_color_list)]
        
    pcds = o3d.open3d.geometry.PointCloud()
    pcds.points = o3d.open3d.utility.Vector3dVector(fake_pcds)
    pcds.colors = o3d.open3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcds], window_name='Open3D downSample',point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False, )
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcds, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])