import torch
import numpy as np


PALETTE = np.asarray([[0.        , 0.        , 0.        ],
       [0.2745098 , 0.50980392, 0.70588235],
       [0.        , 0.        , 0.90196078],
       [0.52941176, 0.80784314, 0.92156863],
       [0.39215686, 0.58431373, 0.92941176],
       [0.85882353, 0.43921569, 0.57647059],
       [0.        , 0.        , 0.50196078],
       [0.94117647, 0.50196078, 0.50196078],
       [0.54117647, 0.16862745, 0.88627451],
       [0.43921569, 0.50196078, 0.56470588],
       [0.82352941, 0.41176471, 0.11764706],
       [255 / 255,  0,          255 / 255],
       [0.18431373, 0.30980392, 0.30980392],
       [0.7372549 , 0.56078431, 0.56078431],
       [0.8627451 , 0.07843137, 0.23529412],
       [1.        , 0.49803922, 0.31372549],
       [0,          175 / 255,  0         ],
       [1.        , 1,          1.        ],
       [0.5,        0.5,        0.5       ],
       [1.        , 0.3254902 , 0.        ],
       [1.        , 0.84313725, 0.        ],
       [1.        , 0.23921569, 0.38823529],
       [1.        , 0.54901961, 0.        ],
       [1.        , 0.38823529, 0.27843137],
       [0.        , 0.81176471, 0.74901961],
       [0.68627451, 0.        , 0.29411765],
       [0.29411765, 0.        , 0.29411765],
       [0.43921569, 0.70588235, 0.23529412],
       [0.87058824, 0.72156863, 0.52941176],
       [1.        , 0.89411765, 0.76862745],
       [0.        , 0.68627451, 0.        ],
       [1.        , 0.94117647, 0.96078431]])
def get_cv_color(i, begin=0):
    return PALETTE[begin+ i%(len(PALETTE)-begin)] * 255
def get_open3d_color(i, begin=0):
    return PALETTE[begin+ i%(len(PALETTE)-begin)]

def display_laser_on_image(img, pcl, vehicle_to_image):
    """
    pcl: ego frame
    """
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1)

    # Filter LIDAR points which are behind the camera.
    mask = np.ones_like(proj_pcl[:, 0], dtype=np.bool)
    mask = np.logical_and(mask, proj_pcl[:,2] > 0)
    # mask = proj_pcl[:,2] > 0
    #proj_pcl = proj_pcl[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]
    mask = np.logical_and(mask, proj_pcl[:, 0] > 1)
    mask = np.logical_and(mask, proj_pcl[:, 0] < img.shape[1] - 1)
    mask = np.logical_and(mask, proj_pcl[:, 1] > 1)
    mask = np.logical_and(mask, proj_pcl[:, 1] < img.shape[0] - 1)
    # Filter points which are outside the image.
    # mask = np.logical_and(
    #     np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
    #     np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    #proj_pcl = proj_pcl[mask]
    return proj_pcl, mask

def volume2points(voxel, voxel_size, point_cloud_range):
    is_numpy = False
    if isinstance(voxel, np.ndarray): 
        voxel = torch.Tensor(voxel)
        is_numpy = True
    _device = voxel.device
    voxel_size_device = torch.tensor(voxel_size).to(_device)
    point_cloud_range_device = torch.tensor(point_cloud_range).to(_device)
    xx = torch.arange(0, voxel.shape[0]).to(_device)
    yy = torch.arange(0, voxel.shape[1]).to(_device)
    zz = torch.arange(0, voxel.shape[2]).to(_device)
    grid_x, grid_y, grid_z = torch.meshgrid(xx, yy, zz, indexing='ij')
    voxel_coors = torch.stack([grid_x, grid_y, grid_z], axis=-1)
    voxel_locs = (voxel_coors + 0.5) * voxel_size_device + point_cloud_range_device[:3]

    if is_numpy:
        voxel_locs = voxel_locs.cpu().numpy()
    return voxel_locs
