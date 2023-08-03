import numpy as np
import cv2
import torch
import os
import pickle as pkl
import open3d as o3d

import matplotlib.pyplot as plt
from glob import glob
from typing import Tuple, List, Dict, Iterable
from PIL import Image

# from mmdet3d.core.visualizer.open3d_vis import Visualizer

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

def gray2rgb(gray):
    rgb = cv2.applyColorMap(src=gray.astype(np.uint8), colormap=cv2.COLORMAP_JET)
    rgb = np.squeeze(rgb, axis=1)
    return rgb

cv2_color_list = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]
def get_cv_color(i, begin=0):
    return PALETTE[begin+ i%(len(PALETTE)-begin)] * 255
def get_open3d_color(i, begin=0):
    return PALETTE[begin+ i%(len(PALETTE)-begin)]
def vis_mesh2image(triangles_vertices, img=None, color=(255, 0, 0), width=1920, height=886):
    '''
    triangles_vertices: [N, 3, 2]
    '''
    # vis check
    # Width and height of the black window 
    # Create a black window of 400 x 300
    if img is None:
        img = np.zeros((height, width, 3), np.uint8)
    # Three vertices(tuples) of the triangle 
    
    # Drawing the triangle with the help of lines
    #  on the black window With given points 
    # cv2.line is the inbuilt function in opencv library
    uvs = np.copy(triangles_vertices).astype(int)
    for i in range(uvs.shape[0]):
        # plot this triangle
        p1 = (uvs[i][0][0], uvs[i][0][1])
        p2 = (uvs[i][1][0], uvs[i][1][1])
        p3 = (uvs[i][2][0], uvs[i][2][1])
        cv2.line(img, p1, p2,color, 1)
        cv2.line(img, p2, p3,color, 1)
        cv2.line(img, p1, p3,color, 1)
    return img        
    # image is the title of the window
    cv2.imshow("image", img)
    cv2.waitKey(0)
    
def vis_point2image(uvs, img=None, labels=None, color=(255, 0, 0), width=1920, height=886):
    '''
    triangles_vertices: [N, 3, 2]
    '''
    # vis check
    # Width and height of the black window 
    # Create a black window of 400 x 300
    if img is None:
        img = np.zeros((height, width, 3), np.uint8)
    # Three vertices(tuples) of the triangle 
    
    # Drawing the triangle with the help of lines
    #  on the black window With given points 
    # cv2.line is the inbuilt function in opencv library
    # uvs = np.copy(img).astype(int)
    if labels is None:
        img = cv2.circle(img, uvs, radius=0, color=get_cv_color(0), thickness=-1)
    else:
        ids = np.unique(labels)
        for id in ids:
            centers = uvs[labels==id].transpose(1,0).astype(int)
            img = cv2.circle(img, centers, radius=0, color=get_cv_color(id), thickness=-1)
    return img        

    
def vis_points_all(points, points_label=None, vis_voxel=True, voxel_size=0.1):
    pcds = o3d.open3d.geometry.PointCloud()
    colors = np.zeros((points.shape[0], 3))
    if points_label is not None:
        for idx in np.unique(points_label):
            colors[points_label==idx] = get_open3d_color(idx)
    pcds.points = o3d.open3d.utility.Vector3dVector(points)
    pcds.colors = o3d.open3d.utility.Vector3dVector(colors)
    need_vis = [pcds]
    if vis_voxel:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcds, voxel_size=voxel_size)
        need_vis.append(voxel_grid)
    vis = o3d.visualization.draw_geometries(need_vis, window_name='Open3D downSample', point_show_normal=False,
                                    mesh_show_wireframe=True,
                                    mesh_show_back_face=True, )
    del vis

def vis_mesh_all(mesh_dir, load_stuff=False):
    files = glob(os.path.join(mesh_dir, '*.ply'))
    mesh_list = []
    for file in files:
        if not load_stuff and 'stuff' in file: continue
        mesh = o3d.io.read_triangle_mesh(file, enable_post_processing=False, print_progress=False)
        mesh_list.append(mesh)
    o3d.visualization.draw_geometries(mesh_list, window_name='Open3D downSample', point_show_normal=True,
                                      mesh_show_wireframe=True,
                                      mesh_show_back_face=True, )


def vis_volume(occ_state, occ_label):
    H, W, Z = occ_state.shape[0], occ_state.shape[1], occ_state.shape[2]
    xx = torch.arange(0, occ_state.shape[0])
    yy = torch.arange(0, occ_state.shape[1])
    zz = torch.arange(0, occ_state.shape[2])
    grid_x, grid_y, grid_z = torch.meshgrid(xx, yy, zz, indexing='ij')
    coors = torch.stack([grid_x, grid_y, grid_z], axis=-1) # [H,W,Z,3]
    
    # get voxel 8 coors
    coors = coors + 0.5
    offsets = [-0.5, -0.5, -0.5] # [8, 3]
    edges = [[0,1]] # [12, 2]
    coors = coors + offsets # [H,W,Z, 8, 3]
    
    # get 12 lines
    coors = coors.reshape(H*W*Z, 8, 3)
    bases_ = torch.arange(0, H*W*Z)
    edges = edges.reshape(1, 12, 2).expand((H*W*Z, 1, 1))
    edges = edges + bases_
    
    points = coors.reshape(H*W*Z*8, 3).cpu().numpy() # [H*W*Z*8, 3]
    lines = edges.reshape(H*W*Z*12, 2).cpu().numpy() # [H*W*Z*12, 2] to int()?
    line_sets = o3d.geometry.LineSet()
    line_sets.points = o3d.open3d.utility.Vector3dVector(points)
    line_sets.lines = o3d.open3d.utility.Vector2dVector(lines)
    line_sets.colors = o3d.open3d.utility.Vector3dVector(colors)
    
