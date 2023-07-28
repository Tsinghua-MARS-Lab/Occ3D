import os
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
from tqdm import tqdm
import cv2
from PIL import Image
from glob import glob

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1

color_map = {
    0: (0,0,0),
    1: (255,255,255),
    2: (255,0,0),
    3: (0,255,0),
    4: (0,0,255),
    5: (255,255,0),
    6: (0,255,255),
    7: (255,0,255),
    8: (192,192,192),
    9: (128,128,128),
    10: (128,0,0),
    11: (128,128,0),
    12: (0,128,0),
    13: (128,0,128),
    14: (0,128,128),
    15: (0,0,128),
 }
colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [128, 128, 128, 255], # 16 for vis
], dtype=np.float32)

LINE_SEGMENTS = [
    [4, 0], [3, 7], [5, 1], [6, 2],  # lines along x-axis
    [5, 4], [5, 6], [6, 7], [7, 4],  # lines along x-axis
    [0, 1], [1, 2], [2, 3], [3, 0]]  # lines along y-axis

def _expand_dim(array):
    return np.concatenate((array, np.ones_like(array)[:, :1]), axis=1)

# def colormap_to_colors(colormap: Dict[str, Iterable[int]]) -> np.ndarray:
#     """
#     Create an array of RGB values from a colormap. Note that the RGB values are normalized
#     between 0 and 1, not 0 and 255.
#     :param colormap: A dictionary containing the mapping from class names to RGB values.
#     :param name2idx: A dictionary containing the mapping form class names to class index.
#     :return: An array of colors.
#     """
#     colors = []
#     for i, (k, v) in enumerate(colormap.items()):
#         # Ensure that the indices from the colormap is same as the class indices.
#         colors.append(v)

#     colors = np.array(colors) / 255  # Normalize RGB values to be between 0 and 1 for each channel.

#     return colors

def voxel2points(voxel, occ_show, voxelSize):
    occIdx = torch.where(occ_show)
    # points = torch.concatenate((np.expand_dims(occIdx[0], axis=1) * voxelSize[0], \
    #                          np.expand_dims(occIdx[1], axis=1) * voxelSize[1], \
    #                          np.expand_dims(occIdx[2], axis=1) * voxelSize[2]), axis=1)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0], \
                        occIdx[1][:, None] * voxelSize[1], \
                        occIdx[2][:, None] * voxelSize[2]), dim=1)
    return points, voxel[occIdx], occIdx

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False, bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], visible=True) -> None:
    """
    :param points:
    :param colors: false 不显示点云颜色
    :param points_colors:
    :param bbox3d: voxel边界， Nx7 (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(visible=visible)
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    #opt.background_color = np.asarray([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        # line_sets.colors = o3d.open3d.utility.Vector3dVector(colors)
        # linesets = _draw_bboxes(bbox3d, vis)

    vis.add_geometry(mesh_frame)
    vis.add_geometry(line_sets)
    # vis.run()
    return vis

def main(occ_state, occ_show, voxel_size, vis=None, offset=[0,0,0]):
    # occ_state, voxel_size = data['occ_state'].cpu(), data['voxel_size']
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]
    bboxes = voxel_profile(pcd, voxel_size)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    #bboxes_corners = torch.cat([my_compute_box_3d(box[0:3], box[3:6], box[6:7])[None, ...] for box in bboxes], dim=0)
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]
    vis = show_point_cloud(points=pcd.numpy(), colors=True, points_colors=pcds_colors, voxelize=True, bbox3d=bboxes.numpy(),
                     bbox_corners=bboxes_corners.numpy(), linesets=edges.numpy(), vis=vis, offset=offset)
    return vis

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    trajectory = o3d.io.read_pinhole_camera_trajectory(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(
            trajectory.intrinsic, trajectory.extrinsic[0])
    vis.run()
    vis.destroy_window()

def video(voxel_path, save_path, vis_voxel01=True, interval=5, vis_3d=False):
    ROAD_LABEL_START = 13
    ROAD_LABEL_STOP = 14
    FILL_ROAD = True
    FREE_LABEL = 23
    fps = 10 // interval

    # you can use Ctrl+c and Ctrl+v to get vis_param
    vis_param  = {
        "class_name" : "ViewTrajectory",
        "interval" : 29,
        "is_loop" : False,
        "trajectory" : 
        [
            {
                "boundingbox_max" : [ 80.000015258789062, 80.000015258789062, 7.8000006675720215 ],
                "boundingbox_min" : [ -80.0, -80.0, -5.0 ],
                "field_of_view" : 60.0,
                "front" : [ -0.63153029317382392, -0.0044086852255532157, 0.77533866942025165 ],
                "lookat" : [ 9.0382292077897386, 0.44317978063165703, 7.0656591627410501 ],
                "up" : [ 0.77529242561670686, 0.0087221808444446977, 0.63154222213774414 ],
                "zoom" : 0.080000000000000002
            }
        ],
        "version_major" : 1,
        "version_minor" : 0
    }
    voxelsize = [0.1, 0.1, 0.2] if vis_voxel01 else [0.4, 0.4, 0.4] 
    point_cloud_range = [-80, -80, -5, 80, 80, 7.8] if vis_voxel01 else [-40, -40, -1, 40, 40, 5.4]

    color = colormap_to_colors / 255
    if not os.path.exists(save_path): os.makedirs(save_path)

    front_vec = np.array(vis_param['trajectory'][0]['front'])[..., None]
    up_vec = np.array(vis_param['trajectory'][0]['up'])[..., None]
    zoom = vis_param['trajectory'][0]['zoom']
    lookat = vis_param['trajectory'][0]['lookat']

    npz_file_list = sorted(glob(os.path.join(voxel_path, "*npz")))
    large_voxel_npz_file_list = sorted(glob(os.path.join(voxel_path, "*_04.npz")))
    if not vis_voxel01:
        voxel_files = large_voxel_npz_file_list
    else:
        voxel_files = sorted(list(set(npz_file_list).difference(set(large_voxel_npz_file_list))))
    for frame_idx, voxel_file in enumerate(tqdm(voxel_files[::interval])):
        voxel = np.load(voxel_file)['voxel_label']
        if FILL_ROAD:
            # fill road for vis
            road=(voxel==ROAD_LABEL_START)
            # road_level=torch.argmax(torch.bincount(torch.nonzero(road)[:, 2]))
            counts = np.bincount(np.nonzero(road)[2])
            road_level = max(np.argmax(counts) - 5, 0)
            voxel[:,:, road_level] = 16 # gray color
        
        voxel = torch.from_numpy(voxel)
        occ_show = voxel != FREE_LABEL
        points, labels, occIdx = voxel2points(voxel, occ_show, voxelsize)
        points[:, 0] += point_cloud_range[0]
        points[:, 1] += point_cloud_range[1]
        points[:, 2] += point_cloud_range[2]

        pcd_colors = color[labels.numpy().astype(int)]
        bboxes = voxel_profile(torch.tensor(points), voxelsize)
        bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
        bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],
                            [3, 7]])  # lines along y-axis
        edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
        edges = edges + bases_[:, None, None]
        vis = show_point_cloud(points=points.numpy(), colors=True, points_colors=pcd_colors, voxelize=True, bbox_corners=bboxes_corners.numpy(), linesets=edges.numpy(), visible=vis_3d)
        
        ego_pcd = o3d.geometry.PointCloud()
        ego_points = generate_the_ego_car()
        ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
        vis.add_geometry(ego_pcd)

        # view control
        view_control = vis.get_view_control()
        view_control.set_zoom(zoom)
        view_control.set_up(up_vec)
        view_control.set_front(front_vec)
        view_control.set_lookat(lookat)
        vis.poll_events()
        vis.update_renderer()
        vis.get_render_option().point_size = 8

        if vis_3d:
            vis.run()
        # save to image
        img = vis.capture_screen_float_buffer(True)
        img = np.array(img)
        img = (img * 255).astype(np.uint8)

        img = increase_brightness(img, value=20)
        im = Image.fromarray(img)
        out_file = os.path.join(save_path, f'{str(frame_idx).zfill(3)}.jpg')
        im.save(out_file)
        print("save image to ", out_file)
        im.close()
        del im

        vis.clear_geometries()
        vis.destroy_window()
        del view_control
        del vis

    
    # CALL THIS COMAND TO GENERATE VIDEO
    command = "ffmpeg -r {} -i {}/%03d.jpg -c:v libx264 -vf \"fps={},format=yuv420p\"  -preset medium -crf 30 {}/video.mp4".format(fps, save_path, fps, save_path)
    print(os.popen(command).read())

if __name__ == '__main__':
    data_dir = "/home/user/tmp/voxel/000/"
    
    NOT_OBSERVED = -1
    FREE = 0
    OCCUPIED = 1
    FREE_LABEL = 23
    MAX_POINT_NUM = 10
    ROAD_LABEL_START_BEFORE_REMAP = 24
    ROAD_LABEL_STOP_BEFORE_REMAP = 27
    ROAD_LABEL_START = 13
    ROAD_LABEL_STOP = 14
    BINARY_OBSERVED = 1
    BINARY_NOT_OBSERVED = 0
    STUFF_START = 9  # 0-10 thing 11-17 stuff

    VOXEL_SIZE=[0.1, 0.1, 0.2]
    POINT_CLOUD_RANGE=[-80, -80, -5, 80, 80, 7.8]
    SPTIAL_SHAPE=[1600, 1600, 64]
    TGT_VOXEL_SIZE=[0.4, 0.4, 0.4]
    TGT_POINT_CLOUD_RANGE=[-40, -40, -1, 40, 40, 5.4]
    VIS = False
    FILL_ROAD = False

    voxel_size = VOXEL_SIZE
    point_cloud_range = POINT_CLOUD_RANGE
    for idx in range(100):
        file = os.path.join(data_dir, f'{str(idx).zfill(3)}.npz')
        data = np.load(file)
        voxel_label = data['voxel_label']
        lidar_mask = data['origin_voxel_state']
        camera_mask = data['final_voxel_state']
        infov = data['infov']
        ego2global = data['ego2global']

        if FILL_ROAD:
            # fill road for vis
            road=(voxel_label==ROAD_LABEL_START)
            # road_level=torch.argmax(torch.bincount(torch.nonzero(road)[:, 2]))
            road_level = (np.nonzero(road)[2]).min()
            voxel_label[:,:, road_level] = 16 # gray color

        ignore_labels = [FREE_LABEL]
        mask = np.zeros_like(voxel_label, dtype=np.bool)
        for ignore_label in ignore_labels:
            mask = np.logical_or(voxel_label == ignore_label, mask)
        mask = np.logical_not(mask)

        voxel_state = lidar_mask
        voxel_label_vis = voxel_label
        voxel_show = voxel_label != FREE_LABEL
        vis = main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=None,
                            offset=[voxel_state.shape[0] * voxel_size[0] * 1.2 * 0, 0, 0])

        voxel_label_vis = voxel_label
        voxel_show = np.logical_and(voxel_label != FREE_LABEL, lidar_mask == BINARY_OBSERVED)
        vis = main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[voxel_state.shape[0] * voxel_size[0] * 1.2 * 1, 0, 0])

        voxel_label_vis = voxel_label
        voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask == BINARY_OBSERVED)
        vis = main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[voxel_state.shape[0] * voxel_size[0] * 1.2 * 2, 0, 0])

        voxel_label_vis = voxel_label
        voxel_show = np.logical_and(voxel_label != FREE_LABEL, infov == True)
        vis = main(torch.from_numpy(voxel_label_vis), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                            offset=[voxel_state.shape[0] * voxel_size[0] * 1.2 * 3, 0, 0])
        
        ego_point = generate_the_ego_car()
        ego_point[:, 0] += point_cloud_range[3]
        ego_point[:, 1] += point_cloud_range[4]
        ego_point[:, 2] += point_cloud_range[5]
        ego_pcd = o3d.geometry.PointCloud()
        ego_pcd.points = o3d.utility.Vector3dVector(ego_point)
        vis.add_geometry(ego_pcd)

        vis.run()
        vis.poll_events()
        vis.update_renderer()
        # vis.capture_screen_image(f'output/ray.jpg')

        del vis
