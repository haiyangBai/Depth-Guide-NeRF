import numpy as np
from blend import BlenderDataset
from torch.utils.data import DataLoader
from utils import *
import cv2, os
import json
import open3d as o3d 
import numpy as np



def gen_ball_points(pc_path, radius=0.1, resolution=4, color=[0.8, 0.45, 0.24]):
    
    pcd = o3d.io.read_point_cloud(pc_path)
    points = np.array(pcd.points)

    points = np.stack([points[:,0], points[:,2], points[:,1]], 1)
    
    res = []
    for point in points:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        ball = ball.translate(point)
        res.append(ball)

    res_ply = res[0]
    for r in res[1:]:
        res_ply += r 
        
    output_name = pc_path.replace('.ply', '-ball.ply')
    res_ply.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(output_name, res_ply)

def get_depth2pc(path, pose, depth_scale=1000):
    '''从深度图变换到点云'''
    depth = cv2.imread(path)[..., 0]
    H, W = depth.shape
    camera_angle_x = 0.6911112070083618
    focal_length = 0.5 * 800 * (W/800) / np.tan(0.5*camera_angle_x)

    point_cloud = rgbd2pc(depth, focal_length, np.array(pose, dtype=np.float32), depth_scale) # 深度图 -> point cloud
    return point_cloud

def get_pose(split='test', idx=0, root='/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair/'):
    json_path = os.path.join(root, f'transforms_{split}.json')
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    pose = None
    for fname in meta['frames']:
        if fname['file_path'].split('_')[-1] == str(idx):
            pose = fname['transform_matrix']
            break
    if pose == None:
        print('No matching pose!')
        return
    return pose


def get_camera_origin_point(json_path):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    camera_o_points = []
    for frame in meta['frames']:
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        camera_o = pose[:3, 3]
        camera_o_points.append(camera_o)
    return np.stack(camera_o_points, axis=0)
    
    
if __name__ == '__main__':
    
    
    # print(depth_pc[::10000, :])
    
    # 相机位置
    names = ['Car', 'Coffee', 'Easyship', 'Scar', 'Scarf']
    splits = ['train', 'val', 'test', 'test_new']
    
    colors = {'train': [0.8, 0.45, 0.24],
              'val': [0.5, 0.5, 1],
              'test': [0, 1, 1],
              'test_new': [0.5, 1, 0.5]}
    
    radius = {'Car': 0.3,
              'Coffee': 0.04,
              'Easyship': 0.15,
              'Scar': 0.5,
              'Scarf': 0.8}
    
    split = 'train'
    exp_name = 'Car'
    # for exp_name in names:
    #     for split in splits:
    #         json_path = f'/home/baihy/competitions/JNeRF/data/Jrender_dataset/{exp_name}/transforms_{split}.json'
    #         camera_o_points = get_camera_origin_point(json_path)
    #         print(f'{exp_name} {split} data origin shape:', camera_o_points.shape)
    #         write_ply(camera_o_points, f'{exp_name}_camera_o_{split}.ply')
    #         gen_ball_points(f'{exp_name}_camera_o_{split}.ply', radius=radius[exp_name],color=colors[split])
    
    json_path = '/home/baihy/my_code/depth-NeRF/data/nerf_synthetic/chair/transforms_test.json'
    camera_o_points = get_camera_origin_point(json_path)
    print('chair_camera_o_test data origin shape:', camera_o_points.shape)
    write_ply(camera_o_points, 'chair_camera_o_test.ply')
    gen_ball_points('chair_camera_o_test.ply', radius=0.2)
    
    # # 目标点云位置
    # depth_path = '/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair/test/r_0_depth_0000.png'
    # pose = get_pose(split='test', idx=0)
    # depth_pc = get_depth2pc(depth_path, pose, depth_scale=1000)
    # print(depth_pc.shape)
    # write_ply(depth_pc, 'depth_pc.ply')
    # gen_ball_points('depth_pc.ply', radius=0.1)
    
    # # 组合点云
    # all_points = np.concatenate([depth_pc[::1000, :], camera_o_points], axis=0)
    # print(all_points.shape)
    # write_ply(all_points, 'all_points.ply')
    # gen_ball_points('all_points.ply', radius=0.1)
    
    




# if __name__ =='__main__':
#     pose = [
#                 [
#                     -1.0,
#                     0.0,
#                     0.0,
#                     0.0
#                 ],
#                 [
#                     0.0,
#                     -0.7341099977493286,
#                     0.6790306568145752,
#                     2.737260103225708
#                 ],
#                 [
#                     0.0,
#                     0.6790306568145752,
#                     0.7341099381446838,
#                     2.959291696548462
#                 ],
#                 [
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0
#                 ]
#             ]

#     depth_file = "/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair/test/r_0_depth_0000.png"
    
#     points = main(depth_file, pose)
#     write_ply(points)



