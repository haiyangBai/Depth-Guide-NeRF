import torch
import numpy as np
from kornia import create_meshgrid


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

def rgbd2rgb_pc(rgbs, depth_map, focal_length, pose=None, depth_scale=1000):
    """Transfer RGB-D image to point cloud.

    Args:
        rgbs (Tensor or ndarray): (H, W, 3)
        depths (Tensor or ndarray): (H, W)
        camera_intrinsics (list, optional): fx, fy, cx, cy.
        pose (Tensor or ndarray): (4, 4)
    """
    H, W = rgbs.shape[:2]
    fx = focal_length
    fy = focal_length
    cx = W / 2
    cy = H / 2
    
    depth = np.asarray(depth_map, dtype=np.uint16).T

    Z = depth / depth_scale
    X = np.zeros((W, H))
    Y = np.zeros((W, H))
    
    for i in range(W):
        X[i, :] = np.full(X.shape[1], i)
    X = ((X - cx / 2) * Z) / fx
    
    for i in range(H):
        Y[:, i] = np.full(Y.shape[0], i)
    Y = ((Y - cy / 2) * Z) / fy

    data_ply = np.zeros((6, W*H))
    data_ply[0] = X.T.reshape(-1)
    data_ply[1] = -Y.T.reshape(-1)
    data_ply[2] = -Z.T.reshape(-1)
    img = np.array(rgbs, dtype=np.uint8)
    data_ply[3] = img[:, :, 0:1].reshape(-1)
    data_ply[4] = img[:, :, 1:2].reshape(-1)
    data_ply[5] = img[:, :, 2:3].reshape(-1)
    
    return data_ply

def rgbd2pc(depth_map, focal_length, pose=None, depth_scale=1000):
    """Transfer RGB-D image to point cloud.

    Args:
        rgbs (Tensor or ndarray): (H, W, 3)
        depths (Tensor or ndarray): (H, W)
        camera_intrinsics (list, optional): fx, fy, cx, cy.
        pose (Tensor or ndarray): (4, 4)
    """
    H, W = depth_map.shape[:2]
    fx = focal_length
    fy = focal_length
    cx = W / 2
    cy = H / 2
    
    depth = np.asarray(depth_map, dtype=np.uint16).T

    Z = depth / depth_scale
    X = np.zeros((W, H))
    Y = np.zeros((W, H))
    
    for i in range(W):
        X[i, :] = np.full(X.shape[1], i)
    X = ((X - cx / 2) * Z) / fx
    
    for i in range(H):
        Y[:, i] = np.full(Y.shape[0], i)
    Y = ((Y - cy / 2) * Z) / fy

    data_ply = np.zeros((3, W*H))
    data_ply[0] = X.T.reshape(-1)
    data_ply[1] = -Y.T.reshape(-1)
    data_ply[2] = -Z.T.reshape(-1)
    data_ply = data_ply.transpose(1, 0)
    data_ply = data_ply @ pose[:3, :3].T
    
    return data_ply

def write_ply(point_cloud, save_name='point_clond.ply'):
    if point_cloud.shape[1] == 3:
        point_cloud = np.concatenate([point_cloud, np.zeros(point_cloud.shape)], axis=1)
        point_cloud[:, 5] = np.full_like(point_cloud[:,0], 255, dtype=np.uint8)
    float_formatter = lambda x: "%.4f" % x
    points = []
    for i in point_cloud:
        points.append("{} {} {} {} {} {} 0\n".format
                        (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                        int(i[3]), int(i[4]), int(i[5])))

    file = open(save_name, "w")
    file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                property uchar alpha
                end_header
                %s
    ''' % (len(points), "".join(points)))
    file.close()


if __name__ == '__main__':
    import cv2
    
    
    H, W = 800, 800
    camera_angle_x = 0.6911112070083618
    focal_length = 0.5 * 800 * (W/800) / np.tan(0.5*camera_angle_x)
    
    rgb_file = "/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair/test/r_0.png"
    depth_file = "/home/baihy/datasets/nerf_synthetic/nerf_synthetic/chair/test/r_0_depth_0000.png"

    rgb = cv2.imread(rgb_file)
    depth = cv2.imread(depth_file, -1)[..., 0]
    
    write_ply(rgb, depth, focal_length, save_name='pc.ply')
    
    