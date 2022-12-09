from PIL import Image
import numpy as np
import open3d as o3d
import time
import cv2
import os
import glob
import json
import torch
from kornia import create_meshgrid
from scipy import interpolate


"""
def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')

    return dispariy, [(height, width, channels), scale]
"""
class point_cloud_generator():

    def __init__(self, rgb_file, depth_file, pc_file, focal_length, scalingfactor):

        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.pc_file = pc_file
        self.focal_length = focal_length
        self.scalingfactor = scalingfactor
        self.rgb = Image.open(rgb_file)
        #self.depth = Image.open(depth_file).convert('I')
        self.depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        self.width = self.rgb.size[0]
        self.height = self.rgb.size[1]

    def calculate(self):
        t1=time.time()
        depth = np.asarray(self.depth).T
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - self.width / 2) * self.Z) / self.focal_length
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - self.height / 2) * self.Z) / self.focal_length

        df=np.zeros((6,self.width*self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = -self.Y.T.reshape(-1)
        df[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        df[3] = img[:, :, 0:1].reshape(-1)
        df[4] = img[:, :, 1:2].reshape(-1)
        df[5] = img[:, :, 2:3].reshape(-1)
        self.df=df
        t2=time.time()
        print('calcualte 3d point cloud Done.',t2-t1)

    def write_ply(self):
        t1=time.time()
        float_formatter = lambda x: "%.4f" % x
        points =[]
        for i in self.df.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.pc_file, "w")
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

        t2=time.time()
        print("Write into .ply file Done.",t2-t1)

    def show_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.pc_file)
        o3d.io.draw_geometries([pcd])
        
        
def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[0])
    v = range(0, rgb.shape[1])
    
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    
    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    
    X = np.ravel(X) # 多维数组拉成一维数组
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    
    valid = Z > 0   # 判断深度不为0的像素位置
    
    X = X[valid]    # 只保留深度不为0的点
    Y = Y[valid]
    Z = Z[valid]
    
    position = np.vstack((X, Y, Z, np.ones(len(X))))    # (4, N)
    # position = np.dot(pose, position)    # (4, 4) @ (4, N) => (4, N)
    
    B = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    R = np.ravel(rgb[:, :, 2])[valid]
    
    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()
    
    return points

def depth_image_to_sample_position(depth_file, K):
    depth = np.load(depth_file)
    u = range(0, depth.shape[0])
    v = range(0, depth.shape[1])
    
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    
    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    
    sample_position = np.sqrt(X**2 + Y**2 + Z**2)
    return sample_position
    


def get_camera_origin_point(json_path):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    camera_o_points = []
    for frame in meta['frames']:
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        camera_o = pose[:3, 3]
        camera_o_points.append(camera_o)
    return np.stack(camera_o_points, axis=0)

def pose_to_rays(H, W, focal, pose):
    
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    
    pose = torch.FloatTensor(pose)
    rays_d = directions @ pose[:3, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = pose[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_samples_alone_rays(rays_o, rays_d, n_samples):
    N = rays_o.shape[0]
    near = torch.full((N, 1), 0.)
    far = torch.full((N, 1), 6.)
    z_steps = torch.linspace(0, 1, n_samples)
    z_vals = near * (1-z_steps) + far * z_steps
    z_vals = z_vals.expand(N, n_samples)
    
    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
    
    return samples
    

def write_point_cloud(ply_filename, points):
    t1 = time.time()
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" \
            % (point[0], point[1], point[2], point[3], point[4], point[5]))
    
    out_file  = open(ply_filename, "w")
    out_file.write('''ply
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
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

    t2=time.time()
    print(f"Write into {ply_filename} file Done.",t2-t1)
        

if __name__ == '__main__':
        
    depth_file = '/home/baihy/datasets/DONeRF-data/barbershop/train/00000_depth.npy'
    rgb_file = '/home/baihy/datasets/DONeRF-data/barbershop/train/00000.png'
    json_path = '/home/baihy/datasets/DONeRF-data/barbershop/transforms_train.json'
    
    W, H = 800, 800
    scale = 1
    camera_angle_x = 1.5271797180175781
    focal = float(0.5 * W / np.tan(0.5 * camera_angle_x))
    x_dist = float(np.tan(camera_angle_x / 2) * focal)
    y_dist = (H / W) * x_dist
    print(x_dist, y_dist, focal)
    
    K = np.array([[focal, 0, 0.5*W],
                  [0, focal, 0.5*H],
                  [0, 0, 1]])
    
    depth = depth_image_to_sample_position(depth_file, K)
    
    pose = [[-0.9957228899002075, 0.008912398479878902,  -0.09195677191019058, 2.4591403007507324],
            [-0.09238765388727188, -0.096054807305336, 0.9910790324211121, 7.037516117095947],
            [0.0, 0.9953359961509705, 0.09646739065647125, 1.4100117683410645],
            [0.0, 0.0, 0.0, 1.0]]
    
    rays_o, rays_d = pose_to_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(H, W, -1).reshape(-1, 3)
    rays_d = rays_d.reshape(H, W, -1).reshape(-1, 3)
    
    # # samples = get_samples_alone_rays(rays_o, rays_d, 64)

    # # samples = samples.reshape(-1, 3)
    # # samples = np.hstack((samples, np.full_like(samples, 150))).tolist()
    
    # # depth = cv2.imread(depth_file)[..., 0]
    rgb = cv2.imread(rgb_file)
    # np_file = np.load(depth_file)
    # depth = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
    # depth = depth.astype(np.float32).reshape(H, W)
    # max_depth = depth.max()
    # depth[depth == max_depth]  = 0
    # depth[depth == 0] = depth.max()
    # # depth = np.ma.masked_invalid(depth)

    # # x = np.arange(0, depth.shape[1])
    # # y = np.arange(0, depth.shape[0])

    # # xx,yy=np.meshgrid(x,y) 

    # # x1 = xx[~depth.mask]
    # # y1 = yy[~depth.mask]
    # # newarr = depth[~depth.mask].data

    # # depth = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
    
    # depth_image = (depth - depth.min()) / (depth.max() - depth.min()) * 255.
    # cv2.imwrite('depth_image-1.png', depth_image.astype(np.uint8)[::-1])
    # cv2.imwrite('rgb_image-1.png', rgb)
    # print(depth.shape, rgb.shape)
    # # print(depth.max(), depth.min(), depth.mean())
    
    rays_o = torch.FloatTensor(rays_o)
    rays_d = torch.FloatTensor(rays_d)
    depth = torch.FloatTensor(depth).flatten()[:, None]
    print(depth.shape)
    
    # # print(rays_o.shape)
    # # print(rays_d.shape)
    # # print(depth.shape)
    
    points = torch.FloatTensor(rays_o) + torch.FloatTensor(depth) * torch.FloatTensor(rays_d)
    # points = torch.cat([points, torch.Tensor(rgb.reshape(-1, 3))], -1)
    
    # print(points.shape)
    # points = depth_image_to_point_cloud(rgb, points, scale, K, pose)
    # print(len(points))
    camera_pos = get_camera_origin_point(json_path)

    camera_pos = np.hstack((camera_pos, np.full_like(camera_pos, 100))).tolist()
    points = np.hstack([np.array(points), rgb.reshape(-1, 3)]).tolist()
    points += camera_pos
    # # points += samples
    print(len(points))
    write_point_cloud('data.ply', points)
    
    
    
    
    
    
    # root_dir = '/home/baihy/Datasets/nerf_synthetic/nerf_synthetic/lego/test'
    # Image_dir = os.path.join(root_dir, 'r_25.png')
    # depth_dir = os.path.join(root_dir, 'r_25_depth_0001.png')
    # point_dir = os.path.join(root_dir, 'r_25.png')
    # # Image_name_list = []
    # # Image_name_list.extend(glob.glob(Image_glob))
    
    # # for Image_i in Image_name_list:
    #     # print(Image_i)
    # depth_ = "xx.pfm"
    # point_ = "xx.ply"
    # a = point_cloud_generator(Image_dir, 
    #                         depth_,
    #                         point_,
    #                         focal_length=50, scalingfactor=1) #50 1000 
    # a.calculate()
    # a.write_ply()
    # #a.show_point_cloud()
    # df = a.df
    # np.save(point_, df)

