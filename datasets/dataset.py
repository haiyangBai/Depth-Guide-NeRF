import os, cv2
import json
import torch
import imageio
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from kornia import create_meshgrid
from scipy import interpolate
from torch.utils.data import DataLoader, Dataset


class DoNeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', data_skip=1, download_sample=1):
        super(DoNeRFDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.data_skip = data_skip
        self.white_back = True
        self.download_sample = download_sample
        self.read_meta()
    
    def load_depth_image(self, depth_path, K, download_sample):
        
        depth = np.load(depth_path)
        u = range(0, depth.shape[0])
        v = range(0, depth.shape[1])
        
        u, v = np.meshgrid(u, v)
        u = u.astype(float)
        v = v.astype(float)
        
        Z = depth.astype(float)
        X = (u - K[0, 2]) * Z / K[0, 0]
        Y = (v - K[1, 2]) * Z / K[1, 1]
        
        sample_position = np.sqrt(X**2 + Y**2 + Z**2)
        return torch.FloatTensor(sample_position.reshape(-1, 1))
    
    def load_rgb_image(self, rgb_path, download_sample):
        H, W = self.resolution
        img = imageio.v2.imread(rgb_path)
        img = (np.array(img) / 255.).astype(np.float32) # 4 channels (RGBA)
        if self.H != img.shape[0]:
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (H//download_sample, W, download_sample), interpolation=cv2.INTER_AREA)
        img = self.transform(img)               # (4, H, W)
        img = img.view(4, -1).permute(1, 0)     # (H*W, 4)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
        return img
           
    def pose_to_rays(self, H, W, focal, pose):
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

    def read_meta(self):
        with open(os.path.join(self.root_dir, "dataset_info.json"), 'r') as f:
            info = json.load(f)
        camera_angle_x = info['camera_angle_x']
        depth_ignore = info['depth_ignore']
        depth_range_warped_log = info['depth_range_warped_log']
        depth_range_warped_lin = info['depth_range_warped_lin']
        depth_range = info['depth_range']
        self.resolution = info['resolution']
        self.W, self.H = self.resolution
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        
        K = np.array([
            [self.focal, 0, 0.5*self.W],
            [0, self.focal, 0.5*self.H],
            [0, 0, 1]
        ])
        
        with open(os.path.join(self.root_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)

        # get rays, depth and rgb data
        if self.split == 'train':
            self.img_paths = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_depth = []
            
            for frame in tqdm(self.meta['frames'][::self.data_skip], desc='Loading Train Dataset'):
                # get poses/rays
                pose = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
                rays_o, rays_d = self.pose_to_rays(self.H//self.download_sample, self.W//self.download_sample, self.focal, pose)
                self.all_rays.append(torch.cat([rays_o, rays_d], 1))

                # get rgbs and depth
                rgb_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                depth_path = rgb_path.replace('.png', '_depth.npy')
                self.img_paths.append(rgb_path)
                
                rgb = self.load_rgb_image(rgb_path, self.download_sample)
                depth = self.load_depth_image(depth_path, K, self.download_sample)
                self.all_rgbs.append(rgb)
                self.all_depth.append(depth)

            self.all_rays = torch.cat(self.all_rays, 0)     # (len(img_paths)*h*w, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)     # (len(img_paths)*h*w, 3)
            self.all_depth = torch.cat(self.all_depth, 0)   # (len(img_paths)*h*w, 1)

        elif self.split == 'val':
            frame = self.meta['frames'][0]
            pose = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            rays_o, rays_d = self.pose_to_rays(self.H, self.W, self.focal, pose)
        
            rgb_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            depth_path = rgb_path.replace('.png', '_depth.npy')
            
            self.rgb = self.load_rgb_image(rgb_path, self.download_sample)        # (H*W, 3)
            self.depth = self.load_depth_image(depth_path, K, self.download_sample)  # (H*W, 1)
            self.rays = torch.cat([rays_o , rays_d], 1)     # (H*W, 6)
            
        elif self.split == 'test':
            pass

    def transform(self, img):
        return T.ToTensor()(img)

    def __len__(self):
        if self.split == 'train':
            return self.all_rays.shape[0]
        elif self.split == 'val':
            return self.W*self.H//(self.download_sample**2)
        elif self.split == 'test':
            return 0

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx], 
                      'depth': self.all_depth[idx]}
        
        elif self.split == 'val':
            sample = {'rays': self.rays[idx],     # (H*W, 6)
                      'rgbs': self.rgb[idx],      # (H*W, 3)
                      'depth': self.depth[idx]}   # (H*W,)
        
        elif self.split == 'test':
            sample = {}
        
        return sample


if __name__ == '__main__':
    import sys
    sys.path.append('/home/baihy/datasets/DONeRF-data/')
    from torch.utils.data import DataLoader

    root_dir = '/home/baihy/datasets/DONeRF-data/barbershop'
    dataset = DoNeRFDataset(root_dir, 'train', data_skip=10, download_sample=1)
    from torch.utils.data import DataLoader
    
    data = DataLoader(dataset, batch_size=1, num_workers=64)
    print(len(data))
    for i, sample in enumerate(data):
        rgb = sample['rgbs']
        rays = sample['rays']
        depth = sample['depth']
        print(i, rgb.shape, rays.shape, depth.shape)
        break
    
   
    
        
    
