import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
import imageio
import torchvision.transforms as T
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from kornia import create_meshgrid
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets.dataset import DoNeRFDataset
from torch.optim import Adam, lr_scheduler
from models.model import DepthNeRF, PosEmbedding
from models.render import render_rays


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
img_to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
depth_to8b = lambda x : (255*((x - x.min()) / (x.max() - x.min()))).astype(np.uint8)



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

def load_depth_image(depth_path):
    np_file = np.load(depth_path)
    depth = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
    depth = depth.astype(np.float32)
    max_depth = depth.max()
    depth[depth == max_depth]  = 0.
    depth[depth == 0.] = depth.max()
    return torch.FloatTensor(depth[:, None])

def load_rgb_image(rgb_path):
    img = imageio.v2.imread(rgb_path)
    img = (np.array(img) / 255.).astype(np.float32) # 4 channels (RGBA)
    H, W = img.shape[:2]
    if H != img.shape[0]:
        img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
    img = T.ToTensor()(img)               # (4, H, W)
    img = img.view(4, -1).permute(1, 0)     # (H*W, 4)
    img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
    return img


def get_val_dataloader(root_dir, split='train', idx=0):
    with open(os.path.join(root_dir, "dataset_info.json"), 'r') as f:
        info = json.load(f)
    
    resolution = info['resolution']
    camera_angle_x = info['camera_angle_x']
    W, H = resolution  
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    with open(os.path.join(root_dir, f'transforms_{split}.json'), 'r') as f:
        meta = json.load(f)
    frame = meta['frames'][0]
    pose = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
    rays_o, rays_d = pose_to_rays(H, W, focal, pose)
    
    rgb_path = os.path.join(root_dir, f"{frame['file_path']}.png")
    depth_path = rgb_path.replace('.png', '_depth.npz')
    
    rgb = load_rgb_image(rgb_path)
    depth = load_depth_image(depth_path)
    rays = torch.cat([rays_o, rays_d], 1)
    return rgb, depth, rays
    
def train():
    
    chunk       = 1024*64
    tot_iters   = 1000000
    batch_size  = 4096
    learning_rate = 8e-4
    dis         = 0.4
    n_samples   = 2
    d_loss_cof  = 0.
    skip        = 1
    desc        = '22'
    root        = '/home/baihy/datasets/DONeRF-data/'
    scene_name  = 'barbershop'
    root_dir    = root + scene_name
    device      = torch.device('cuda')
    writer      = SummaryWriter(f'logs-bhy/{scene_name}-{desc}')
    sample_scheduler = ['depth_sample', 'gauss_sample']
    
    xyz_embedding = PosEmbedding(3, 10).to(device)
    dir_embedding = PosEmbedding(3, 4).to(device)
    embeddings = [xyz_embedding, dir_embedding]
    
    train_dataset = DoNeRFDataset(root_dir, 'train', data_skip=skip)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    
    val_dataset = DoNeRFDataset(root_dir, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=64)  
    
    
    model = DepthNeRF(D=10, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4, 8])
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    iters = len(train_dataloader)
    epochs = tot_iters // iters
    milestones = list(range(3, epochs, 3))
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = lr_scheduler.StepLR(optimizer, gamma=0.8)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=5e-6)    
    
    # training
    t_start = time.time()
    for epoch in range(epochs):
        
        for it, sample in tqdm(enumerate(train_dataloader), desc=f"Training [{desc}] Epoch:{epoch}"):
            # sample: {'rays': (bz, 6), 'rgbs': (bz, 3), 'depth': (bz, 1)}
            rgbs = sample['rgbs'].to(device)
            depth = sample['depth'].to(device)
            rays = sample['rays'].to(device)
            
            B = rgbs.shape[0]
                        
            results = defaultdict(list)
            for i in range(0, B, chunk):
                rays_ = rays[i: i+chunk]
                depth_ = depth[i: i+chunk]
                render_rays_chunk = render_rays(model, embeddings, rays_, depth_, dis=dis,
                                                n_samples=n_samples, chunk=1024*32, white_back=True)
                for k, v in render_rays_chunk.items():
                    results[k] += [v]
            for k, v in results.items():
                results[k] = torch.cat(v, 0)
            
            optimizer.zero_grad()
            loss = img2mse(results['rgb'], rgbs)
            depth_loss = img2mse(results['depth'], depth.squeeze(-1))             
            loss = loss + d_loss_cof * depth_loss
            
            loss.backward()
            optimizer.step()
            # del results
            
            if (epoch*iters+it) % 500 == 0 and it > 0:  
                print(f" [{(time.time()-t_start):.2f}s] [LR {optimizer.state_dict()['param_groups'][0]['lr']:.6f}]", 
                      f"[LOSS {loss.item():.8f}] [D_Loss {depth_loss.item():.10f}] [PSNR {mse2psnr(loss.cpu()).item():.4f}]")
                writer.add_scalar('train/loss', loss.item(), epoch*iters+it)
                writer.add_scalar('train/psnr', mse2psnr(loss.cpu()).item(), epoch*iters+it)
                writer.add_scalar('train/d_loss', depth_loss.item(), epoch*iters+it)
                writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            
            if (epoch*iters+it) % 20000 == 0 and it > 0:
                with torch.no_grad():
                    results = defaultdict(list)
                    target_rgb = []
                    target_depth = []
                    for sample in tqdm(val_dataloader, desc=f"Validating [{scene_name}] Epoch:{epoch}"):
                    # sample: {'rays': (bz, 6), 'rgbs': (bz, 3), 'depth': (bz, 1)}
                        target_rgb.append(sample['rgbs'])
                        target_depth.append(sample['depth'])
                        depth = sample['depth'].to(device)
                        rays = sample['rays'].to(device)
                        
                        B = depth.shape[0]
                        
                        for i in range(0, B, chunk):
                            rays_ = rays[i: i+chunk]
                            depth_ = depth[i: i+chunk]
                            render_rays_chunk = render_rays(model, embeddings, rays_, depth_, dis=dis,
                                                            n_samples=n_samples, chunk=1024*32, white_back=True)
                            for k, v in render_rays_chunk.items():
                                results[k] += [v]
                                
                    for k, v in results.items():
                        results[k] = torch.cat(v, 0)
                    target_rgb = torch.cat(target_rgb, 0).reshape(800, 800, 3).cpu()
                    target_depth = torch.cat(target_depth, 0).reshape(800, 800).cpu()
        
                    rgbs = results['rgb'].reshape(800, 800, 3).cpu()
                    depth = results['depth'].reshape(800, 800).cpu()
                                                                
                    val_loss = img2mse(rgbs, target_rgb)
                    depth_loss = img2mse(depth, target_depth)
                    val_psnr = mse2psnr(val_loss)
                    print(f'[VAL LOSS: {val_loss.item():.6f}] [VAL PSNR: {val_psnr.item():.4f}]',
                          f'[D_LOSS {depth_loss:.10f}]\n')
                    writer.add_scalar('val/loss', val_loss.item(), epoch*iters+it)
                    writer.add_scalar('val/psnr', val_psnr.item(), epoch*iters+it)
                    writer.add_scalar('val/d_loss', depth_loss.item(), epoch*iters+it)
                    
                    img = torch.cat([target_rgb, rgbs], 1)[:, :, [2, 1, 0]]
                    depth = torch.cat([target_depth, depth], 1)
                    val_rgb = img_to8b(np.array(img.detach()))
                    val_depth = depth_to8b(np.array(depth.detach()))
                    result_dir = f'logs-bhy/{scene_name}-{desc}'
                    os.makedirs(result_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(result_dir, f'rgb_{epoch*iters+it}.png'), val_rgb)
                    cv2.imwrite(os.path.join(result_dir, f'depth_{epoch*iters+it}.png'), val_depth)
                    del target_rgb, rgbs, depth, img, val_rgb, val_depth, results
        scheduler.step()
    
                    
                    
if __name__ == '__main__':
    train()