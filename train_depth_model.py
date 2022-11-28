import torch
import os, cv2
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from datasets.dataset import DoNeRFDataset
from models.model import PosEmbedding, DepthEstimator
from tensorboardX import SummaryWriter
from torch.optim import Adam, lr_scheduler




depth_to8b = lambda x : (255*((x - x.min()) / (x.max() - x.min()))).astype(np.uint8)




def get_onehot(depth, depth_range, n_class=10):
    """
    Inputs:
        depth: (bz, 1), input depth map
        depth_range: (bz, 2) as name 
        n_calss: one hot calsses
    Outputs:
        onehot: (bz, n)
    """
    N = depth_range.shape[0]
    steps = torch.linspace(0, 1, n_class, device=depth.device)  # (n_class)
    near, far = depth_range[:, :1], depth_range[:, 1:2]

    z_vals = near * (1-steps) + far * steps
    h_index = list(range(N))
    v_index = torch.searchsorted(z_vals, depth).squeeze(-1).tolist()
    
    onehot = torch.zeros_like(z_vals, dtype=torch.float32, device=depth.device)
    onehot[h_index, v_index] = 1
    return onehot, z_vals



def train():
    
    
    root        = '/home/baihy/datasets/DONeRF-data/'
    scene_name  = 'barbershop'
    desc        = '0'
    lr          = 0.001
    epochs      = 1000
    data_skip   = 1
    writer      = SummaryWriter(f'depth-logs-bhy/{scene_name}-{desc}')
    
    
    root_dir    = root + scene_name
    n_class     = 1
    device      = torch.device('cuda')
    depth_range = torch.FloatTensor([0.31855402439832686, 8.704841423034669])
    
    train_dataset = DoNeRFDataset(root_dir, 'train', data_skip=data_skip)
    train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=64)
    val_dataset = DoNeRFDataset(root_dir, 'val', data_skip=1)
    val_dataloader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=64)
    
    model = DepthEstimator(out_channels=n_class)
    xyz_emb = PosEmbedding(3, 10)
    dir_emb = PosEmbedding(3, 4)
    model = model.to(device)
    xyz_emb = xyz_emb.to(device)
    dir_emb = dir_emb.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # optimizer = SGD(model.parameters(), lr=0.001,  momentum=0.9, weight_decay=5e-4)
    MSE_loss = lambda x, y: torch.mean((x - y) ** 2)
    BCE_loss = torch.nn.CrossEntropyLoss()
    milestones = list(range(1, epochs, 2))
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = lr_scheduler.StepLR(optimizer, gamma=0.8)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.98)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*epochs, eta_min=5e-6)  
    
    
    for epoch in range(epochs):
        iters = len(train_dataloader)
        for it, sample in tqdm(enumerate(train_dataloader)):
            depth = sample['depth'].to(device)
            rays = sample['rays'].to(device)
            rays_o = xyz_emb(rays[:, :3])
            rays_d = dir_emb(rays[:, 3:])
            # rays_o = rays[:, 3:]
            # rays_d = rays[:, 3:]
            
            pred_depth = model(rays_o, rays_d).squeeze(-1)
            pred_depth = pred_depth * (depth_range[1] - depth_range[0]) + depth_range[0]
            target_depth = depth.squeeze(-1)
                        
            # depth_range_ = torch.stack([depth_range]*depth.shape[0], 0).to(device)
            # gt, z_val = get_onehot(depth, depth_range_, n_class=n_class)
            # print('============================================================')
            # print(gt[:5])
            # print('-------------------------------------------------------------')
            # print(pred_depth[:5])
            # print('=========================================================')
                        
            optimizer.zero_grad()
            
            loss = MSE_loss(pred_depth, target_depth)
            loss.backward()
            optimizer.step()
            
            if (epoch*iters+it) % 100 == 0 and it > 0:
                writer.add_scalar('train/loss', loss.item(), epoch*iters+it)
                writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch*iters+it)
                tqdm.write(f"[Epoch {epoch}] [Iters {epoch*iters+it}] [Training Loss: {loss.item():.6f}] [LR {optimizer.state_dict()['param_groups'][0]['lr']:.6f}]")
                
            if (epoch*iters+it) % 20000 == 0 and it > 0:
                with torch.no_grad():
                    val_depth = []
                    target_depth = []
                    for sample in tqdm(val_dataloader, desc='Validating...'):
                        depth = sample['depth'].to(device)
                        rays = sample['rays'].to(device)
                        rays_o = xyz_emb(rays[:, :3])
                        rays_d = dir_emb(rays[:, 3:])
                        
                        depth_ = model(rays_o, rays_d)                    
                        val_depth += depth_
                        target_depth += depth
                        
                    val_depth = torch.cat(val_depth, 0).reshape(800, 800).cpu()
                    target_depth = torch.cat(target_depth, 0).reshape(800, 800).cpu()
                    val_depth = val_depth * (depth_range[1] - depth_range[0]) + depth_range[0]
                    
                    loss = MSE_loss(val_depth, target_depth)
                    print(f'[Epoch {epoch}] [Validating Loss: {loss.item():.6f}]')
                    writer.add_scalar('val/loss', loss.item(), (epoch*iters+it))
                    
                    depth = torch.cat([target_depth, val_depth], 1)
                    val_depth = depth_to8b(np.array(depth.detach()))
                    cv2.imwrite(os.path.join('result', f'depth_{epoch*iters+it}.png'), val_depth)
        scheduler.step()
        

if __name__ == "__main__":
    
    train()