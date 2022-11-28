import cv2, os
import torch
import numpy as np

from opt import get_opts
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
from utils.utils import get_learning_rate, get_optimizer, get_scheduler
from datasets.dataset import DoNeRFDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from models.render import render_rays
from models.model import DepthNeRF, PosEmbedding
from utils.losses import MSELoss, MAELoss
from pytorch_lightning              import LightningModule, Trainer
from pytorch_lightning.callbacks    import ModelCheckpoint
from pytorch_lightning.loggers      import TensorBoardLogger


class DGNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(DGNeRFSystem, self).__init__()
        self.hy_params = hparams
        self.batch = 0
        self.save_hyperparameters(hparams)
        
        self.warmup_steps = 256
        
        self.loss = MSELoss(self.hy_params.coff_depth_loss)
        self.embeddings = [PosEmbedding(3, 10), PosEmbedding(3, 4)]
        self.model = DepthNeRF()
        
    def forward(self, rays, depth):
        B = rays.shape[0]
        chunk = self.hy_params.chunk             
        results = defaultdict(list)
        for i in range(0, B, chunk):
            rays_ = rays[i: i+chunk]
            depth_ = depth[i: i+chunk]
            render_rays_chunk = render_rays(self.model, self.embeddings, 
                                            rays_, depth_, dis=self.hy_params.dis,
                                            n_samples=self.hy_params.n_samples, 
                                            chunk=chunk , white_back=True)
            for k, v in render_rays_chunk.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results
            
    def training_step(self, batch, batch_idx):
        rgbs = batch['rgbs']
        depth = batch['depth']
        rays = batch['rays']
        
        result = self(rays, depth)
        rgb_loss = torch.mean((result['rgb'] - rgbs) ** 2)
        depth_loss = torch.mean((result['depth'] - depth.squeeze(-1)) ** 2)
        loss = rgb_loss + self.hy_params.coff_depth_loss * depth_loss
        psnr = -10. * torch.log(loss).to(rays.device) / \
                torch.log(torch.Tensor([10.])).to(rays.device)
        
        lr = self.get_learning_rate(self.optimizer)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr, prog_bar=True)
        self.log('train/lr', lr, prog_bar=True)
        
        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()
        return loss
        
    def configure_optimizers(self):
        eps = 1e-8
        if self.hy_params.optimizer == 'sgd':
            self.optimizer = SGD(self.parameters(), lr=self.hy_params.lr, 
                                    momentum=self.hy_params.momentum)
        elif self.hy_params.optimizer == 'adam':
            self.optimizer = Adam(self.parameters(), lr=self.hy_params.lr, eps=eps)
        elif self.hy_params.optimizer == 'adamw':
            self.optimizer = AdamW(self.parameters(), lr=self.hy_params.lr, eps=eps)
        else:
            raise ValueError('optimizer not recognized!')

        if self.hy_params.lr_scheduler == 'steplr':
            scheduler = MultiStepLR(self.optimizer, 
                                milestones=self.hy_params.decay_step, gamma=self.hy_params.decay_gamma)
        elif self.hy_params.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, 
                                T_max=self.hy_params.num_epochs, eta_min=eps)
        elif self.hy_params.lr_scheduler == 'poly':
            scheduler = LambdaLR(self.optimizer, 
                                lambda epoch: 0.01**(epoch/self.hy_params.num_epochs))
        else:
            raise ValueError('scheduler not recognized!')
        
        return [self.optimizer], [scheduler]
        
        
    def train_dataloader(self):
        train_dataset = DoNeRFDataset(self.hy_params.root_dir, 'train', data_skip=self.hy_params.skip)
        return DataLoader(train_dataset, batch_size=self.hy_params.batch_size, shuffle=True, num_workers=64)
    
    def val_dataloader(self):
        val_dataset = DoNeRFDataset(self.hy_params.root_dir, 'val', data_skip=1)
        return DataLoader(val_dataset, batch_size=self.hy_params.batch_size, shuffle=True, num_workers=64)
    
    def get_learning_rate(self, optimizer):
        return get_learning_rate(optimizer)
        
    def img2mse(self, target_rgb, pred_rgb):
        return torch.mean((target_rgb - pred_rgb) ** 2)
    
    def mse2psnr(self, mse_loss):
        return -10. * torch.log(mse_loss) / torch.log(torch.Tensor([10.], device=mse_loss.device))
    
    def img_to8b(self, normalized_img):
        return (255*np.clip(normalized_img,0,1)).astype(np.uint8)
    
    def depth_to8b(self, depth):
        return (255*((depth - depth.min()) / (depth.max() - depth.min()))).astype(np.uint8)



if __name__ == '__main__':
    hparams = get_opts()
    model = DGNeRFSystem(hparams)
    # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'logs/{hparams.exp_name}/ckpts',
    #                                                             '{epoch:d}'),
    #                                       monitor='val/psnr', mode='max', save_top_k=5,)
    # logger = TensorBoardLogger(save_dir="logs", name=hparams.exp_name)
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      gpus=4)
                    #   checkpoint_callback=checkpoint_callback,
                    #   logger=logger,)
    trainer.fit(model=model)