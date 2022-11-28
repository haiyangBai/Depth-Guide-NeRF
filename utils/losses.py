import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self, coff_loss):
        super(MSELoss, self).__init__()
        self.coff_loss = coff_loss
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets_rgb, target_depth):
        rgb_loss = self.loss(inputs['rgb'], targets_rgb)
        depth_loss = self.loss(inputs['depth'], target_depth)
        return rgb_loss + self.coff_loss * depth_loss

class MAELoss(nn.Module):
    def __init__(self, coff_loss):
        super(MSELoss, self).__init__()
        self.coff_loss = coff_loss
        self.loss = nn.L1Loss()

    def forward(self, inputs, targets_rgb, target_depth):
        rgb_loss = self.loss(inputs['rgb'], targets_rgb)
        depth_loss = self.loss(inputs['depth'], target_depth)
        return rgb_loss + self.coff_loss * depth_loss
               
