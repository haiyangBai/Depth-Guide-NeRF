import torch
import torch.nn as nn

class PosEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(PosEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, dim=-1)


class DepthNeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 in_channels_xyz=63,
                 in_channels_dir=27,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels fir direction
        skips: add skip connection in the D_th layer 
        """
        super(DepthNeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        for i in range(D-1):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            nn.init.kaiming_normal_(layer.weight)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i+1}', layer)

        layer_xyz = nn.Linear(W, W+1)
        layer_dir = nn.Linear(W+in_channels_dir, W//2)
        layer_rgb = nn.Linear(W//2, 3)
        
        nn.init.kaiming_normal_(layer_xyz.weight)
        nn.init.kaiming_normal_(layer_dir.weight)
        nn.init.kaiming_normal_(layer_rgb.weight)
        
        self.sigma_xyz_encoding = nn.Sequential(layer_xyz, nn.Softplus())
        self.dir_encoding = nn.Sequential(layer_dir, nn.ReLU(True))
        self.rgb = nn.Sequential(layer_rgb, nn.Sigmoid())
    
    def forward(self, xyz, dir, sigma_only=False):
        """
        Encoder input (xyz + dir) to rgb+sigma (not ready to render yet),
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
                the embedded vector of postion and direction
            sigma_only: whether to infer sigma only, if True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_only:
                sigma: (B, 1)
            else:
                out: (B, 4)
        """
        input_xyz = xyz

        xyz_ = input_xyz
        for i in range(self.D-1):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f'xyz_encoding_{i+1}')(xyz_)

        sigma_xyz_encoding = self.sigma_xyz_encoding(xyz_)
        sigma = sigma_xyz_encoding[..., :1]
        xyz_encoding = sigma_xyz_encoding[..., 1:]

        if sigma_only:
            return sigma
        
        dir_encoding = self.dir_encoding(torch.cat([xyz_encoding, dir], -1))
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)
        return out
    
    
class DepthEstimator(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 in_channels_xyz=63,
                 in_channels_dir=27,
                 skips=[4],
                 out_channels=1):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels fir direction
        skips: add skip connection in the D_th layer 
        out_channels: output depth
        """
        super(DepthEstimator, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.out_channels = out_channels

        for i in range(D-1):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            # nn.init.kaiming_normal_(layer.weight)
            layer = nn.Sequential(layer, nn.ReLU())
            setattr(self, f'xyz_encoding_{i+1}', layer)

        layer_xyz = nn.Linear(W, W)
        layer_dir = nn.Linear(W+in_channels_dir, W//2)
        layer_depth = nn.Linear(W//2, out_channels)
        
        # nn.init.kaiming_normal_(layer_xyz.weight)
        # nn.init.kaiming_normal_(layer_dir.weight)
        # nn.init.kaiming_normal_(layer_depth.weight)
        
        self.xyz_encoding = nn.Sequential(layer_xyz, nn.Softplus())
        self.dir_encoding = nn.Sequential(layer_dir, nn.ReLU())
        self.depth = nn.Sequential(layer_depth, nn.Sigmoid())
    
    def forward(self, xyz, dir):
        """
        Encoder input (xyz + dir) to rgb+sigma (not ready to render yet),
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
                the embedded vector of postion and direction
        Outputs:
            out: (B, n_class)
        """
        input_xyz = xyz

        xyz_ = input_xyz
        for i in range(self.D-1):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f'xyz_encoding_{i+1}')(xyz_)

        xyz_encoding = self.xyz_encoding(xyz_)
        
        dir_encoding = self.dir_encoding(torch.cat([xyz_encoding, dir], -1))
        depth = self.depth(dir_encoding)

        return depth


if __name__ == '__main__':
    
    from time import time
    device = torch.device('cuda')
    
    t1 = time()
    for i in range(1):
        x = torch.randn(640000, 63).to(device)
        y = torch.randn(640000, 27).to(device)
        
        model = DepthEstimator(D=4, W=128, skips=[2]).to(device)
        
        out = model(x, y)
        
    t2 = time()
    
    print(t2-t1)