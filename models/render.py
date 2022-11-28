import torch


def sample_directions(rays_d):
    """Get another two dimensional direction vector perpendicular to rays_d

    Inputs:
        rays_d: (N_rays, 3) Input directions

    Outputs:
        rays_y, rays_z: (N_rays, 3)
    """

    eps = 1e-9
    N_rays = rays_d.shape[0]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    
    rays_y = torch.ones((N_rays, 3), dtype=torch.float32)
    rays_y[:, 1] = rays_d[:, 2]
    rays_y[:, 2] = rays_d[:, 1]

    rays_y[:, 0] = -(rays_d[:, 1] * rays_y[:, 1] + rays_d[:, 2] * rays_y[:, 2]) / (rays_d[:, 0] + eps)
    rays_y = rays_y / torch.norm(rays_y, dim=1, keepdim=True)

    x1, x2, x3 = rays_d[:, 0], rays_d[:, 1], rays_d[:, 2]
    y1, y2, y3 = rays_y[:, 0], rays_y[:, 1], rays_y[:, 2]

    rays_z = torch.ones_like(rays_d)
    rays_z[:, 2] = (rays_d[:, 2] + rays_y[:, 2]) / 2
    rays_z[:, 1] = rays_z[:, 2] * (x3*y1 - x1*y3) / (x1*y2 - x2*y1)
    rays_z[:, 0] = -(x2*rays_z[:, 1] + x3*rays_z[:, 2]) / x1
    rays_z = rays_z / torch.norm(rays_z, dim=1, keepdim=True)
    
    return rays_y, rays_z



def volumne_render(model, embeddings, samples, rays_d, depth, chunk, dis, white_back):
        """process of volumne render

        Inputs:
            model (nn.Module): NeRF-based model
            embeddings (nn.Module): postions embedders
            samples (torch.Tensor): (N_rays, n_samples+1) samples at specific position alone ray
            chunk: batch size of samples to inference
            white_back: white background or not 
        """
        N_rays = samples.shape[0]
        N_samples = samples.shape[1]
        samples = samples.reshape(-1, 3)
        dirs = rays_d[:, 0, :]
        rays_d = rays_d.reshape(-1, 3)
        
        out_chunk = []
        for i in range(0, N_rays, chunk):
            xyz_embedded = embeddings[0](samples[i: i+chunk])   # (chunk, 63)
            dir_embedded = embeddings[1](rays_d[i: i+chunk])    # (chunk, 27)
            out_chunk += [model(xyz_embedded, dir_embedded)]    # (chunk, 4)
        out = torch.cat(out_chunk, 0)
        out = out.reshape(N_rays, N_samples, -1)   # (N_rays, N_samples, 4)
        
        # Get rgb and sigma to render
        rgbs = out[..., :3] # (N_rays, N_samples_, 3)
        sigmas = out[..., 3] # (N_rays, N_samples_)
        
        # Compute weights
        ## deltas
        deltas = torch.full((N_rays, N_samples-1), dis, device=samples.device) # (N_rays, n_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1], device=samples.device)
        deltas = torch.cat([deltas, delta_inf], -1) # (N_rays, N_samples_)
        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dirs.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * 0.0
        
        ## compute alpha by formula (3)
        alphas = 1 - torch.exp(-deltas * torch.nn.Softplus()(sigmas+noise))  # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]    # (N_rays, N_samples_)
        weights_sum = weights.sum(1)    # (N_rays)
        
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2)   # (N_rays, 3)
        depth_final = torch.sum(weights*depth.squeeze(2), -1)             # (N_rays)
        
        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
        
        return rgb_final, depth_final, weights, rgbs, sigmas


def render_rays(model, embeddings, rays, depth, 
                dis=0.1, n_samples=2, chunk=1024*32, 
                white_back=True, guide='depth_sample'):
    
    assert guide in ['depth_sample', 'gauss_sample']
    
    N_rays = rays.shape[0]
    
    if guide == 'depth_sample':
        z_steps = torch.linspace(-dis, dis, n_samples+1, device=depth.device)
        depth = depth.expand(N_rays, n_samples+1)# N_rays,1 -> N_rays, n_samples+1
        depth = (depth + z_steps).reshape(-1, 1)# N_rays, n_samples+1 -> (N_rays * n_samples+1), 1
        
        rays_o = rays[:, :3][:, None, :]
        rays_d = rays[:, 3:][:, None, :]
        rays_o = rays_o.repeat(1, n_samples+1, 1).reshape(-1, 3)
        rays_d = rays_d.repeat(1, n_samples+1, 1).reshape(-1, 3)
        samples = rays_o + rays_d * depth
        
        depth = depth.reshape(N_rays, -1, 1)        # (N_rays, n_sample+1, 1)
        rays_d = rays_d.reshape(N_rays, -1, 3)
        samples = samples.reshape(N_rays, -1, 3)
        
        
    elif guide == 'gauss_sample':
        z_steps = torch.linspace(-dis, dis, n_samples+1, device=depth.device)
        
        rays_o = rays[:, :3]
        rays_d = rays[:, 3:]
        
        depth_samples = rays_o + rays_d * depth
        rays_y, rays_z = sample_directions(rays_d)
        directions = torch.cat([rays_d, -rays_d, rays_y, -rays_y, rays_z, -rays_z], 0)
        print(directions.shape)
        depth_samples_ = depth_samples.repeat(6, 1)
        yz_samples = depth_samples_ + dis * directions
        samples = torch.cat([depth_samples, yz_samples], 0)
        print(samples.shape)
        
        
        
    
    rgb_final, depth_final, weights, rgbs, sigmas = \
        volumne_render(model, embeddings, samples, rays_d, depth, chunk, dis, white_back)
    
    result = {'rgb': rgb_final, 
              'depth': depth_final}
    
    return result
    

if __name__ == '__main__':
    rays_d = torch.randn(10, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = torch.ones((10, 3))
    depth = torch.arange(2, 12, 1)[:, None]
    
    render_rays(None, None, torch.cat([rays_o, rays_d], -1), depth, dis=0.1, n_samples=2, guide='gauss_sample')
