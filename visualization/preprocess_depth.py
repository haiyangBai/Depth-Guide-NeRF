import numpy as np
from scipy import interpolate
import cv2, os
import glob
from tqdm import tqdm
from pprint import pprint




def load_depth_image(root_dir, split='train'):
    root = os.path.join(root_dir, split)
    paths = []
    paths += glob.glob(os.path.join(root, '*.npz'))
    
    for path in tqdm(sorted(paths)):
        save_path = path.replace('.npz', '.npy')
        np_file = np.load(path)
        depth = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
        depth = depth.astype(np.float32).reshape(800, 800)[::-1, :]
        max_depth = depth.max()
        depth[depth == max_depth]  = np.nan
        depth = np.ma.masked_invalid(depth)

        x = np.arange(0, depth.shape[1])
        y = np.arange(0, depth.shape[0])

        xx,yy=np.meshgrid(x,y) 

        x1 = xx[~depth.mask]
        y1 = yy[~depth.mask]
        newarr = depth[~depth.mask].data

        depth = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
        img = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.).astype(np.uint8)
        name = save_path.split('/')[-1].split('.')[0] + '.png'
        res = 'result'
        os.makedirs(res, exist_ok=True)
        cv2.imwrite(os.path.join(res, name), img)
        np.save(save_path, depth)
 
depth_path = '/home/baihy/datasets/DONeRF-data/classroom/'
load_depth_image(depth_path, 'train')

# data = np.load('/home/baihy/my_code/depth-NeRF/depth.npy')
# print(data.shape)