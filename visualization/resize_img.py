import os, cv2, glob

root_dir = '/home/baihy/datasets/nju图书馆_remove_callnumber/nju图书馆_remove_callnumber'
img_paths = glob.glob(os.path.join(root_dir, '*.jpeg'))
save_dir = 'resize_' + root_dir.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)

hw = (1080, 1944)

for path in img_paths:
    img = cv2.imread(path)
    print(img.shape)
    img = cv2.resize(img, hw)
    print(img.shape)
    save_path = os.path.join(save_dir, path.split('/')[-1])
    cv2.imwrite(save_path, img)