import open3d as o3d 
import numpy as np


radius = 0.03
resolution = 4


# pc_path = r'E:\data_check\realcom\rc_net_20\res_1e1\1\res_2.ply'
# pc_path = r'E:\data_check\realcom\rc_net_20\res_1e1\1\gts.ply'
pc_path = r'E:\data_check\realcom\rc_net_20\res_1e1\0\input.ply'


pcd = o3d.io.read_point_cloud(pc_path)
points = np.array(pcd.points)

points = np.stack([points[:,0], points[:,2], points[:,1]], 1)

# points[:,0] *= -1
# points[:,1] *= -1


res = []

for point in points:
    ball = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
    ball = ball.translate(point)
    res.append(ball)

res_ply = res[0]
for r in res[1:]:
    res_ply += r 

res_ply.paint_uniform_color([0.8, 0.45, 0.24])

# o3d.io.write_triangle_mesh('res.ply', res_ply)
o3d.io.write_triangle_mesh('res_1.ply', res_ply)