import open3d as o3d
import numpy as np

# 定义路径（替换为你的实际路径和 scan_id）
scan_path = "../TSP3D-main/scannet/scans/scene0000_02"  # 示例路径
scan_id = "scene0000_02"
ply_file = f"{scan_path}/{scan_id}_vh_clean_2.ply"

# 步骤 1: 加载三角网格
mesh = o3d.io.read_triangle_mesh(ply_file)
print(f"加载网格: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个面")

# 步骤 2: 从网格提取点云（使用顶点作为点云）
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices  # 点坐标
pcd.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))  # 颜色（如果有）

# 可选: 如果你想采样更多点或均匀分布，可以在网格表面采样
pcd = mesh.sample_points_uniformly(number_of_points=10000000)  # 采样 10 万点以加速

# 步骤 3: 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="ScanNet Point Cloud Visualization")

# 可选: 保存点云为 PLY 文件（纯点云格式）
o3d.io.write_point_cloud("extracted_point_cloud.ply", pcd)