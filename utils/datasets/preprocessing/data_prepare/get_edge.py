'''
python get_edge.py --input /path/to/scaled pointclouds/directory --output /path/to/labeled pointclouds/directory
'''

import os
import open3d as o3d
import numpy as np
import argparse
def get_edge(input_path,output_path):
    files = os.listdir(input_path)
    for file_name in files:
        full_file_path = os.path.join(input_path, file_name)

        points = np.loadtxt(full_file_path)
        pcd = o3d.t.geometry.PointCloud(points[:, :3])
        pcd.estimate_normals(max_nn=20)

        boundary_points, mask = pcd.compute_boundary_points(radius=10, max_nn=20, angle_threshold=90)

        edge_point_index = mask.numpy()
        labeled_points = points.copy()

        # Add labels: 1 for boundary points, 0 for core points
        labels = np.zeros(labeled_points.shape[0], dtype=int)
        labels[edge_point_index] = 1

        labeled_points = np.hstack((labeled_points, labels[:, np.newaxis]))  # Add edge labels as the last column

        np.savetxt(os.path.join(output_path, file_name), labeled_points, fmt="%.6f")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Set edge points of the point cloud to label 1, and non-edge points to label 0.')
    parser.add_argument('--input', required=True, help='Input directory containing point cloud files')
    parser.add_argument('--output', required=True, help='Output directory for labeled point clouds')
    args = parser.parse_args()
    get_edge(args.input, args.output)
    print(f"Processing completed. labeled point clouds saved to {args.output}")
