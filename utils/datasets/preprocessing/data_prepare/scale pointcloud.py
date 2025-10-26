'''
python scale_pointcloud.py --input /path/to/input/directory --output /path/to/output/directory --range 30
'''

import os
import numpy as np
import argparse

def scale_point_cloud(point_cloud, target_range):
    min_coords = point_cloud[:, :3].min(axis=0)
    max_coords = point_cloud[:, :3].max(axis=0)
    scale_factors = target_range / np.max(max_coords - min_coords)
    scaled_points = (point_cloud[:, :3] - min_coords) * scale_factors
    return np.hstack((scaled_points, point_cloud[:, 3:]))


def process_point_cloud_files(input_dir, output_dir, target_range=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            data = np.loadtxt(filepath)
            scaled_data = scale_point_cloud(data, target_range)

            output_filepath = os.path.join(output_dir, filename)
            np.savetxt(output_filepath, scaled_data, fmt='%f %f %f %d %d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scale point cloud data to target range.')
    parser.add_argument('--input', required=True, help='Input directory containing point cloud files')
    parser.add_argument('--output', required=True, help='Output directory for scaled point clouds')
    parser.add_argument('--range', type=float, default=30, help='Target range for scaling (default: 30)')

    args = parser.parse_args()

    process_point_cloud_files(args.input, args.output, args.range)
    print(f"Processing completed. Scaled point clouds saved to {args.output}")

