'''
python traindata_agument03.py --input /path/to/train_dataset --output /path/to/output/directory
'''
import os
import numpy as np
import shutil
import argparse


class PointCloudProcessor:
    def farthest_point_sampling(self, points, num_samples):

        N, C = points.shape
        centroids = np.zeros((num_samples,), dtype=np.int64)
        distances = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = np.argmax(distances)
        sampled_points = points[centroids]
        return sampled_points

    def splitPointBlock(self, cloud, size=20.0, size1=20.0, num_fps_points=8):
        """
        Split the point cloud into blocks based on FPS.
        """
        coordinates = cloud[:, :3]
        used_centers = []
        blocks = []

        cloud_center = np.mean(coordinates, axis=0)
        perturbation = np.random.uniform(-1, 1, 3)
        center_with_perturbation = cloud_center + perturbation
        x_center, y_center, z_center = center_with_perturbation

        x_start = x_center - size1 / 2
        x_end = x_center + size1 / 2
        y_start = y_center - size1 / 2
        y_end = y_center + size1 / 2
        z_start = z_center - size1 / 2
        z_end = z_center + size1 / 2

        center_block = cloud[(cloud[:, 0] >= x_start) & (cloud[:, 0] < x_end) &
                             (cloud[:, 1] >= y_start) & (cloud[:, 1] < y_end) &
                             (cloud[:, 2] >= z_start) & (cloud[:, 2] < z_end)]
        blocks.append(center_block)
        used_centers.append(tuple(center_with_perturbation))

        while len(blocks) < 10:
            fps_points = self.farthest_point_sampling(coordinates, num_fps_points)

            for center in fps_points:
                center_tuple = tuple(center)
                if center_tuple in used_centers:
                    continue

                used_centers.append(center_tuple)

                x_center, y_center, z_center = center
                x_start = x_center - size / 2
                x_end = x_center + size / 2
                y_start = y_center - size / 2
                y_end = y_center + size / 2
                z_start = z_center - size / 2
                z_end = z_center + size / 2

                block = cloud[(cloud[:, 0] >= x_start) & (cloud[:, 0] < x_end) &
                              (cloud[:, 1] >= y_start) & (cloud[:, 1] < y_end) &
                              (cloud[:, 2] >= z_start) & (cloud[:, 2] < z_end)]

                if block.shape[0] >= (0.2 * cloud.shape[0]):
                    blocks.append(block)
                if len(blocks) >= 9:
                    break

        return blocks

    def process_directory(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                cloud = np.loadtxt(os.path.join(input_dir, filename))
                print(f"Processing {filename}, num_points: {cloud.shape[0]}")

                blocks = self.splitPointBlock(cloud)
                for i, block in enumerate(blocks):
                    output_filename = os.path.join(output_dir, f"{filename[:-4]}_block_{i+1}.txt")
                    np.savetxt(output_filename, block, fmt='%.6f')
                    print(f"Saved {output_filename}, num_points: {block.shape[0]}")

        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                new_filename = filename.replace('.txt', '_block_0.txt')
                new_file_path = os.path.join(output_dir, new_filename)
                shutil.copy(file_path, new_file_path)

        print("File processing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process point cloud files and split them into blocks.')
    parser.add_argument('--input', required=True, help='Input directory containing point cloud files')
    parser.add_argument('--output', required=True, help='Output directory for processed point clouds')

    args = parser.parse_args()
    processor = PointCloudProcessor()
    processor.process_directory(args.input, args.output)
