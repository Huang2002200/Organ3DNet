'''
python 03_Calculate_the_phenotypic_traits.py --search 200 --importance 3 --folder "/origin(align results)" --dest "/OUTPUT"
'''
import numpy as np
import open3d as o3d
import os
from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class SLIC:
    def __init__(self, cloud, m=1.0, s=1.0, L2_min=1.0, seed_count=1024):
        self.m = m
        self.s = s
        self.L2_min = L2_min
        self.superpixelcount = 0
        self.cloud = cloud
        self.seed = None
        self.clusters = []
        self.label = [] 
        self.seed_count = seed_count

    def set_input_cloud(self, cloud):
        self.cloud = cloud

    def random_sampling(self, nsample):
        indices = np.random.choice(len(self.cloud.points), nsample, replace=False)
        sampled_cloud = o3d.geometry.PointCloud()
        for idx in indices:
            sampled_cloud.points.append(self.cloud.points[idx])
        return sampled_cloud

    def uniform_sampling(self, radius):
        voxel_down_pcd = self.cloud.voxel_down_sample(voxel_size=radius)
        return voxel_down_pcd

    def slic_superpointcloudclusting(self):
        slic_dist = [1000.0] * len(self.cloud.points)
        clusting_center = o3d.geometry.PointCloud()
        clusting_center.points = self.cloud.points
        new_clusting_center = o3d.geometry.PointCloud()
        new_clusting_center.points = self.cloud.points

        self.seed = self.random_sampling(self.seed_count)
        new_clusting_center.points = self.seed.points

        kd_tree = o3d.geometry.KDTreeFlann(self.cloud)
        print("Start SLIC")
        iter_count = 0
        L2 = float('inf')
        while L2 >= self.L2_min and iter_count < 10:
            self.label = [-1] * len(self.cloud.points)
            slic_dist = [1000.0] * len(self.cloud.points)
            clusting_center.points = new_clusting_center.points
            counter = [0] * len(self.seed.points)
            for i in range(len(self.seed.points)):
                [_, idx, _] = kd_tree.search_radius_vector_3d(self.seed.points[i], 2.0 * self.s)
                for j in idx:
                    dist_slic = self.calculate_slic_dist(self.cloud.points[j], clusting_center.points[i])
                    if dist_slic < slic_dist[j]:
                        slic_dist[j] = dist_slic
                        self.label[j] = i

            # clear clusting center
            for center in new_clusting_center.points:
                center[0], center[1], center[2] = 0, 0, 0

            for i, lab in enumerate(self.label):
                if lab != -1:
                    new_clusting_center.points[lab][0] += self.cloud.points[i][0]
                    new_clusting_center.points[lab][1] += self.cloud.points[i][1]
                    new_clusting_center.points[lab][2] += self.cloud.points[i][2]
                    counter[lab] += 1

            for i, center in enumerate(new_clusting_center.points):
                if counter[i] != 0:
                    center[0] /= counter[i]
                    center[1] /= counter[i]
                    center[2] /= counter[i]

            L2 = sum([(new_clusting_center.points[i][0] - clusting_center.points[i][0]) ** 2 +
                      (new_clusting_center.points[i][1] - clusting_center.points[i][1]) ** 2 +
                      (new_clusting_center.points[i][2] - clusting_center.points[i][2]) ** 2
                      for i in range(len(clusting_center.points))])

            iter_count += 1
        print("SLIC finished")
        self.seed.points = new_clusting_center.points

    def calculate_slic_dist(self, cloud_point, seed_point):
        sp_dist = np.linalg.norm(np.array(cloud_point) - np.array(seed_point))
        return np.sqrt(sp_dist) * self.m / self.s

    def get_labeled_cloud(self):
        labels = self.label
        return labels

    def get_seed(self):
        return self.seed

def get_label_count(cloud):
    return max([int(p.label) for p in cloud.points]) + 1

def cal_undersegmentation_error(a, gt, is_new=True):
    supercount = get_label_count(a)
    classcount = get_label_count(gt)
    
    hashmap = np.zeros((classcount, supercount))
    label_count = [0] * classcount
    pixel_count = [0] * supercount
    label_pixel = [0] * classcount
    
    for i in range(len(gt.points)):
        hashmap[int(gt.points[i].label)][int(a.points[i].label)] += 1
        label_count[int(gt.points[i].label)] += 1
        pixel_count[int(a.points[i].label)] += 1
    
    if is_new:
        for i in range(classcount):
            for j in range(supercount):
                if hashmap[i][j] != 0:
                    label_pixel[i] += min(pixel_count[j] - hashmap[i][j], hashmap[i][j])
        errcount = sum(label_pixel)
    else:
        for i in range(classcount):
            for j in range(supercount):
                if hashmap[i][j] != 0:
                    label_pixel[i] += pixel_count[j]
        errcount = sum(label_pixel) - len(gt.points)
    
    return float(errcount) / float(len(gt.points))

def cal_achievable_seg_acc(a, gt):
    supercount = get_label_count(a)
    classcount = get_label_count(gt)
    
    hashmap = np.zeros((supercount, classcount))
    classpred = [0] * classcount
    classgt = [0] * classcount
    
    for i in range(len(gt.points)):
        hashmap[int(a.points[i].label)][int(gt.points[i].label)] += 1
        classgt[int(gt.points[i].label)] += 1
    
    for i in range(supercount):
        classpred[np.argmax(hashmap[i])] += max(hashmap[i])
    
    return float(sum(classpred)) / float(sum(classgt))


def color_point_cloud_by_labels(cloud, labels, unique_flag=False, target_label=0, del_flag=False):
    unique_labels = np.unique(labels)
    
    if unique_flag == False:
        all_colors = np.random.rand(10000, 3)

        np.random.shuffle(all_colors)
        label_to_color = {label: all_colors[i] for i, label in enumerate(unique_labels)}
        colors = np.array([label_to_color[label] for label in labels])

        cloud.colors = o3d.utility.Vector3dVector(colors)

    elif del_flag == False:
        colors = np.array([[1, 0, 0] if label == target_label else [0.5, 0.5, 0.5] for label in labels])

        cloud.colors = o3d.utility.Vector3dVector(colors)
    
    else:
        colors = np.array([[1, 0, 0] if label == target_label else [1, 1, 1] for label in labels])
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def compute_rotation_matrix_to_align_with_xy(normal_vector):
    """
    Compute a rotation matrix that rotates the given normal vector to align with the z-axis.
    
    Parameters:
        - normal_vector: 3D normal vector to be aligned with z-axis
        
    Returns:
        - rotation_matrix: Rotation matrix that rotates normal_vector to be aligned with z-axis
    """
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        raise ValueError("Zero vector provided")

    normal_vector /= norm
    z_axis = np.array([0, 0, -1])
    
    if np.allclose(normal_vector, z_axis):
        return np.eye(3)

    v = np.cross(normal_vector, z_axis)
    c = np.dot(normal_vector, z_axis)
    s = np.linalg.norm(v)
    v_skew_symmetric = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    rotation_matrix = np.eye(3) + v_skew_symmetric + (v_skew_symmetric @ v_skew_symmetric) * ((1 - c) / s**2)
    return rotation_matrix


def delaunay_triangulation(points):
    points = np.array(points, dtype=np.float32)
    tri = Delaunay(points, incremental=True, qhull_options="QJ Qs" )
    delaunay_triangles = tri.points[tri.simplices]
    return delaunay_triangles

def get_bounding_rect(points, padding=10):
    """获取自适应的矩形边界并加上一定的填充"""

    min_x = min([p[0] for p in points]) - padding
    min_y = min([p[1] for p in points]) - padding
    width = max([p[0] for p in points]) - min_x + 2 * padding
    height = max([p[1] for p in points]) - min_y + 2 * padding

    return (min_x, min_y, width, height)

def rect_contains(rect, point):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

def compute_total_projected_area_delaunay(points, labels, lambda_threshold=10):
    """
    Compute the total projected area of the point cloud using Delaunay triangulation.
    """
    unique_labels = np.unique(labels)
    total_area = 0.0
    
    for label in unique_labels:
        label_points = np.asarray(points.points)[labels == label]

        # Compute PCA to get the normal of the plane
        mean = np.mean(label_points, axis=0)
        centered_points = label_points - mean
        _, _, eigenvectors = np.linalg.svd(centered_points)

        normal_vector = eigenvectors[-1]
        rotation_matrix = compute_rotation_matrix_to_align_with_xy(normal_vector)
        
        # Rotate points to align with XY plane
        rotated_points = (rotation_matrix @ label_points.T).T

        # Project to XY plane (just remove Z coordinate)
        projected_points_2d = rotated_points[:, :2]

        # Apply Delaunay triangulation
        delaunay_triangles = delaunay_triangulation(projected_points_2d)

        # Compute edge lengths for all triangles and get median
        edge_lengths = []
        for triangle in delaunay_triangles:
            edge_lengths.extend([
                np.linalg.norm(np.array(triangle[1]) - np.array(triangle[0])),
                np.linalg.norm(np.array(triangle[2]) - np.array(triangle[1])),
                np.linalg.norm(np.array(triangle[0]) - np.array(triangle[2])),
            ])
        median_length = np.median(edge_lengths)

        # Compute area for triangles satisfying the condition
        for triangle in delaunay_triangles:
            lengths = [
                np.linalg.norm(np.array(triangle[1]) - np.array(triangle[0])),
                np.linalg.norm(np.array(triangle[2]) - np.array(triangle[1])),
                np.linalg.norm(np.array(triangle[0]) - np.array(triangle[2])),
            ]
            if all(length <= lambda_threshold * median_length for length in lengths):
                v1 = np.array(triangle[1]) - np.array(triangle[0])
                v2 = np.array(triangle[2]) - np.array(triangle[0])
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                total_area += area

    return total_area

def visualize_delaunay_triangulations(points, triangles, labels, target=0, lambda_threshold=10):
    plt.figure()
    edge_lengths = []
    for triangle in triangles:
        edge_lengths.extend([
            np.linalg.norm(np.array(triangle[1]) - np.array(triangle[0])),
            np.linalg.norm(np.array(triangle[2]) - np.array(triangle[1])),
            np.linalg.norm(np.array(triangle[0]) - np.array(triangle[2])),
        ])
    median_length = np.median(edge_lengths)

    # Compute area for triangles satisfying the condition
            
    for tri in triangles:
        lengths = [
                np.linalg.norm(np.array(tri[1]) - np.array(tri[0])),
                np.linalg.norm(np.array(tri[2]) - np.array(tri[1])),
                np.linalg.norm(np.array(tri[0]) - np.array(tri[2])),
            ]
        if all(length <= lambda_threshold * median_length for length in lengths):
            x1, y1 = tri[0]
            x2, y2 = tri[1]
            x3, y3 = tri[2]
            plt.plot([x1, x2], [y1, y2], 'b-')
            plt.plot([x2, x3], [y2, y3], 'b-')
            plt.plot([x3, x1], [y3, y1], 'b-')

    plt.gca().invert_yaxis()  # Y轴坐标与图像坐标系相反，因此进行翻转
    plt.show()

def compute_principal_normal(point_cloud):
    """Compute the principal normal of the point cloud using PCA."""
    covariance_matrix = np.cov(np.asarray(point_cloud.points).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvectors[:, np.argmin(eigenvalues)]

def compute_rotation_matrix_to_align_vectors(vector_from, vector_to):
    """Compute the rotation matrix to align vector_from to vector_to."""
    v = np.cross(vector_from, vector_to)
    c = np.dot(vector_from, vector_to)
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def compute_gpt(point_cloud):
    
    # 计算主法向量
    principal_normal = compute_principal_normal(point_cloud)
    
    # 计算将主法向量与z轴对齐所需的旋转矩阵
    rotation_matrix = compute_rotation_matrix_to_align_vectors(principal_normal, [0, 0, -1])
    
    # 旋转点云
    point_cloud.points = o3d.utility.Vector3dVector(np.dot(rotation_matrix, np.asarray(point_cloud.points).T).T)
    
    # 估计法线
    point_cloud.estimate_normals()

    depth = 5  # Depth for the Poisson reconstruction. You can adjust this based on your dataset.
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth)
    
    # 计算所有三角形的法线
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    # 寻找z分量为正的三角形
    upper_triangles_indices = np.where(triangle_normals[:, 2] > -10)[0]

    # 保留z分量为正的三角形
    mesh.triangles = o3d.utility.Vector3iVector(triangles[upper_triangles_indices])
    mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals[upper_triangles_indices])

    # 计算表面积
    surface_area = mesh.get_surface_area()
    o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
    return surface_area
def compute_principal_axes(point_cloud):
    """Compute the principal axes of the point cloud using PCA."""
    covariance_matrix = np.cov(np.asarray(point_cloud.points).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)
    return eigenvectors[:, sorted_indices]

def compute_rotation_matrix_to_align_vectors_leaf(v1, v2):
    """Compute the rotation matrix to align vector v1 to v2."""
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def homogeneous_transformation_matrix(rotation_matrix):
    """Construct a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    return transform


def transform_aabb(aabb, transformation):
    """Transform an AABB using a given transformation matrix."""
    corners = np.asarray(aabb.get_box_points())
    # Apply rotation
    rotated_corners = np.dot(corners, transformation[:3, :3].T)
    # Apply translation
    transformed_corners = rotated_corners + transformation[:3, -1]
    min_bound = np.min(transformed_corners, axis=0)
    max_bound = np.max(transformed_corners, axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

def compute_principal_axes(point_cloud):
    """Compute the principal axes of the point cloud using PCA."""
    covariance_matrix = np.cov(np.asarray(point_cloud.points).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)
    return eigenvectors[:, sorted_indices]

def compute_leaf_metrics_and_box(point_cloud):
    """Compute leaf length, width, tilt angle, and visualize with a bounding box."""
    # 1. Compute the centroid
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    
    # 2. Compute principal axes using PCA
    principal_axes = compute_principal_axes(point_cloud)
    length_idx = np.argmax(np.abs(principal_axes[0, :]))

    # 从剩余的列中找到第二个元素绝对值最大的列索引
    remaining_axes = np.delete(principal_axes, length_idx, axis=1)
    width_idx_in_remaining = np.argmax(np.abs(remaining_axes[1, :]))

    # 找到剩余的列，这将是 leaf_normal_dir
    remaining_axes = np.delete(remaining_axes, width_idx_in_remaining, axis=1)
    leaf_normal_dir = remaining_axes[:, 0]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.points)

    rotation_matrix = compute_rotation_matrix_to_align_vectors_leaf(leaf_normal_dir, [0, 0, -1])
    rotated_points = np.dot(rotation_matrix, (np.asarray(point_cloud.points) - centroid).T).T
    aligned_cloud = o3d.geometry.PointCloud()
    aligned_cloud.points = o3d.utility.Vector3dVector(rotated_points)
    min_bound_aligned = np.min(rotated_points, axis=0)
    max_bound_aligned = np.max(rotated_points, axis=0)  
    extents_aligned = max_bound_aligned - min_bound_aligned
    leaf_length = max(extents_aligned)
    leaf_width = sorted(extents_aligned)[-2]

    possible_vectors = [
        [0, 0, 1], [0, 0, -1]
    ]
    min_tilt_angle = float('inf')
    for v in possible_vectors:
        dot_product = np.dot(leaf_normal_dir, v)

        tilt_angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
        print(tilt_angle)
        # 如果 tilt_angle 不为0 并且比之前计算的小，则更新 min_tilt_angle
        if tilt_angle != 0 and tilt_angle < min_tilt_angle:
            min_tilt_angle = tilt_angle


    return leaf_length, leaf_width, min_tilt_angle

def remove_outliers(pcd, method="statistical", nb_neighbors=20, std_ratio=2.0, radius=0.1, min_nb_neighbors=10):
    if method == "statistical":
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        cl, ind = pcd.remove_radius_outlier(nb_points=min_nb_neighbors, radius=radius)
    else:
        raise ValueError("Unknown method: {}".format(method))
    return cl

def get_seed_count(num_points):
    if num_points <= 100:
        return 1
    if num_points <= 200:
        return 2
    if num_points <= 500:
        return 3
    elif num_points <= 1000:
        return 6
    elif num_points <= 5000:
        return 20
    elif num_points <= 10000:
        return 30
    elif num_points <= 30000:
        return 60
    elif num_points <= 60000:
        return 100
    else:
        return 150  # For points > 60000

def process_instance(filename, label, label_data, label_data_downscale, args, areas):
    # Scale coordinates
    p = 10
    label_data[:, :3] = label_data[:, :3] / p * 0.1 * 100.0
    label_data_downscale[:, :3] = label_data_downscale[:, :3] / p * 0.1 * 100.0

    # Create PointCloud objects
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(label_data[:, :3])

    cloud_downscale = o3d.geometry.PointCloud()
    cloud_downscale.points = o3d.utility.Vector3dVector(label_data_downscale[:, :3])

    # Get seed_count based on number of points
    seed_count = get_seed_count(label_data.shape[0])

    # Apply SLIC segmentation
    slic = SLIC(cloud_downscale, args.importance, args.search, args.l2, seed_count)
    slic.slic_superpointcloudclusting()

    retries = 0
    max_retries = 9
    success = False

    while retries < max_retries and not success:
        try:
            labeled_o3d = o3d.geometry.PointCloud()
            labeled = slic.get_labeled_cloud()
            labeled_o3d.points = slic.cloud.points
            labeled_o3d.colors = slic.cloud.colors

            # Visualize and save point cloud
            color_point_cloud_by_labels(labeled_o3d, labeled, unique_flag=True, target_label=0)
            color_point_cloud_by_labels(labeled_o3d, labeled, unique_flag=True, target_label=0, del_flag=True)

            # Save as txt and PLY
            base_filename = f"slic_{filename}_{label}.txt"

            with open(os.path.join(args.dest, base_filename), 'w') as f:
                for i, point in enumerate(np.asarray(labeled_o3d.points)):
                    f.write(f"{point[0]} {point[1]} {point[2]} {labeled[i]}\n")

            # Compute area and leaf metrics
            area = compute_total_projected_area_delaunay(labeled_o3d, labeled)
            l, w, t = compute_leaf_metrics_and_box(cloud)

            areas[label] = {
                'total_area': area,
                'leaf_length': l,
                'leaf_width': w,
                'tile_angle': t,
            }

            print(f"Label {label} - Area: {area}, Length: {l}, Width: {w}, angle: {t}")
            success = True
        except IndexError as e:
            print(f"Error: {e}. Retrying...")
            retries += 1

    return success

def main_updated(args):
    if not os.path.exists(args.folder):
        print("No such path")
        return -1

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    max_retries = 9  # Set a maximum number of retries to avoid infinite loops

    # Open the results file for appending
    results_file_path = os.path.join(args.dest, "leaf_areas_and_metrics.txt")
    with open(results_file_path, 'a') as results_file:

        for filename in os.listdir(args.folder):
            if filename.endswith(".txt"):
                areas = {}
                start = time.time()
                print(f"Processing {filename}...")

                file_path = os.path.join(args.folder, filename)
                data = np.loadtxt(file_path)
                downscaled_file = os.path.join(file_path)
                data_downscale = np.loadtxt(downscaled_file)

                # Extract unique labels greater than 0 (leaf instances)
                labels = np.unique(data[:, 3])
                labels = labels[labels > 0]

                for label in labels:
                    # Extract data for the current label
                    label_data = data[data[:, 3] == label]
                    label_data_downscale = data_downscale[data_downscale[:, 3] == label]

                    # Process the instance and calculate metrics
                    success = process_instance(filename, label, label_data, label_data_downscale, args, areas)

                    if not success:
                        print(f"Failed to process label {label} for {filename} after {max_retries} retries. Skipping...")

                if areas:
                    results_file.write(f"{filename}:\n")
                    for label, metrics in areas.items():
                        results_file.write(f"  Label {label}:\n")
                        for metric, value in metrics.items():
                            results_file.write(f"    {metric}: {value}\n")
                    results_file.flush()  # Force flush after writing to file
                else:
                    print(f"No valid data for {filename}. Skipping writing to file.")

                end = time.time()
                print(f"Done processing {filename} in {end - start:.2f} seconds")

    print(f"All results saved to {results_file_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="SLIC like Superpixel using PCL Library in Python")
    parser.add_argument("--search", "-s", type=float, default=200, help="Search radius for superpixel segmentation (bigger = larger superpixels)")
    parser.add_argument("--importance", "-m", type=float, default=3, help="Weight of spatial position (bigger = more regular superpixel shape)")
    parser.add_argument("--l2", "-l", type=float, default=0.000001, help="Minimum error to stop algorithm iteration (smaller = more accurate)")
    #parser.add_argument("--seed_count", type=int, default=30, help="Number of initial seeds (more = smaller/more superpixels)")
    parser.add_argument("--folder", "-f", type=str, default="/instances", help="Folder with input point cloud data to be segmented")
    parser.add_argument("--dest", "-d", type=str, default="/outputs", help="Folder to save the segmentation results")
    args = parser.parse_args()
    main_updated(args)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 