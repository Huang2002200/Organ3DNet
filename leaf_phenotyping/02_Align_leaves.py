'''
python 02_Align_leaves.py --true-dir "/origin" --pred-dir "/unscale" --output-dir "/align results"
'''

import os
import numpy as np
import argparse

def load_point_cloud(file_path, is_true_file=False):
    """file format: .txt"""
    data = np.loadtxt(file_path)
    if is_true_file:
        points = data[:, :3]  # xyz
        labels = data[:, 3]  # ground ins label
    else:
        points = data[:, :3]  # xyz
        labels = data[:, 3]  # pred ins label
    return points, labels


def save_point_cloud(file_path, points, labels):
    """file format: .txt"""
    data = np.hstack((points, labels.reshape(-1, 1)))
    np.savetxt(file_path, data, fmt='%.6f')


def compute_iou_matrix(true_labels, pred_labels):
    true_instances = np.unique(true_labels)
    pred_instances = np.unique(pred_labels)
    iou_matrix = np.zeros((len(true_instances), len(pred_instances)))

    for i, true_instance in enumerate(true_instances):
        true_mask = (true_labels == true_instance)
        for j, pred_instance in enumerate(pred_instances):
            pred_mask = (pred_labels == pred_instance)
            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix, true_instances, pred_instances


def match_instances(true_labels, pred_labels):
    iou_matrix, true_instances, pred_instances = compute_iou_matrix(true_labels, pred_labels)

    mapping = {}
    pred_iou_max = {p: 0.0 for p in pred_instances}

    for i, true_instance in enumerate(true_instances):
        iou_values = iou_matrix[i, :]
        max_iou_idx = np.argmax(iou_values)
        max_iou = iou_values[max_iou_idx]
        pred_instance = pred_instances[max_iou_idx]

        if max_iou > pred_iou_max[pred_instance] and max_iou >= 0.3:
            if pred_instance in mapping:
                del mapping[pred_instance]
            mapping[pred_instance] = true_instance
            pred_iou_max[pred_instance] = max_iou

    return mapping


def remap_labels(pred_labels, mapping):
    remapped_labels = np.zeros_like(pred_labels)
    for pred_instance, true_instance in mapping.items():
        remapped_labels[pred_labels == pred_instance] = true_instance
    return remapped_labels


def process_directory(true_dir, pred_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    true_files = sorted(os.listdir(true_dir))
    pred_files = sorted(os.listdir(pred_dir))

    if len(true_files) != len(pred_files):
        print(f"Warning：The number of true_files({len(true_files)}) is not match the number of pred_files ({len(pred_files)})")
        return

    for true_file, pred_file in zip(true_files, pred_files):
        true_file_path = os.path.join(true_dir, true_file)
        pred_file_path = os.path.join(pred_dir, pred_file)

        if not (true_file.endswith('.txt') and pred_file.endswith('.txt')):
            print(f"skip file：{true_file} / {pred_file}")
            continue

        try:
            true_points, true_labels = load_point_cloud(true_file_path, is_true_file=True)
            pred_points, pred_labels = load_point_cloud(pred_file_path)

            mapping = match_instances(true_labels, pred_labels)
            remapped_labels = remap_labels(pred_labels, mapping)

            output_file = os.path.join(output_dir, pred_file)
            save_point_cloud(output_file, pred_points, remapped_labels)

            print(f"{pred_file} -> {output_file} (number of instances ：{len(mapping)})")
        except Exception as e:
            print(f"Wrong with {pred_file}：{str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align leaves via max IOU')
    parser.add_argument('--true-dir', required=True, help='Directory containing ground truth labels')
    parser.add_argument('--pred-dir', required=True, help='Directory containing predicted labels (absolute path)')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed results (absolute path)')
    args = parser.parse_args()
    process_directory(args.true_dir, args.pred_dir, args.output_dir)
