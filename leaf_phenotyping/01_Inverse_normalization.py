'''
python 01_Inverse_normalization.py --original-dir "/origin" --normalized-dir "/pred" --output-dir "/unscale"
'''

import os
import numpy as np
import argparse
from typing import Dict


def load_original_scaling_params(original_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract scaling parameters (min/max coordinates, scale factor) from original point cloud files
    for inverse normalization.

    Parameters
    ----------
    original_dir : str
        Directory containing original (non-normalized) point cloud files (.txt)

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Scaling parameters dictionary with structure:
        {
            "filename.txt": {
                "min": np.ndarray (3,),  # Min coordinates (x, y, z) of original point cloud
                "max": np.ndarray (3,),  # Max coordinates (x, y, z) of original point cloud
                "scale_factor": float    # Normalization scale factor (target range = 30)
            }
        }

    """
    if not os.path.isdir(original_dir):
        raise FileNotFoundError(f"Original directory not found: {original_dir}")

    scaling_params = {}
    txt_files = [f for f in os.listdir(original_dir) if f.endswith('.txt')]

    if not txt_files:
        raise ValueError(f"No .txt files found in original directory: {original_dir}")

    for file_name in txt_files:
        file_path = os.path.join(original_dir, file_name)
        try:
            # Load original point cloud data (xyz + other attributes)
            data = np.loadtxt(file_path)

            # Ensure data has at least 3 columns (x, y, z)
            if data.shape[1] < 3:
                raise ValueError(f"Insufficient columns in {file_name}: expected ≥3, got {data.shape[1]}")

            points = data[:, :3]
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            coord_range = max_coords - min_coords

            # Handle zero range (single point case)
            if np.max(coord_range) == 0:
                scale_factor = 1.0
            else:
                scale_factor = 30.0 / np.max(coord_range)  # Target normalization range = 30

            scaling_params[file_name] = {
                'min': min_coords,
                'max': max_coords,
                'scale_factor': scale_factor
            }

        except Exception as e:
            raise RuntimeError(f"Failed to process {file_name}: {str(e)}")

    return scaling_params


def reverse_normalization(normalized_data: np.ndarray,
                          original_params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Inverse normalize the normalized point cloud coordinates back to original coordinate system.

    Parameters
    ----------
    normalized_data : np.ndarray
        Normalized point cloud data (n x d, d ≥3), first 3 columns are normalized xyz coordinates
    original_params : Dict[str, np.ndarray]
        Scaling parameters of original point cloud (min, max, scale_factor)

    Returns
    -------
    np.ndarray
        Point cloud data with coordinates mapped back to original system (same shape as input)

    """
    if normalized_data.shape[1] < 3:
        raise ValueError(f"Normalized data must have at least 3 columns (xyz), got {normalized_data.shape[1]}")

    # Extract normalized coordinates and other attributes
    normalized_points = normalized_data[:, :3]
    other_attributes = normalized_data[:, 3:] if normalized_data.shape[1] > 3 else None

    # Inverse normalization calculation
    original_points = normalized_points / original_params['scale_factor'] + original_params['min']

    # Recombine with other attributes if present
    if other_attributes is not None:
        original_data = np.hstack((original_points, other_attributes))
    else:
        original_data = original_points

    return original_data


def process_directory(original_dir: str,
                      normalized_dir: str,
                      output_dir: str) -> None:
    """
    Main function to inverse normalize all point cloud files in batch:
    1. Load scaling parameters from original directory
    2. Inverse normalize each normalized file
    3. Save results to output directory

    Parameters
    ----------
    original_dir : str
        Directory of original (non-normalized) point cloud files
    normalized_dir : str
        Directory of normalized point cloud files to be inverse transformed
    output_dir : str
        Directory to save inverse normalized results
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load scaling parameters from original files
    try:
        scaling_params = load_original_scaling_params(original_dir)
        print(f"Successfully loaded scaling parameters for {len(scaling_params)} files")
    except Exception as e:
        print(f"Error loading scaling parameters: {str(e)}")
        return

    # Process each normalized file
    normalized_files = [f for f in os.listdir(normalized_dir) if f.endswith('.txt')]
    processed_count = 0

    for file_name in normalized_files:
        # Skip if no matching scaling parameters
        if file_name not in scaling_params:
            print(f"Warning: No scaling parameters found for {file_name}, skipping")
            continue

        try:
            # Load normalized data
            normalized_path = os.path.join(normalized_dir, file_name)
            normalized_data = np.loadtxt(normalized_path)

            if normalized_data.size == 0:
                print(f"Warning: {file_name} is empty, saving empty file")
                np.savetxt(os.path.join(output_dir, file_name), normalized_data, fmt='%.6f')
                processed_count += 1
                continue
            original_data = reverse_normalization(normalized_data, scaling_params[file_name])
            output_path = os.path.join(output_dir, file_name)
            np.savetxt(output_path, original_data, fmt='%.6f', delimiter=' ')

            processed_count += 1
            print(f"Processed: {file_name} -> {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    print(f"\nProcessing completed: {processed_count}/{len(normalized_files)} files successfully processed")


def main():
    parser = argparse.ArgumentParser(
        description="Inverse normalize normalized point cloud coordinates back to original coordinate system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--original-dir', '-o',required=True,
                        help='Directory containing original (non-normalized) point cloud files (.txt)')
    parser.add_argument('--normalized-dir', '-n', required=True,
                        help='Directory containing normalized point cloud files to be inverse transformed')
    parser.add_argument('--output-dir', '-d',required=True,
                        help='Directory to save inverse normalized point cloud results')
    args = parser.parse_args()
    process_directory(args.original_dir, args.normalized_dir, args.output_dir)


if __name__ == "__main__":
    main()