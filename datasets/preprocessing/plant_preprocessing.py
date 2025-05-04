import os
import re

import numpy as np
from fire import Fire
from loguru import logger
from natsort import natsorted

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class PlantPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/5plant",
        save_dir: str = "./data/processed/5plant",
        modes: tuple = (
            "Area_1",
            "Area_2",
        ),
        n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.class_map = {
            "benthi_stem": 0,
            "benthi_leaf": 1,
            "tomato_stem": 2,
            "tomato_leaf": 3,
            "sorghum_stem": 4,
            "sorghum_leaf": 5,
            "soybean_stem": 6,
            "soybean_leaf": 7,
            "pepper_stem": 8,
            "pepper_leaf": 9,
             # stairs are also mapped to clutter
        }

        self.color_map = [
            [0, 255, 0],  # ceiling
            [0, 0, 255],  # floor
            [0, 255, 255],  # wall
            [255, 255, 0],  # beam
            [255, 0, 255],  # column
            [100, 100, 255],  # window
            [200, 200, 100],  # door
            [170, 120, 200],  # table
            [180, 200, 120],  # door
            [170, 100, 180],  # table
         ]  # clutter

        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for txt_file in os.scandir(self.data_dir / mode):
                if txt_file.is_file() and txt_file.name.endswith('.txt'):
                    filepaths.append(txt_file.path)
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def _buf_count_newlines_gen(self, fname):
        def _make_gen(reader):
            while True:
                b = reader(2**16)
                if not b:
                    break
                yield b

        with open(fname, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
        return count

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "area": mode,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        scene_name = os.path.basename(filepath)
        data_path = self.data_dir / mode / scene_name
        scene_name = os.path.splitext(scene_name)[0]
        filebase["scene"] = scene_name
        raw_file_path = self.data_dir / mode / scene_name
        filebase["raw_filepath"] = str(raw_file_path)
        print(f"data_path:{data_path}, scene_name:{scene_name}")
        points = np.loadtxt(data_path)

        pcd_size = self._buf_count_newlines_gen(f"{filepath}")
        if points.shape[0] != pcd_size:
            print(f"FILE SIZE DOES NOT MATCH FOR {filepath}")
            print(f"({points.shape[0]} vs. {pcd_size})")

        filebase["raw_segmentation_filepath"] = ""
        xyz_info = points[:, :3]

        sem_label = points[:, 4].astype(int)
        ins_label = points[:, 3].astype(int)
        is_edge = points[:, -1:]

        ones_data = np.ones((xyz_info.shape[0], 1))
        xyz_info_with_colors_and_ones = np.hstack((xyz_info, ones_data, is_edge))

        points = np.hstack((xyz_info_with_colors_and_ones, sem_label[:, None], ins_label[:, None]))  #xyz_segid_isedge_sem_ins
        gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = (
            self.save_dir / "instance_gt" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        return filebase

    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def joint_database(
        self,
        train_modes=(
            "Area_1",
            "Area_2",
        ),
    ):
        for mode in train_modes:
            joint_db = []
            for let_out in train_modes:
                if mode == let_out:
                    continue
                joint_db.extend(
                    self._load_yaml(
                        self.save_dir / (let_out + "_database.yaml")
                    )
                )
            self._save_yaml(
                self.save_dir / f"train_{mode}_database.yaml", joint_db
            )


if __name__ == "__main__":
    Fire(PlantPreprocessing)
