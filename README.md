# Organ3DNet

This repo contains the official data and code for our paper:

### Organ3DNet: a deep network for segmenting organ semantics and instances from dense plant point clouds
[D. Li†](https://davidleepp.github.io/), J. Huang†, B. Zhao, and W. Wen*<br>
† Equal contribution<br>
Published online on *Artificial Intelligence in Agriculture* in 2025<br>
[[Paper](https://www.sciencedirect.com/science/article/pii/S2589721725000911)]
[[8-minute presentation](https://www.bilibili.com/video/BV1mn9uBME4X)]<br>

## Prerequisites<br>
You can install a conda environment by following the steps below:<br>
```
conda env create -f environment.yml
conda activate Organ3DNet

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
pip install pytorch-lightning==1.7.2
pip install -U git+https://github.com/kumuji/volumentations
pip install volumentations --no-build-isolation

cd third_party
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
cd pointnet2
python setup.py install
```
## Dataset<br>
The dataset (file 5plant_xyz_ins_sem.rar)(https://doi.org/10.5281/zenodo.17165870) contains 889 colorless point clouds of five crop species (tobacco, tomato, sorghum, soybean, and pepper), including 635 point clouds for training (folder Area_1) and 254 point clouds for testing (folder Area_2). In the total dataset, tobacco has 105 point clouds, tomato has 326 point clouds, sorghum has 129 point clouds, soybean has 92 point clouds, and pepper has 237 point clouds. Data of the tobacco, tomato, and sorghum species is descended from [PlantNet](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000119); data of the soybean is from [Soybean-MVS](https://www.mdpi.com/2077-0472/13/7/1321); and data of the pepper is selected from [Pepper-4D](https://www.mdpi.com/2223-7747/15/4/599).<br><br>
All raw point clouds are represented in the form of txt files. Each txt file stands for the full point cloud of a 3D plant. Each row of the txt represents a point in the point cloud. Each txt file contains 5 columns, in which the first three columns give the "xyz" coordinate in centimeter, the fourth column gives the instance label, and the fifth column shows the semantic label.<br><br>
The value of the semantic label starts from "0" to "9". "0" means "tobacco stem system", "1" means "tobacco leaves"; "2" means "tomato stem system", "3" means "tomato leaves"; "4" means "sorghum stem system", "5" means "sorghum leaves"; "6" means "soybean stem system", "7" means "soybean leaves"; "8" means "pepper stem system", "9" means "pepper leaves".<br><br>
The value of the instance label represents the index of that organ instance (in most cases means a leaf instance). For example, "1" represents the 1st leaf in the current plant point cloud, and "18" represents the 18th leaf in the current plant point cloud. Every point in the stem system (regardless of species) is assigned an instance label of 0 because how to further separate the whole stem system into meaningful segments is still controversial in the field.<br>
## Data preprocessing<br>
The relevant files for preprocessing the dataset are stored in the datasets/preprocessing folder. The preprocessing steps are as follows: <br>
* Normalize each coordinate axis of the point clouds in the training set (or testing set) by scaling it to the interval [0,30].
  ```
  cd datasets/preprocessing/data_prepare
  python scale_pointcloud.py --input /path/to/dataset/Area_1(Area_2) --output /path/to/scaled dataset/Area_1(Area_2) --range 30
  ```
* Perform 10-times (10x) data augmentation on the point clouds in the training set. First, record the center point of the crop point cloud as the first center point. Second, use the Farthest Point Sampling (FPS) to produce another 8 center points from the plant. Third, use the above 9 center points to form 9 cubic area with a side length of 20 (this number should not be larger than the longest length of the plant bounding box, usually set as the 2/3 longest length), respectively. The original crop point cloud as well as the 9 formed cube areas are used as point clouds in the augmented training set. Because 1 point cloud turns into 10 point clouds, we call it 10x augmentation.
  ```
  python traindata_agument.py --input /path/to/scaled dataset/Area_1 --output /path/to/augmented dataset/Area_1
  ```
* Extract the edge points in the augmented training set (or testing set) point cloud and add label "1" to the corresponding edge points, and add label "0" to the inner points. This step is for better distinguishing edge points and non-edge points during subsequent [3DEPS](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000119) sampling.
  ```
  python get_edge.py --input /path/to/augmented dataset/Area_1(Area_2) --output /path/to/labeled dataset/Area_1(Area_2)
  ```
* Convert the point cloud files processed by the above steps into npy files with a file size of (N,7), where N is the number of point cloud points, the first three columns are xyz information, the fourth column is the segment id, all initialized to 1 (not used during training and testing), the fifth column is the binary label to determine whether it is an edge point, the sixth column is the semantic label, and the seventh column is the instance label.
  ```
  cd ..\..\..
  python -m datasets.preprocessing.plant_preprocessing preprocess --data_dir="/path/to/labeled dataset" --save_dir="./data/processed/5plant"
  ```
## Calculate leaf phenotypic traits<br>
The prediction (segmentation) results of Organ3DNet can be used to calculate individual leaf phenotypic traits, such as leaf area, leaf length, leaf width, and leaf inclination angle. The computational code resides in the /leaf_phenotyping directory:
* Restore the segmented leaf instances from the inference results to the original 3D space via inverse normalization.
  ```
  python 01_Inverse_normalization.py --original-dir "/origin" --normalized-dir "/pred" --output-dir "/unscale"
  ```
* Precisely align each predicted single leaf with GT. For each GT leaf region, we compute its IOU with all predicted leaf instances, and establish a correspondence based on the highest one.
  ```
  python 02_Align_leaves.py --true-dir "/origin" --pred-dir "/unscale" --output-dir "/align results"
  ```
* Calculate the phenotypic traits (e.g., leaf area, leaf length, leaf width, and leaf inclination angle) for the paired leaf regions (predicted and GT).
  ```
  python 03_Calculate_the_phenotypic_traits.py --search 200 --importance 3 --folder "/origin(align results)" --dest "/OUTPUT"
  ```
## Quick Start<br>
After the processed data set is stored in the directory ./data/processed/5plant, training and testing can be performed.
* Training
  ```
  conda activate Organ3DNet
  sh train.sh
  ```
* Testing<br>
  Set general.checkpoint in the test.sh file to the trained model path
  ```
  conda activate Organ3DNet
  sh test.sh
  ```
