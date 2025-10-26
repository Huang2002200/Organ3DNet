# Organ3DNet
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
The dataset (file 5plant_xyz_ins_sem.rar)((https://doi.org/10.5281/zenodo.17165870) contains 889 colorless point clouds of five crops (tobacco, tomato, sorghum, soybean, pepper), including 635 training point clouds (folder Area_1) and 254 test point clouds (folder Area_2). Tobacco has 105 point clouds, tomato has 326 point clouds, sorghum has 129 point clouds, soybean has 92 point clouds, and pepper has 237 point clouds. <br><br>
All raw point clouds are represented in the form of txt files. Each txt file represents a 3D plant. Each line of the txt file represents a point in the point cloud. Each txt file contains 5 columns, of which the first three columns show the "xyz" spatial information, the coordinate unit is centimeter, the fourth column is the instance label, and the fifth column is the semantic label.<br><br>
The value of the semantic label starts from "0" to "9". "0" means "tobacco stem system", "1" means "tobacco leaf"; "2" means "tomato stem system", "3" means "tomato leaf"; "4" means "sorghum stem system", "5" means "sorghum leaf"; "6" means "soybean stem system", "7" means "soybean leaf"; "8" means "pepper stem system", "9" means "pepper leaf".<br><br>
The value of the instance label in most cases represents the label of each leaf organ instance. For example, "1" represents the 1st leaf in the current point cloud, and "18" represents the 18th leaf in the current point cloud.Every point in the stem system (regardless of species) is assigned an instance label of 0.<br>
## Data_preprocessing<br>
The relevant files for preprocessing the dataset are stored in the datasets/preprocessing folder. The preprocessing steps are as follows: <br>
* Normalize the coordinates of the training set (test set) point cloud and scale it to the interval [0,30].
  ```
  cd datasets/preprocessing/data_prepare
  python scale_pointcloud.py --input /path/to/dataset/Area_1(Area_2) --output /path/to/scaled dataset/Area_1(Area_2) --range 30
  ```
* Perform 10x data enhancement on the training set point cloud. First, record its center point, then use the farthest point sampling (FPS) to sample 8 points. Use these 9 points as the center point to form a cube area with a side length of 20. The entire crop point cloud and the crop part contained in these 9 cubes are used as the augmented training dataset.
  ```
  python traindata_agument.py --input /path/to/scaled dataset/Area_1 --output /path/to/augmented dataset/Area_1
  ```
* Extract the edge points in the augmented training set (test set) point cloud and add the label 1 to the corresponding points, and add the label 0 to the remaining points. This step is to easily distinguish edge points and non-edge points during subsequent 3DEPS sampling.
  ```
  python get_edge.py --input /path/to/augmented dataset/Area_1(Area_2) --output /path/to/labeled dataset/Area_1(Area_2)
  ```
* Convert the point cloud files processed by the above steps into npy files with a file size of (N,7), where N is the number of point cloud points, the first three columns are xyz information, the fourth column is the segment id, all initialized to 1 (not used during training and testing), the fifth column is the binary label to determine whether it is an edge point, the sixth column is the semantic label, and the seventh column is the instance label.
  ```
  cd ..\..\..
  python -m datasets.preprocessing.plant_preprocessing preprocess --data_dir="/path/to/labeled dataset" --save_dir="./data/processed/5plant"
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
