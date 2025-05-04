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
## DataSet<br>
The dataset (file 5plant_xyz_ins_sem.rar) contains 889 colorless point clouds of five crops (tobacco, tomato, sorghum, soybean, pepper), including 635 training point clouds (folder Area_1) and 254 test point clouds (folder Area_2). Tobacco has 105 point clouds, tomato has 326 point clouds, sorghum has 129 point clouds, soybean has 92 point clouds, and pepper has 237 point clouds. <br><br>
All raw point clouds are represented in the form of txt files. Each txt file represents a 3D plant. Each line of the txt file represents a point in the point cloud. Each txt file contains 5 columns, of which the first three columns show the "xyz" spatial information, the coordinate unit is centimeter, the fourth column is the instance label, and the fifth column is the semantic label.<br><br>
The value of the semantic label starts from "0" to "9". "0" means "tobacco stem system", "1" means "tobacco leaf"; "2" means "tomato stem system", "3" means "tomato leaf"; "4" means "sorghum stem system", "5" means "sorghum leaf"; "6" means "soybean stem system", "7" means "soybean leaf"; "8" means "pepper stem system", "9" means "pepper leaf".<br><br>
The value of the instance label in most cases represents the label of each leaf organ instance. For example, "1" represents the 1st leaf in the current point cloud, and "18" represents the 18th leaf in the current point cloud.Every point in the stem system (regardless of species) is assigned an instance label of 0.<br>
## Data_preprocessing<br>
