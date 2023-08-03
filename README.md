# Models which supports Occ3D dataset

Currently, the Occ3D dataset supports the following models:

 - CTF-Occ (Ours)
 - [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
 - [TPVFormer](https://github.com/wzzheng/TPVFormer)

## Installation instructions

Following https://mmdetection3d.readthedocs.io/en/v0.17.1/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n occ3d python=3.8 -y
conda activate occ3d
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Recommended torch==1.10
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.22.0 # Other versions may cause problem.
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
**g. Clone Occ3D.**
```
git clone https://github.com/Tsinghua-MARS-Lab/Occ3D
git checkout code
```

**h. Prepare pretrained models.**
```shell
cd Occ3D
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

## Preparing Dataset

# eval
## single machine
```sh
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ_waymo.py work_dirs/bevformer_base_occ_waymo/latest.pth 8 --eval mIoU
```
## multi machine
```sh
GPUS=8 ./tools/slurm_test.sh brie1 test projects/configs/bevformer/bevformer_base_occ_waymo.py work_dirs/bevformer_base_occ_waymo/latest.pth --eval mIoU
```

# save results
```sh
./tools/dist_test.sh projects/configs/bevformer/bevformer_base_occ_waymo.py work_dirs/bevformer_base_occ_waymo/latest.pth 8 --out work_dirs/bevformer_base_occ_waymo/results.pkl
```

# train
## single GPU
```sh
./tools/dist_train.sh projects/configs/bevformer/bevformer_base_occ_waymo.py 1
```
## single machine
```sh
./tools/dist_train.sh projects/configs/bevformer/bevformer_base_occ_waymo.py 8
```
## multi machine
```sh
NNODES=4 NODE_RANK=0 PORT=2850 MASTER_ADDR=10.200.5.154 ./tools/my_dist_train.sh projects/configs/bevformer/bevformer_base_occ_conv3d_waymo.py 8
```
or
```sh
GPUS=32 ./tools/slurm_train.sh brie1 test projects/configs/bevformer/bevformer_base_occ_conv3d_waymo.py
```

# vis
```sh
python vis_preds.py
```

# custom eval
```sh
python tools/eval_waymo.py # waymo
or 
python tools/eval.py # nuscene
```