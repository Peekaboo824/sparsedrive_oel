## 关于环境配置

参考https://github.com/swc-17/SparseDrive/blob/main/docs/quick_start.md

### Set up a new virtual environment

```
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages

```
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op

```
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data

Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.

```
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.

```
sh scripts/create_data.sh
```

### Generate anchors by K-means

Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.

```
sh scripts/kmeans.sh
```

### Download pre-trained weights

Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).

```
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing

```
# train
sh scripts/train.sh

# test
sh scripts/test.sh
```

### Visualization

```
sh scripts/visualize.sh
```
