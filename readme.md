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

（create_data.sh这一步主要是生成.pkl文件的，在data/infos文件夹下，已经生成了，不需要再跑一遍了~）

### Generate anchors by K-means

Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.

```
sh scripts/kmeans.sh
```

（kmeans.sh这一步也已经生成好了，在data/kmeans文件夹下）

### Download pre-trained weights

Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).

```
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

（这个是感知部分用到的resnet50的预训练权重，已经下载好了，在ckpt文件夹下）

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



## 目前在做的

原本的sparsedrive是先训练感知，作为第一阶段，之后感知预测规划几个模块一起训练，作为第二阶段，在nuscenes数据集官方划分的训练集和验证集上训练和测试。

但是因为现在的目标是想让模型在预训练之后在新的环境中也能通过OEL的方式自适应，所以目前是把数据集分成了新加坡和波士顿两个区域（nuscenes数据集中的数据只在新加坡和波士顿两个地方采集），先在新加坡上面做的预训练，预训练把上述两个阶段都训练了，训练完的权重在ckpt/nuscenes_boston_sg_split_v2文件夹；然后再在波士顿场景下做正式训练，这个时候只训练第二阶段，在新加坡场景下预训练得到的权重ckpt/nuscenes_boston_sg_split_v2/sparsedrive_stage2.pth的基础上继续训练，因为第二阶段是三个模块一起训练的，所以可以协同起来得到一个运动补偿损失。

使用的方法是比较预测的其他车辆的下一帧位置和实际在下一帧感知到的其他车辆的位置之间的差距，作为损失，来更新模型。

代码实现在projects/mmdet3d_plugin文件夹下，相比sparsedrive原代码，主要修改了projects/mmdet3d_plugin/models/sparsedrive_head.py（把原本的感知损失去掉了），projects/mmdet3d_plugin/models/sparsedrive.py（核心算法实现，主要在get_evo_loss函数中），projects/mmdet3d_plugin/models/motion/motion_planning_head.py（把原本的预测损失去掉了）三个文件。



## 关于一些文件的说明

##### ckpt

存放预训练模型权重。

nuscenes_boston_sg_split_v2是在新加坡做两阶段预训练后的权重；

nuscenes_boston_sg_split_v3是在波士顿做两阶段预训练后的权重；

##### data

###### infos

custom_split_boston：是对nuscenes数据集中波士顿场景继续按照8:2划分了训练集和验证集，通过create_data.sh得到的.pkl文件；

custom_split_v2：是对nuscenes数据集按照新加坡和波士顿划分，其中新加坡为训练集，波士顿为验证集，通过create_data.sh得到的.pkl文件；

custom_split_v3：是对nuscenes数据集按照新加坡和波士顿划分，其中波士顿为训练集，新加坡为验证集，通过create_data.sh得到的.pkl文件；

nuscenes_infos_test/train/val.pkl：nuscenes原始划分的训练集验证集和测试集，通过create_data.sh得到的.pkl文件。

###### kmeans

和infos文件夹是对应的，通过kmeans.sh得到的文件。

##### projects

###### configs 配置文件

ori: 原代码中的配置；

sg_boston_v2：新加坡作为训练集，波士顿作为验证集时的配置；

sg_boston_v3：波士顿作为训练集，新加坡作为验证集时的配置；

sparsedrive_small_stage1.py：这个可以不用管；

sparsedrive_small_stage2.py：目前的对波士顿场景按照8:2划分为训练集和验证集的配置文件。

###### mmdet3d_plugin

核心代码实现部分，主要修改的是models下面的sparsedrive.py、sparsedrive_head.py和motion_planning_head.py三个文件。

##### scripts

直接运行的bash脚本。

##### tools

scripts中的bash脚本调用该目录下的对应程序执行。

##### splits.py

用于划分数据集，在训练和验证的时候需要用这个文件替换原有的nuscenes库下面的splits.py文件，大约在下面这这个路径下：miniconda3/envs/sparsedrive/lib/python3.8/site-packages/nuscenes/utils/splits.py。