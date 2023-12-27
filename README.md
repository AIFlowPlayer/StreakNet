[English](./README.md) | [中文](./README_CN.md) | [Gitee码云](#)

<hr>
<div align="right"><img src="./assets/iopen.jpg" width="250"></div><br>


<div align="center"><img src="./assets/streaknet_logo.png" width="400"></div><br>
<div align="center"><img src="./assets/demo.png"></div>

## Introduction

[StreakNet](https://github.com/BestAnHongjun/StreakNet) is a Deep-Learning (DL) based neural network for underwater target detection in **Underwater Streak Camera LiDAR (USCL)** systems. It is capable of detecting underwater objects at a **millimeter-level** accuracy up to a distance of **20m**. For further details, please refer to our [paper](#).

<div align="center"><img src="./assets/streaknet_architecture.png"></div>

## Benchmark

StreakNet Benchmark

|Model|F1-Score (%)|PSNR (dB)|Speed V100 (ms/pixel)|Speed NX (ms/pixel)|Params(M)|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[StreakNet-s](./exps/streaknet/streaknet_s.py)|85.46|8.75|00.0|00.00|1.25|[BaiduDisk](#)|
|[StreakNet-m](./exps/streaknet/streaknet_m.py)|85.87|8.91|00.0|00.00|3.15|[BaiduDisk](#)|
|[StreakNet-l](./exps/streaknet/streaknet_l.py)|85.90|8.92|00.0|00.00|10.51|[BaiduDisk](#)|
|[StreakNet-x](./exps/streaknet/streaknet_x.py)|86.23|9.00|00.0|00.00|50.40|[BaiduDisk](#)|
|(baseline)|41.07|4.64|00.0|00.00|---|---|

## Dataset
<details>
<summary>Introduction</summary>

**StreakData** is an underwater imaging dataset for **USCL** systems. It comprises a collection of streak images captured by a **USCL** system at distances of 10m, 13m, 15m, and 20m. See the table below to learn more details of the dataset.

|Distance|Number of streak images|Resolution of streak images|Resolution of imaged image|Data type|Sample size|
|:---:|:---:|:---:|:---:|:---:|:---:|
|10m|400|2048x2048|2048x400|uint16|819200|
|13m|349|2048x2048|2048x349|uint16|714752|
|15m|300|2048x2048|2048x300|uint16|614400|
|20m|267|2048x2048|2048x267|uint16|546816|

You can download **StreakData** for free at [GoogleDrive](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link) or [BaiduDisk](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76).
</details>

<details>
<summary>Organizational Structure</summary>

After downloading **StreakData** from [GoogleDrive](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link) or [BaiduDisk](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76), please unzip the file and you will see the following directory structure.
```sh
YOUR_UNZIP_DIRECTORY
    |- clean_water_10m      # The directory of data taken at a distance of 10m
    |   |- data             # Original streak images
    |   |   |- 001.tif
    |   |   |- 002.tif
    |   |   |- 003.tif
    |   |   |- ...
    |   |
    |   |- groundtruth.npy  # The ground-truth of the final imaged image
    |   |- preview.jpg      # A preview of the ground-truth
    |
    |- clean_water_13m      # The directory of data taken at a distance of 13m (has the same structure as 10m)
    |- clean_water_15m      # The directory of data taken at a distance of 15m (has the same structure as 10m)
    |- clean_water_20m      # The directory of data taken at a distance of 20m (has the same structure as 10m)
    |- template.npy         # The 1-D time sequence of the template signal
    |- test_config.yaml     # The config file of test-set
    |- train_config.yaml    # The config file of training-set
    |- valid_config.yaml    # The config file of validation-set
```

</details>

## Quick Start
<details>
<summary id="quickstartinstallation">Installation</summary>

* Step1. Setup your conda environment. ([What is Anaconda?](https://www.anaconda.com/download))
```sh
conda create -n streaknet python=3.7
conda activate streaknet
```

* Step2. Install StreakNet from source.
```sh
git clone https://github.com/BestAnHongjun/StreakNet.git
cd StreakNet
pip install -e .
```
</details>

<details>
<summary>Reproduce Experimental Results</summary>

* Step1. Install the StreakNet module by following the [*Installation*](#quickstartinstallation) section.

* Step2. Download the [**StreakData**](#dataset) dataset from [GoogleDrive](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link) or [BaiduDisk](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76), unzip it to *datasets* directory under the root directory of the project. Specifically, your project directory should look like this:

```sh
StreakNet
    |- datasets
    |   |- clean_water_10m
    |   |- clean_water_13m
    |   |- clean_water_15m
    |   |- ...
    |
    |- assets
    |- exps
    |- scripts
    |- streaknet
    |- ...
```

* Step3. *cd* to the root directory of the project.
```sh
cd StreakNet
```

* Step4. Run the following commands to train the respective models.
```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py --cache
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_m.py --cache
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_l.py --cache
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_x.py --cache
```
> Arguments:
> **-b**: set the batch-size when training.
> **-d**: set the number of GPU when training (Currently, only d=1 is supported).
> **-f**: specify the experiment profile.
> **--cache**: use RAM cache when training

**Attention**: 

(1) When you enable the --cache option, the program will preload the dataset into the RAM to accelerate the training process. Please ensure that your server has at least **25GB** of free RAM space to use this option. If your RAM space is insufficient, please disable the --cache option. In that case, the program will load data directly from the disk when needed. However, this approach often results in 10 times longer training times.

(2) The program will utilize CUDA to accelerate the training process. Please ensure that your server is equipped with at least one NVIDIA GPU with a graphics memory capacity of more than **2GB**.

```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_m.py
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_l.py
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_x.py
```

* Step5. Real-time training status will be saved to *StreakNet_outputs* folder. Run *tensorboard* to visualize the status of the training process.

```sh
tensorboard --logdir=StreakNet_outputs
```

</details>