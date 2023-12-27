[English](./README.md) | [中文](./README_CN.md) | [Gitee码云](#)

<hr>
<div align="right"><img src="./assets/iopen.jpg" width="300"></div><br>


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
<summary id="preparedataset">Prepare Dataset</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Create a directory named '*datasets*' under the root directory.

```sh
cd StreakNet
mkdir datasets
```

* Step3. Download the [**StreakData**](#dataset) dataset from [GoogleDrive](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link) or [BaiduDisk](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76), unzip it to the '*datasets*' directory. Specifically, your project directory should appear as follows:

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

</details>

<details>
<summary id="reproduceexperimentalresults">Reproduce Experimental Results</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakData**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Run the following commands to train the respective models in the root directory.
```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py --cache
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```
> Arguments: \
> **-b**: set the batch-size when training. \
> **-d**: set the number of GPU when training (Currently, only d=1 is supported). \
> **-f**: specify the experiment profile. \
> **--cache**: use RAM cache when training

**Attention**: 

(1) When you enable the --cache option, the program will preload the dataset into the RAM to accelerate the training process. Please ensure that your server has at least **25GB** of free RAM space to use this option. If your RAM space is insufficient, please disable the --cache option. In that case, the program will load data directly from the disk when needed. However, this approach often results in 10 times longer training times.

(2) The program will utilize CUDA to accelerate the training process. Please ensure that your server is equipped with at least one NVIDIA GPU with a graphics memory capacity of more than **2GB**.

```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```

* Step4. Real-time training status will be saved to *StreakNet_outputs* folder. Run *tensorboard* to visualize the status of the training process.

```sh
tensorboard --logdir=StreakNet_outputs
```

</details>

<details>
<summary>Demo</summary>

* Step1. Download a pretrained model from the [benchmark](#benchmark) table. Alternatively, you can directly use the model you just trained in the ['*Reproduce Experimental Results*'](#reproduceexperimentalresults) section.

* Step2. Run the following command to start demo:

```sh
python tools/demo.py --path datasets/clean_water_10m/data -f exps/streaknet/streaknet_s.py -b 512 -c <path/to/your/pretrained/model/streaknet_s_ckpt.pth>
                                     clean_water_13m                        streaknet_m.py                                          streaknet_m_ckpt.pth
                                     clean_water_15m                        streaknet_l.py                                          streaknet_l_ckpt.pth
                                     clean_water_20m                        streaknet_x.py                                          streaknet_x_ckpt.pth
```

> Arguments: \
> **--path**: path to the streak images (.tif). \
> **-f**: specify the experiment profile. \
> **-b**: set the batch-size when inferring. \
> **-c**: specify the model weights when inferring.

**Attention**: If you omit the -c option, the program will automatically use the '*best_ckpt.pth*' file located in the '*StreakNet_outputs*' directory, which you just trained in the ['*Reproduce Experimental Results*'](#reproduceexperimentalresults) section.

```sh
python tools/demo.py --path datasets/clean_water_13m/data -f exps/streaknet/streaknet_s.py -b 512
                                     clean_water_13m                        streaknet_m.py
                                     clean_water_15m                        streaknet_l.py
                                     clean_water_20m                        streaknet_x.py
```

</details>

<details>
<summary>Evaluation</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakData**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Train models by following the ['*Reproduce Experimental Results*'](#reproduceexperimentalresults) section.

* Step4. Evaluation.

```sh
python tools/valid.py -d 1 -b 512 -f exps/streaknet/streaknet_s.py --cache
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```

> Arguments: \
> **-b**: set the batch-size when training. \
> **-d**: set the number of GPU when training (Currently, only d=1 is supported). \
> **-f**: specify the experiment profile. \
> **--cache**: use RAM cache when training

</details>

<details>
<summary>Traditional Signal Processing Method</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakData**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Run traditional signal processing method.

```sh
python scripts/traditional_gpu_process.py
```

* The results will save to '*StreakNet_outputs/traditional*'.

</details>

## Deployment

1. [ONNX export and an ONNXRuntime](#)
2. [TensorRT in C++ and Python](#)

## Cite StreakNet
If you use StreakNet in your research, please cite out work by using the following BibTeX entry:

```latex
@article{streaknet2024,
  title={xxx},
  author={xxx},
  journal={xxx},
  year={2024}
}
```

## Respect to Predecessors
* During the development of this open-source project, we drew inspiration from the excellent engineering architecture of the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) project by [Megvii](https://www.megvii.com/) Technology.  The project was led by [Dr. Jian Sun](https://baike.baidu.com/item/%E5%AD%99%E5%89%91/19814032) (1976.10-2022.6.14), a respected scientist, who made significant contributions to the advancement of computer vision.🕯️🕯️🕯️
* We were deeply saddened to hear the news of the passing of [Prof. Xiaoou Tang](https://baike.baidu.com/item/%E6%B1%A4%E6%99%93%E9%B8%A5/7200225) (1968.1-2023.12.15) on December 16, 2023, shortly after completing all the preliminary experiments for this project.  Prof. Tang devoted his entire life to computer science research and made outstanding contributions to the advancement of computer vision and artificial intelligence. We express our utmost respect to Prof. Tang.🕯️🕯️🕯️

## Copyright

> Developer: Hongjun An \
> Supervisor: [Prof. Xuelong Li](https://iopen.nwpu.edu.cn/info/1329/1171.htm), [Assoc. Prof. Zhe Sun](https://iopen.nwpu.edu.cn/info/1251/2076.htm)

<br>
<div align="center"><img src="./assets/iopen.jpg" width="500"></div>
<div align="center"><p>Copyright &copy; <a href="https://iopen.nwpu.edu.cn/">School of Artificial Intelligence, OPtics and ElectroNics(iOPEN)</a>, <a href="https://www.nwpu.edu.cn/index.html">Northwestern PolyTechnical University</a>. All righs reserved.</p></div>

