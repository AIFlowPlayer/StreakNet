
<div align="center"><img src="./assets/streaknet_logo.png" width="400"></div><br>
<div align="center"><img src="./assets/overview.jpg"></div>

## Introduction

In this paper, we introduce StreakNet-Arch, a novel signal processing architecture designed for Underwater Carrier LiDAR-Radar (UCLR) imaging systems, to address the limitations in scatter suppression and real-time imaging. StreakNet-Arch formulates the signal processing as a real-time, end-to-end binary classification task, enabling real-time image acquisition. To achieve this, we leverage Self-Attention networks and propose a novel Double Branch Cross Attention (DBC-Attention) mechanism that surpasses the performance of traditional methods. Furthermore, we present a method for embedding streak-tube camera images into attention networks, effectively acting as a learned bandpass filter. To facilitate further research, we contribute a publicly available streak-tube camera image dataset. The dataset contains 2,695,168 real-world underwater 3D point cloud data. These advancements significantly improve UCLR capabilities, enhancing its performance and applicability in underwater imaging tasks.

For further details, please refer to our [paper](https://arxiv.org/abs/2404.09158).

## Dataset
<details>
<summary>Introduction</summary>

**StreakNet-Dataset** is an underwater laser imaging dataset for **UCLR** systems. It comprises a collection of streak-tube images captured by a **UCLR** system at distances of 10m, 13m, 15m, and 20m. See the table below to learn more details of the dataset.

|Distance|Number of streak-tube images|Resolution of streak-tube images|Data type|Training set|Validation set|Test set|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|10m|400|2048x2048|uint16|315,200|40,800|819,200|
|13m|349|2048x2048|uint16|281,992|47,530|714,752|
|15m|300|2048x2048|uint16|245,400|39,200|614,400|
|20m|267|2048x2048|uint16|229,086|31,240|546,816|

</details>

<details>
<summary id="datasetdownload">Download</summary>

You can download **StreakNet-Dataset** for free from [HuggingFace](https://huggingface.co/datasets/Coder-AN/StreakNet-Dataset) or [ModelScope](https://modelscope.cn/datasets/CoderAN/StreakNet-Dataset/) by Git.

Firstly, install `git-lfs`.

```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update
sudo apt install git-lfs   
sudo git lfs install  --system
```

Then, download **StreakNet-Dataset** in work directory of StreakNet.

* From [HuggingFace](https://huggingface.co/datasets/Coder-AN/StreakNet-Dataset): For Global Users

```sh
cd StreakNet
git clone https://huggingface.co/datasets/Coder-AN/StreakNet-Dataset ./datasets
```

* From [ModelScope](https://modelscope.cn/datasets/CoderAN/StreakNet-Dataset): For Chinese Users

```sh
cd StreakNet
git clone https://www.modelscope.cn/datasets/CoderAN/StreakNet-Dataset.git ./datasets
```

</details>

<details>
<summary>Organizational Structure</summary>

After downloading **StreakNet-Dataset** from [HuggingFace](https://huggingface.co/datasets/Coder-AN/StreakNet-Dataset) or [ModelScope](https://modelscope.cn/datasets/CoderAN/StreakNet-Dataset/), you will see the following directory structure.

```sh
datasets
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
conda create -n streaknet python=3.10
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

* Step2. Download the [**StreakNet-Dataset**](#dataset) by following the ['*Download*'](#datasetdownload) section, then you will see the following directory structure.

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
<summary id="trainmodels">Train Models</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakData**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Run the following commands to train the respective models in the root directory.
```sh
python tools/train_streaknet.py -b 512 -f exps/streaknet/streaknet_s.py --cache
                                                         streaknet_m.py
                                                         streaknet_l.py
                                                         streaknet_x.py
```

```sh
python tools/train_streaknet.py -b 512 -f exps/streaknetv2/streaknetv2_s.py --cache
                                                           streaknetv2_m.py
                                                           streaknetv2_l.py
                                                           streaknetv2_x.py
```
> Arguments: \
> **-b**: set the batch-size when training. \
> **-f**: specify the experiment profile. \
> **--cache**: use RAM cache when training

**Attention**: 

(1) When you enable the `--cache` option, the program will preload the dataset into the RAM to accelerate the training process. Please ensure that your server has at least **25GB** of free RAM space to use this option. If your RAM space is insufficient, please disable the `--cache` option. In that case, the program will load data directly from the disk when needed. However, this approach often results in 10 times longer training times.

(2) The program will utilize CUDA to accelerate the training process. Please ensure that your server is equipped with at least one NVIDIA GPU with a graphics memory capacity of more than **2GB**.

```sh
python tools/train.py -b 512 -f exps/streaknet/streaknet_s.py
                                               streaknet_m.py
                                               streaknet_l.py
                                               streaknet_x.py
```

```sh
python tools/train.py -b 512 -f exps/streaknetv2/streaknetv2_s.py
                                                 streaknetv2_m.py
                                                 streaknetv2_l.py
                                                 streaknetv2_x.py
```

* Step4. Real-time training status will be saved to *StreakNet_outputs* folder. Run *tensorboard* to visualize the status of the training process.

```sh
tensorboard --logdir=StreakNet_outputs
```

</details>

<details>
<summary>Demo</summary>

* Step1. Download a pretrained model from [HuggingFace](https://huggingface.co/Coder-AN/StreakNet-Models) or [ModelScope](https://modelscope.cn/models/CoderAN/StreakNet-Models/summary). Alternatively, you can directly use the model you just trained in the ['*Train Models*'](#trainmodels) section.

```sh
# From HuggingFace: For Global Users
cd StreakNet
git clone https://huggingface.co/Coder-AN/StreakNet-Models ./checkpoints
```

```sh
# From ModelScope: For Chinese Users
cd StreakNet
git clone https://www.modelscope.cn/CoderAN/StreakNet-Models.git ./checkpoints
```

* Step2. Run the following command to run StreakNet demo:

```sh
python tools/demo_streaknet.py -b 2 \
  --path datasets/clean_water_13m \
  -f exps/streaknet/streaknet_s.py \
  -c checkpoints/streaknet_s_ckpt.pth \
  --device "cuda:0" \
  --cache --real-time
```

> Arguments: \
> **--path**: path to the dataset. \
> **-f**: specify the experiment profile. \
> **-b**: set the batch-size when inferring. \
> **-c**: specify the model weights when inferring. \
> **--device**: specify the GPU when inferring. \
> **--realtime**: enable real-time preview. \
> **--save**: save imaging results.

**Attention**: If you omit the `-c` option, the program will automatically use the '*best_ckpt.pth*' file located in the '*StreakNet_outputs*' directory, which you just trained in the ['*Train Models*'](#trainmodels) section.

```sh
python tools/demo_streaknet.py -b 2 \
  --path datasets/clean_water_13m \
  -f exps/streaknet/streaknet_s.py \
  --device "cuda:0" \
  --save
```

* Step3. Run the following command to run traditional bandpass-filter demo:

```sh
python tools/demo_bandpass.py -b 2 --path datasets/clean_water_13m --device "cuda:0" --cache
```

> Arguments: \
> **--path**: path to the dataset. \
> **-b**: set the batch-size when inferring. \
> **--device**: specify the GPU when inferring. \
> **--save**: save imaging results.

* Step4. Use FDEL as an equivalent bandpass filter:

```sh
python tools/demo_bandpass.py -b 2 \
  --path datasets/clean_water_13m \
  -f exps/streaknet/streaknet_s.py \
  -c checkpoints/streaknet_s_ckpt.pth \
  --device "cuda:0" --cache
```

> Arguments: \
> **--path**: path to the dataset. \
> **-f**: specify the experiment profile. \
> **-b**: set the batch-size when inferring. \
> **-c**: specify the model weights when inferring. \
> **--device**: specify the GPU when inferring. \
> **--save**: save imaging results.

</details>

<details>
<summary>Evaluation</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakNet-Dataset**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Train models by following the ['*Train Models*'](#trainmodels) section.

* Step4. Evaluate StreakNet:

```sh
python tools/valid_streaknet.py -b 2 \
  -f exps/streaknet/streaknet_s.py \
  -c checkpoints/streaknet_s_ckpt.pth \
  -d "cuda:0" --cache
```

> Arguments: \
> **-f**: specify the experiment profile. \
> **-b**: set the batch-size when inferring. \
> **-c**: specify the model weights when inferring. \
> **-d**: specify the GPU when inferring. \
> **--save**: save imaging results.

* Step5. Evaluate traditional bandpass filter algorithm:

```sh
python tools/valid_bandpass.py -b 2 -d "cuda:0" --cache
```

> Arguments: \
> **-b**: set the batch-size when inferring. \
> **--device**: specify the GPU when inferring. \
> **--save**: save imaging results.

* Step 6. Evaluate the equivalent bandpass filter:

```sh
python tools/valid_bandpass.py -b 2 \
  -f exps/streaknet/streaknet_s.py \
  -c checkpoints/streaknet_s_ckpt.pth \
  -d "cuda:0" --cache
```

> Arguments: \
> **-f**: specify the experiment profile. \
> **-b**: set the batch-size when inferring. \
> **-c**: specify the model weights when inferring. \
> **-d**: specify the GPU when inferring. \
> **--save**: save imaging results.

</details>

<details>
<summary>Test speed benchmark</summary>

* Step1. Install the StreakNet module by following the ['*Installation*'](#quickstartinstallation) section.

* Step2. Prepare the [**StreakNet-Dataset**](#dataset) dataset by following the ['*Prepare Dataset*'](#preparedataset) setction.

* Step3. Test AIT of StreakNets.

```sh
python tools/benchmark_streaknet.py -f exps/streaknet/streaknet_s.py -d "cuda:0" --save
```

* Step 4. Test AIT of traditional bandpass filter algorithm.

```sh
python tools/benchmark_bandpass.py -d "cuda:0" --save
```

</details>

<!-- ## Deployment

1. [ONNX export and an ONNXRuntime](./demo/ONNXRuntime/)
2. [TensorRT in C++ and Python](./demo/TensorRT/) -->

## Cite StreakNet
If you use StreakNet in your research, please cite our work by using the following BibTeX entry:

```latex
@article{streaknet2024,
  title={xxx},
  author={xxx},
  journal={xxx},
  year={2024}
}
```

## Respect to Predecessors
* During the development of this open-source project, we drew inspiration from the excellent engineering architecture of the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) project by [Megvii](https://www.megvii.com/) Technology.  The YOLOX project was led by [Dr. Jian Sun](https://baike.baidu.com/item/%E5%AD%99%E5%89%91/19814032) (1976.10-2022.6.14), a respected scientist, who made significant contributions to the advancement of computer vision.🕯️🕯️🕯️
* We were deeply saddened to hear the news of the passing of [Prof. Xiaoou Tang](https://baike.baidu.com/item/%E6%B1%A4%E6%99%93%E9%B8%A5/7200225) (1968.1-2023.12.15) on December 16, 2023, shortly after completing all the preliminary experiments for this project.  Prof. Tang devoted his entire life to computer science research and made outstanding contributions to the advancement of computer vision and artificial intelligence. We express our utmost respect to Prof. Tang.🕯️🕯️🕯️

## Copyright

<br>
<div align="center"><img src="./assets/iopen.jpg" width="500"></div>
<div align="center"><p>Copyright &copy; School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. <br>All rights reserved.</p></div>

