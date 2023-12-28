[English](./README.md) | [中文](./README_CN.md) | [Gitee码云](#)

<hr>


<div align="center"><img src="./assets/streaknet_logo.png" width="400"></div><br>
<div align="center"><img src="./assets/demo.png"></div>

## 项目介绍

[StreakNet](https://github.com/BestAnHongjun/StreakNet)是一种应用于**水下条纹相机激光雷达系统(Underwater Streak Camera LiDAR, USCL)**，进行水下目标探测的深度学习神经网络。它可以在距离目标**20m**时，以**毫米级**精度进行水下目标探测。了解更多细节，请阅读我们的[论文](#)。

<div align="center"><img src="./assets/streaknet_architecture.png"></div>

## 基准性能

StreakNet基准性能

|模型|F1评分(%)|峰值信噪比(dB)|V100推理速度(毫秒/像素)|NX推理速度(毫秒/像素)|参数量(M)|权重文件|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[StreakNet-s](./exps/streaknet/streaknet_s.py)|85.46|8.75|00.0|00.00|1.25|[谷歌云盘](https://drive.google.com/file/d/1938oCFHBFmlaTmLYb7Jan1TOAy26-ge1/view?usp=drive_link)<br>[百度网盘](https://pan.baidu.com/s/1tT_aXlzfttNEWIaRVBpSng?pwd=8rai)|
|[StreakNet-m](./exps/streaknet/streaknet_m.py)|85.87|8.91|00.0|00.00|3.15|[谷歌云盘](https://drive.google.com/file/d/17vpuiSYOK8m-qOtA4yQTwv-hofZT0o_W/view?usp=drive_link)<br>[百度网盘](https://pan.baidu.com/s/1A2144Y7f0MMEOlNDcbCRrg?pwd=wnxv)|
|[StreakNet-l](./exps/streaknet/streaknet_l.py)|85.90|8.92|00.0|00.00|10.51|[谷歌云盘](https://drive.google.com/file/d/146c8fSDOPtsmUDedHA7jN704a8btBqtR/view?usp=drive_link)<br>[百度网盘](https://pan.baidu.com/s/1K9oOgCrI-t5MF8RuC62RSA?pwd=et43)|
|[StreakNet-x](./exps/streaknet/streaknet_x.py)|86.23|9.00|00.0|00.00|50.40|[谷歌云盘](https://drive.google.com/file/d/1c7VP4C7pFSd-kgZXpLitLLRty-g_Mlws/view?usp=drive_link)<br>[百度网盘](https://pan.baidu.com/s/1nvHj4aWo4pXhP0LJB78Tkg?pwd=fl8o)|
|(基线)|42.56|4.91|00.0|00.00|---|---|

## 数据集
<details>
<summary>数据集简介</summary>

**StreakData**是用于水下成像实验的数据集，其包含由**USCL**系统在10m、13m、15m和20m距离下采集的一系列条纹图像。有关数据集详情请看以下表格：

|采集距离|条纹像数量|条纹像分辨率|最终成像分辨率|数据类型|样本数量|
|:---:|:---:|:---:|:---:|:---:|:---:|
|10m|400|2048x2048|2048x400|uint16|819200|
|13m|349|2048x2048|2048x349|uint16|714752|
|15m|300|2048x2048|2048x300|uint16|614400|
|20m|267|2048x2048|2048x267|uint16|546816|

你可以在[谷歌云盘](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link)或[百度网盘](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76)免费下载**StreakData**数据集。

</details>

<details>
<summary>数据集组织结构</summary>

由[谷歌云盘](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link)或[百度网盘](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76)下载数据集后，请解压文件，随后得到以下目录结构：

```sh
你的解压目录
    |- clean_water_10m      # 距离10m数据目录
    |   |- data             # 条纹相机拍摄的原数据
    |   |   |- 001.tif
    |   |   |- 002.tif
    |   |   |- 003.tif
    |   |   |- ...
    |   |
    |   |- groundtruth.npy  # 最终成像的人工标注数据
    |   |- preview.jpg      # 最终成像的预览图
    |
    |- clean_water_13m      # 距离13m数据目录（目录结构与10m一致）
    |- clean_water_15m      # 距离15m数据目录（目录结构与10m一致）
    |- clean_water_20m      # 距离20m数据目录（目录结构与10m一致）
    |- template.npy         # 模板信号一维时间序列文件
    |- test_config.yaml     # 测试集配置文件
    |- train_config.yaml    # 训练集配置文件
    |- valid_config.yaml    # 验证集配置文件
```

</details>

## 快速开始
<details>
<summary id="quickstartinstallation">安装StreakNet</summary>

* 第一步：初始化conda环境。 ([什么是Anaconda？](https://www.anaconda.com/download))
```sh
conda create -n streaknet python=3.7
conda activate streaknet
```

* 第二步：由源码安装StreakNet。
```sh
git clone https://github.com/BestAnHongjun/StreakNet.git
cd StreakNet
pip install -e .
```
</details>

<details>
<summary id="preparedataset">准备数据集</summary>

* 第一步：执行[“*安装StreakNet*”](#quickstartinstallation)部分步骤。

* 第二步：在工程根目录下创建一个名为“*datasets*”的目录。

```sh
cd StreakNet
mkdir datasets
```

* 第三步：由[谷歌云盘](https://drive.google.com/file/d/16RiV8JRL2GVe0GH1oXF4ZcrN2okQq6qG/view?usp=drive_link)或[百度网盘](https://pan.baidu.com/s/1QQ0nGwlq0KzwvY8yi2PCaw?pwd=zl76)下载[**StreakData**](#dataset)数据集后，解压至“*datasets*”目录。具体来说，你个工程目录最终应呈现如下结构：

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
<summary id="reproduceexperimentalresults">复现实验结果</summary>

* 第一步：执行[“*安装StreakNet*”](#quickstartinstallation)部分步骤。

* 第二步：执行[“*准备数据集*”](#preparedataset)部分步骤。

* 第三步：在工程根目录下，分别执行如下指令训练模型。

```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py --cache
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```
> 参数说明: \
> **-b**: 设置训练时的batch-size； \
> **-d**: 设置训练时使用的GPU数量（目前仅支持单卡训练）； \
> **-f**: 指定实验配置文件； \
> **--cache**: 训练时启用RAM缓存。

**注意**: 

（1）当你启用`--cache`选项时，程序会预加载数据集至RAM以加速训练过程。请确保你的服务器有至少**25G**的RAM空间（注意不是磁盘空间）。如果你的RAM空间不足，请禁用`--cache`选项，但这通常会消耗10倍以上的训练时间。

（2）在训练过程中，程序会调用CUDA以加速训练过程。请确保你的服务器插有至少1张显存**2GB**以上的英伟达显卡。

```sh
python tools/train.py -b 512 -d 1 -f exps/streaknet/streaknet_s.py
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```

* 第四步：实时训练日志会被保存到*StreakNet_outputs*文件夹，运行*tensorboard*可以可视化训练过程。

```sh
tensorboard --logdir=StreakNet_outputs
```

</details>

<details>
<summary>演示样例(Demo)</summary>

* 第一步：由[基准性能](#benchmark)表格下载一个预训练的模型。或者你可以使用你在['*复现实验结果*'](#reproduceexperimentalresults)部分训练得到的模型。

* 第二部：执行以下指令，启动演示样例：

```sh
python tools/demo.py --path datasets/clean_water_10m/data -f exps/streaknet/streaknet_s.py -b 512 -c <path/to/your/pretrained/model/streaknet_s_ckpt.pth>
                                     clean_water_13m                        streaknet_m.py                                          streaknet_m_ckpt.pth
                                     clean_water_15m                        streaknet_l.py                                          streaknet_l_ckpt.pth
                                     clean_water_20m                        streaknet_x.py                                          streaknet_x_ckpt.pth
```

> 参数说明: \
> **--path**: 条纹像(.tif)所在目录； \
> **-f**: 指定实验配置文件； \
> **-b**: 设置推理时的batch-size； \
> **-c**: 指定推理时的权重文件。

**注意**: 如果你省略`-c`选项，程序将自动调用你在[“*复现实验结果*”](#reproduceexperimentalresults)部分保存到“*StreakNet_outputs*”文件夹中的“*best_ckpt.pth*”权重文件。

```sh
python tools/demo.py --path datasets/clean_water_10m/data -f exps/streaknet/streaknet_s.py -b 512
                                     clean_water_13m                        streaknet_m.py
                                     clean_water_15m                        streaknet_l.py
                                     clean_water_20m                        streaknet_x.py
```

</details>

<details>
<summary>评估模型</summary>

* 第一步：执行[“*安装StreakNet*”](#quickstartinstallation)部分步骤。

* 第二步：执行[“*准备数据集*”](#preparedataset)部分步骤。

* 第三步：执行 [“*复现实验结果*”](#reproduceexperimentalresults)部分步骤。

* 第四步：评估模型。

```sh
python tools/valid.py -d 1 -b 512 -f exps/streaknet/streaknet_s.py --cache
                                                    streaknet_m.py
                                                    streaknet_l.py
                                                    streaknet_x.py
```

> 参数说明: \
> **-b**: 设置评估时的batch-size； \
> **-d**: 设置评估时使用的GPU数量（目前仅支持单卡评估）； \
> **-f**: 指定实验配置文件； \
> **--cache**: 启用RAM缓存加速。

</details>

<details>
<summary>传统信号处理方法</summary>

* 第一步：执行[“*安装StreakNet*”](#quickstartinstallation)部分步骤。

* 第二步：执行[“*准备数据集*”](#preparedataset)部分步骤。

* 第三步：运行传统信号处理方法。

```sh
python scripts/traditional_gpu_process.py
```

* 实验结果将被保存至“*StreakNet_outputs/traditional*”文件夹。

</details>

## 模型部署

1. [ONNX导出与ONNXRuntime示例](./demo/ONNXRuntime/)
2. [C++和Python语言的TensorRT部署示例](./demo/TensorRT/)

## 引用StreakNet
如果您在您的研究中使用了StreakNet，请使用以下BibTex格式引用我们的工作，谢谢！

```latex
@article{streaknet2024,
  title={xxx},
  author={xxx},
  journal={xxx},
  year={2024}
}
```

## 致敬前辈
* 我们在开发本开源项目时，借鉴了由[旷世科技](https://www.megvii.com/)发布的[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)项目的优秀工程架构。YOLOX项目由尊敬的[孙剑博士](https://baike.baidu.com/item/%E5%AD%99%E5%89%91/19814032)(1976年10月-2022年6月14日)主持开发，为计算机视觉领域的发展作出了卓越贡献。🕯️🕯️🕯️
* 当我们完成该项目的全部预实验时(2023年12月16日)，我们悲痛地听闻了[汤晓鸥教授](https://baike.baidu.com/item/%E6%B1%A4%E6%99%93%E9%B8%A5/7200225)(1968年1月-2023年12月15日)离世的消息。汤教授将全部精力奉献于计算机科学领域研究，为计算机视觉和人工智能技术发展作出了巨大贡献。谨以此向汤教授表达崇高的敬意和沉痛的哀悼。🕯️🕯️🕯️

## 版权说明

> 开发者：安泓郡(博士生)\
> 导师：[李学龙教授](https://iopen.nwpu.edu.cn/info/1329/1171.htm)， [孙哲副教授](https://iopen.nwpu.edu.cn/info/1251/2076.htm)

<br>
<div align="center"><img src="./assets/iopen.jpg" width="500"></div>
<div align="center"><p>Copyright &copy; School of Artificial Intelligence, OPtics and ElectroNics(iOPEN), Northwestern PolyTechnical University. <br>All rights reserved.</p></div>

