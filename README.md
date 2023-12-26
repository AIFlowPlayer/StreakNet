<div align="center"><img src="./assets/streaknet_logo.png" width="400"></div>
<div align="center"><img src="./assets/demo.png"></div>

## Introduction

[StreakNet](https://github.com/BestAnHongjun/StreakNet) is a Deep-Learning (DL) based underwater imaging augmentation network for **Underwater Streak Camera LiDAR (USCL)** systems. It can achieves **millimeter** resolution underwater imaging at a distance of **20m**. For more details, please refer to out [paper](#).

<div align="center"><img src="./assets/streaknet_architecture.png"></div>

## Benchmark

StreakNet Benchmark

|Model|F1-Score (%)|PSNR (dB)|Speed V100 (ms/pixel)|Speed NX (ms/pixel)|Params(M)|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[StreakNet-s](./exps/streaknet/streaknet_s.py)|85.46|8.75|00.0|00.00|1.25|[BaiduDisk](#)|
|[StreakNet-m](./exps/streaknet/streaknet_s.py)|85.87|8.91|00.0|00.00|3.15|[BaiduDisk](#)|
|[StreakNet-l](./exps/streaknet/streaknet_s.py)|85.90|8.92|00.0|00.00|10.51|[BaiduDisk](#)|
|[StreakNet-x](./exps/streaknet/streaknet_s.py)|86.23|9.00|00.0|00.00|50.40|[BaiduDisk](#)|
|(baseline)|41.07|4.64|00.0|00.00|---|---|

## Dataset
**StreakData** is a underwater imaging dataset for Underwater Streak Camera LiDAR (USCL) systems. 

## Quick Start
<details>
<summary>Installation</summary>

Step1. Setup your conda environment.
```sh
conda create -n streaknet python=3.7
conda activate streaknet
```

Step2. Install StreakNet from source.
```sh
git clone https://github.com/BestAnHongjun/StreakNet.git
cd StreakNet
pip install -e .
```
</details>