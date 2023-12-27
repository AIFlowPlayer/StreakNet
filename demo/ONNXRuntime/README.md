## StreakNet-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Step1: Install onnxruntime

Run the following command to install onnxruntime:

```sh
pip install onnxruntime
```

### Step2: Get ONNX models

You can download our pre-generated ONNX models or convert your own models to ONNX.

#### Download ONNX models

|Model|F1-Score (%)|PSNR (dB)|Speed V100 (ms/pixel)|Speed NX (ms/pixel)|Params(M)|ONNX model|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[StreakNet-s](./exps/streaknet/streaknet_s.py)|85.46|8.75|00.0|00.00|1.25|[BaiduDisk](#)|
|[StreakNet-m](./exps/streaknet/streaknet_m.py)|85.87|8.91|00.0|00.00|3.15|[BaiduDisk](#)|
|[StreakNet-l](./exps/streaknet/streaknet_l.py)|85.90|8.92|00.0|00.00|10.51|[BaiduDisk](#)|
|[StreakNet-x](./exps/streaknet/streaknet_x.py)|86.23|9.00|00.0|00.00|50.40|[BaiduDisk](#)|
|(baseline)|41.07|4.64|00.0|00.00|---|---|

#### Convert Your Model to ONNX

First, you should move to the root directory of the project.

```sh
cd StreakNet
```

Then, you can convert your model by -f:

```sh
python tools/export_onnx.py --output-name streaknet_s.onnx -f exps/streaknet/streaknet_s.py 
                                          streaknet_m.onnx                   streaknet_m.py
                                          streaknet_l.onnx                   streaknet_l.py
                                          streaknet_x.onnx                   streaknet_x.py
```

You can visualize your onnx model by [Netron.app](https://netron.app/).

### Step3: ONNXRuntime Demo

* Step1.

```sh
cd StreakNet/demo/ONNXRuntime
```

* Step2.

```sh
python onnx_inference -m <ONNX_MODEL_PATH> --path <PATH_TO_STREAK_IMAGES> --template <PATH_TO_TEMPLATE_SIGNAL> --width 0.125 -b 256
```

Notes:
* -m: your converted onnx model
* -b: batch-size when inferring
* --path: path to streak images, e.g. *../../datasets/clean_water_10m/data*
* --template: path to template signal, e.g. *../../datasets/template.npy*
* --width: width factor of the network, see table below

|Model|Width Factor|
|:---:|:---:|
|StreakNet-s|0.125|
|StreakNet-m|0.25|
|StreakNet-l|0.50|
|StreakNet-x|1.00|
