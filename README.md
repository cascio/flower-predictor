# flower-predictor

This project uses Tensorflow2.0 to predict what kind of flower is in an image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Python Version

This project was built using Python 3.7.6

```
python
```
```
Python 3.7.6 (tags/v3.7.6:43364a7ae0, Dec 19 2019, 00:42:30) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Prerequisites

What packages are needed to run the code

```
git clone git@github.com:cascio/learning-tf.git
pip install requirements.txt
```

## Files

What each file does

### test_for_gpu.py

This runs several tests to check that the device has and can properly use GPUs for certain computations

```
python test_for_gpu.py
```
```
2020-01-02 01:23:18.712194: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
True
2020-01-02 01:23:20.539551: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-01-02 01:23:20.546753: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-01-02 01:23:20.570814: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1050 Ti with Max-Q Design major: 6 minor: 1 memoryClockRate(GHz): 1.2905
pciBusID: 0000:01:00.0
2020-01-02 01:23:20.576485: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-01-02 01:23:20.578936: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-01-02 01:23:21.095981: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-02 01:23:21.099437: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-01-02 01:23:21.101355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-01-02 01:23:21.103648: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 2996 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 6.1)
True
2.0.0
2020-01-02 01:23:21.109809: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1050 Ti with Max-Q Design major: 6 minor: 1 memoryClockRate(GHz): 1.2905
pciBusID: 0000:01:00.0
2020-01-02 01:23:21.114444: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-01-02 01:23:21.119480: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
Num GPUs Available:  1
2020-01-02 01:23:21.122323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1050 Ti with Max-Q Design major: 6 minor: 1 memoryClockRate(GHz): 1.2905
pciBusID: 0000:01:00.0
2020-01-02 01:23:21.127845: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2020-01-02 01:23:21.132461: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-01-02 01:23:21.135108: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-01-02 01:23:21.137514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2020-01-02 01:23:21.139189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2020-01-02 01:23:21.144584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2996 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 6.1)
2020-01-02 01:23:21.152821: I tensorflow/core/common_runtime/eager/execute.cc:574] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2020-01-02 01:23:21.157799: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
 ```

[See tensorflow.org/install/gpu for GPU support](https://www.tensorflow.org/install/gpu)

### train.py

This code first uses tensorflow_datasets to download the 'tf_flowers' dataset. tensorflow_hub is used to load a pre-trained MobileNetV2 model, not including its final layer. The layers of the pretrained model are frozen and a Keras Dense layer is appended. A model is compiled and trained on the tf_flowers dataset. Keras h5 model weights are saved for future use.

```
python train.py
```

### predict_with_h5_model.py

Previously trained Keras h5 model weights are used to predict which flower is in an image. This file saves a .PNG file with a series of image predictions and associated accuarcy.

```
python predict_with_h5_model.py
```