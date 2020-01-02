# flower-predictor

This project uses Tensorflow2.0 to predict what kind of flower is in an image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

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