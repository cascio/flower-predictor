# flower-predictor

This project uses Tensorflow2.0 to predict what kind of flower is in an image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Setup Virtual Environment 

```
$ virtualenv --python=/c/users/michael.cascio/appdata/local/programs/python/python37/python ./venv

Running virtualenv with interpreter C:/users/michael.cascio/appdata/local/programs/python/python37/python.exe
Already using interpreter C:\users\michael.cascio\appdata\local\programs\python\python37\python.exe
Using base prefix 'C:\\users\\michael.cascio\\appdata\\local\\programs\\python\\python37'
New python executable in C:\Users\michael.cascio\Documents\GitHub\flower-predictor\venv\Scripts\python.exe
Installing setuptools, pip, wheel...
done.
```
```
$ python

Python 3.7.6 (tags/v3.7.6:43364a7ae0, Dec 19 2019, 00:42:30) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Prerequisites

What packages are needed to run the code

```
$ git clone git@github.com:cascio/learning-tf.git
$ pip install -r requirements.txt
```

## Files

What each file does

### test_for_gpu.py

This runs several tests to check that the device has and can properly use GPUs for certain computations

```
$ python test_for_gpu.py
```


[See tensorflow.org/install/gpu for GPU support](https://www.tensorflow.org/install/gpu)

### train.py

This code first uses tensorflow_datasets to download the 'tf_flowers' dataset. tensorflow_hub is used to load a pre-trained MobileNetV2 model, not including its final layer. The layers of the pretrained model are frozen and a Keras Dense layer is appended. A model is compiled and trained on the tf_flowers dataset. Keras h5 model weights are saved for future use.

```
$ python train.py
```

### predict_with_h5_model.py

Previously trained Keras h5 model weights are used to predict which flower is in an image. This file saves a .PNG file with a series of image predictions and associated accuarcy.

```
$ python predict_with_h5_model.py
```

Here is an example:
![](https://github.com/cascio/flower-predictor/blob/master/0.png?raw=true)