# How to retrain the model zoo using own dataset to enhance specific model. 

This repository is a simple tutorial. it comes from idea that how to enhance model by adding new dataset based on [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

I am going to explain that how to make record files from official dataset of Coco dataset and how to add your own dataset with your labeled record files. Eventually it could be trained with tensorflow graph. 

## Set up Training environment. 
Basically, I worked to refer up the Tensorflow model posted Github [repository](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md). So you might have to follow it to set up environment for Tensorflow.
<pre> environment in my case
* Ubuntu 18.04.3 LTS 
* Conda 4.5.11 
* Tensorflow-gpu 1.13.2 
* Nvida GeForce GTX 1080Ti
</pre>

## 1. Make Record file using Coco Dataset. 

## 2. Re-train Coco dataset including Own Dataset. 

## 3. Export Non-frozen model for Tensorflow serving. 



