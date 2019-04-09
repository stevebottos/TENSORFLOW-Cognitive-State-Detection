# TENSORFLOW-Cognitive-State-Detection
A Tensorflow (non-keras) version of previous work. This repository does not include the data cleaning steps, since that portion was written in Matlab and can be found in the repository MATLAB-Cognitive-State-Detection and MATLAB-Cognitive-State-Detection. 

## A Quick Note
This repo is a bit sparse, for full context and a full description, please check out the main codebase (the Matlab implementation). The main purpose of this repo is to provide a comparison between how a Neural Network is implemented from scratch (as in PYTHON-Cognitive-State-Detection or MATLAB-Cognitive-State-Detection) and how a NN is implemented using Tensorflow (non-Keras, as I had written this program prior to TF adopting the Keras wrapper), and illustrate the accuracy improvements that Tensorflow offers - as well as faster training times.

## Conclusions
Tensorflow offers increased productivity in the form of quicker-to-write code and faster training times. This was written pre-Keras, however with the Keras wrapper code is much more intuitive and simple to write. The optimization functions that TF offers are great, and allowed me to train my models in a fraction of the time that Matlab or Python+Numpy in conjunction with the GPU acceleration capabilities. Comparing the two files **MatlabResults** in MATLAB-Cognitive-State-Detection and **TensorflowResults** in this repo demonstrate the notable accuracy improvements that Tensorflow offers.

Please message me for data if you wish to run this code for yourself using the same inputs as I did.
