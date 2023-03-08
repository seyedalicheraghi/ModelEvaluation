# ModelEvaluation
The purpose of this repository is to provide some python codes to manipulate models.
These scripts are examples to show how to start with ONNX models. I selected MobileNet V2
from the following link to prepare it for any AI accelerator. 

http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

## Goal of this repository
Given the SSD MobileNet v2 320x320 model from TF2 model zoo, please write a python program that:
* Converts the TF model to ONNX format.
* Optimizes the model for inference.
* Quantizes the model to int8 (PTQ is fine) and assesses the accuracy drop vs. the floating point model
* Prints out all the layers in a NN and their parameters:
    + Type of operation
    + Shape of input, output and weights (if applicable).
    + Total number of operations (e.g. flops).
    + Attempts to recover the accuracy drop due to quantization. 

#### Step 1
In this section we will go through the process to convert TensorFlow into the ONNX.
Please change the options based on your preference. You need to choose the Opset version
using `-o` option, output ONNX model using `-m`, `-v` to enable the verbose, and `-t` 
to direct to the path of the `saved_model`.  

```commandline
python Step1_TFToONNX.py -o 13 -m mobileNetV2.onnx -t ./ssd_mobilenet_v2/saved_model -v True
```
#### Step 2  
In this step we try to separate the Head from the body and optimize the backbone. 
You need to change YAML file for each model. Nodes that will be considered as the 
new heads should be inserted into the YAML file. 
You can use the following command to do that. You need to select the YAML
file using `-y`, new name for models to cut and optimized using `-n`, and the model 
you want to manipulate using `-m` option. 
```commandline
python Step2_BBSepSSD.py -y nodesToIgnore.yaml -n SSD -m mobileNetV2.onnx 
```
This command will generate 3 ONNX files:
* SSDBackbone.onnx
* SSDOutputHead.onnx
* SSDOpt.onnx

#### Step 3
Separate the normalization layers (first layers of the model) from the backbone and head.
The code is similar to the Step 2 but the output layer name is different. This code is derived 
from code presented in Step 2. You need to run the code using:
```commandline
python Step3_BBSepSSD.py
```
#### Step 4
To quantize the ONNX model using ONNX quantization you don't need to go through 
explained steps. You can directly use the following command to get the quantized model.

```commandline
python ONNXQuantization.py -m mobileNetV2.onnx -q mobileNetV2_quantized.onnx
```
For tflite quantization you need to use the Google Colab and run the tflite.py code into it.


#### Step 5
This step takes the list of supported operators and generate a .csv file which includes
details about operators within the YAML file.
```commandline
python LayerPrinting.py -y supportedOperators.yaml -m SSDOptOutputHead.onnx
```

#### Step 6
To run Quantized model and the original ONNX model side by side you need to use 
the `ONNX_runtime.py`. It creates a random numpy array and pass it through both
quantized and none-quantized models, then produce the result and use MSE to 
compare them with each other. Please use the following command line that `-m` 
represent the original model and `-q` refers to the quantized model:
```commandline
python ONNX_runtime.py -m mobileNetV2.onnx -q mobileNetV2_quantized.onnx
```
#### Results
After running the previous 3 steps you should have the following ONNX files:

* mobileNetV2.onnx (original model after converting Tensorflow to ONNX)
* SSDBackbone.onnx (Backbone generated after going through Step 2)
* SSDOutputHead.onnx (Head generated after going through Step 2)
* SSDOpt.onnx (Optimized backbone generated after going through Step 2)
* SSDOptBackbone.onnx (Normalization layers separated after Step 3)
* SSDOptOutputHead.onnx (Final model to feed into an accelerator)



