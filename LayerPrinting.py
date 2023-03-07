import onnx
from onnx import shape_inference
import os
import pathlib
# from google.protobuf.json_format import MessageToDict
import csv
import glob
import onnxruntime

supportedOperators = ['Conv', 'LeakyRelu', 'MaxPool', 'Add', 'Clip']


onnx_directory = "./SSDOPTOutputHead.onnx"
print("*" * 100)
print(onnx_directory)
original_onnx = onnx.load(onnx_directory)
print(onnx_directory)
# Check the model
try:
    onnx.checker.check_model(original_onnx)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s')
else:
    print('The model is valid!')

input_output_dims = {}
model_layers = {}
input_data = original_onnx.graph.input[0]
input_output_dims[input_data.name] = {"name": input_data.name,
                                      "elem_type": input_data.type.tensor_type.elem_type,
                                      "shape": input_data.type.tensor_type.shape.dim}
Onnx_with_sizes = onnx.shape_inference.infer_shapes(original_onnx, strict_mode=True)
onnx.save(Onnx_with_sizes, onnx_directory.split('.')[0] + '.onnx')

session = onnxruntime.InferenceSession(onnx_directory, None)
# input_name = session.get_inputs()[0].name
input_name = [items.name for items in session.get_inputs()]
last_layer_name = [items.name for items in session.get_outputs()]
# last_layer_name = session.get_outputs()[0].name
last_layer_shape = [items.shape for items in session.get_outputs()]
# last_layer_shape = session.get_outputs()[0].shape

# This section saves layers shapes
layer_dimensions = Onnx_with_sizes.graph.value_info

for layer_dims in layer_dimensions:
    input_output_dims[layer_dims.name] = {"name": layer_dims.name,
                                          "elem_type": layer_dims.type.tensor_type.elem_type,
                                          "shape": layer_dims.type.tensor_type.shape.dim}

# This section tries to save graph information
header = ['Type', 'Support', 'Input Name', 'Output Name',
          'Input Batch', 'Input Width', 'Input Height', 'Input Depth',
          'Left Pad', 'Right Pad', 'Top Pad', 'Bottom Pad',
          'Stride Width', 'Stride Height',
          'Conv width', 'Conv height', 'Conv depth',
          'Output width', 'Output height', 'Output depth']

with open('results.csv', 'w') as f1:

    writer = csv.writer(f1)
    writer.writerow(header)
    for node in Onnx_with_sizes.graph.node:
        if node.op_type in supportedOperators:
            OpSupport = 'Yes'
        else:
            OpSupport = 'No'
        # Check if we are evaluating conv operation
        if node.op_type == 'Conv':

            for item in node.input:
                if item in input_output_dims:
                    input_batch_size = 0
                    input_channel = 0
                    input_width = 0
                    input_height = 0
                    conv_pad = [0, 0, 0, 0]
                    output_name = node.output[0]
                    input_name = item
                    conv_stride = [-1, -1]
                    conv_Kernel = [0, 0]
                    output_channel = 0
                    output_width = 0
                    output_height = 0
                    OpType = node.op_type
                    for elem in node.attribute:
                        if elem.name == "pads":
                            conv_pad = [elem.ints[0], elem.ints[1], elem.ints[2], elem.ints[3]]
                        elif elem.name == "kernel_shape":
                            conv_Kernel = [elem.ints[0], elem.ints[1]]
                        elif elem.name == "strides":
                            conv_stride = [elem.ints[0], elem.ints[1]]
                        elif elem.name == "dilations":
                            conv_dilation = elem.ints[0], elem.ints[1]
                        # Get input dimensions
                        try:
                            input_batch_size = input_output_dims[input_name]['shape'][0].dim_value
                            input_channel = input_output_dims[input_name]['shape'][1].dim_value
                            input_width = input_output_dims[input_name]['shape'][2].dim_value
                            input_height = input_output_dims[input_name]['shape'][3].dim_value
                        except:
                            print('This code works for models with known input sizes. Some ONNX models do not have '
                                  'input sizes for each layer. These models generates an error to visit the model '
                                  'first and check if it has shapes in each layer. Need to get calculated by hand')

                        if output_name in last_layer_name:
                            i = last_layer_name.index(output_name)
                            # Get output dimensions
                            output_batch_size = last_layer_shape[i][0]
                            output_channel = last_layer_shape[i][1]
                            output_width = last_layer_shape[i][2]
                            output_height = last_layer_shape[i][3]
                        else:
                            # Get output dimensions
                            output_batch_size = input_output_dims[output_name]['shape'][0].dim_value
                            output_channel = input_output_dims[output_name]['shape'][1].dim_value
                            output_width = input_output_dims[output_name]['shape'][2].dim_value
                            output_height = input_output_dims[output_name]['shape'][3].dim_value

                        data = [OpType, OpSupport, input_name, output_name,  # Input and Output Names
                                input_batch_size, input_width, input_height, input_channel,  # Input Size
                                conv_pad[0], conv_pad[1], conv_pad[2], conv_pad[3],  # Convolution Size
                                conv_stride[0], conv_stride[1],  # Stride Size
                                conv_Kernel[0], conv_Kernel[1], output_channel,  # Convolution Kernel Size
                                output_width, output_height, output_channel  # Output Size
                                ]
                    writer.writerow(data)
        else:
            OpType = node.op_type
            if OpType == 'BatchNormalization':
                OpType = 'BN'

            input_batch_size = ''
            input_channel = ''
            input_width = ''
            input_height = ''
            conv_pad = ['', '', '', '']
            output_name = node.output[0]
            if len(output_name) > 7:
                output_name = output_name[-7:]
            input_name = item
            if len(input_name) > 7:
                input_name = input_name[-7:]
            conv_stride = ['', '']
            conv_Kernel = ['', '']
            output_channel = ''
            output_width = ''
            output_height = ''
            data = [OpType, OpSupport, input_name, output_name,  # Input and Output Names
                    input_batch_size, input_width, input_height, input_channel,  # Input Size
                    conv_pad[0], conv_pad[1], conv_pad[2], conv_pad[3],  # Convolution Size
                    conv_stride[0], conv_stride[1],  # Stride Size
                    conv_Kernel[0], conv_Kernel[1], output_channel,  # Convolution Kernel Size
                    output_width, output_height, output_channel  # Output Size
                    ]
            writer.writerow(data)
f1.close()
