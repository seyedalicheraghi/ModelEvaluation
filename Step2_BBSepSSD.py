from ONNXUtilities import ONNXSurgery
import onnx

if __name__ == "__main__":
    # # Output nodes after the cut
    outputNodes = ['StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_0/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_1/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_2/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_3/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_4/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_0/ClassPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_1/ClassPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_2/ClassPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_3/ClassPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_4/ClassPredictor/BiasAdd:0',
                   'StatefulPartitionedCall/BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/BiasAdd:0',
                   ]
    # Name of the ONNX file to get altered
    modelsName = "./model.onnx"
    obj = ONNXSurgery(pathToOnnxModel=modelsName, newOutputNodes=outputNodes, newModelName="SSD")
    obj.cutting_head()
    obj.bn_conv2d(onnx.load("./SSDBackbone.onnx"))
