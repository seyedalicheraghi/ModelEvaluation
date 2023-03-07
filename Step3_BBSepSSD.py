from ONNXUtilities import ONNXSurgery
import onnx

if __name__ == "__main__":
    # # Output nodes after the cut
    outputNodes = ['StatefulPartitionedCall/ssd_mobile_net_v2keras_feature_extractor/functional_1/Conv1/Conv2D__47:0']
    # Name of the ONNX file to get altered
    modelsName = "./newModeOptimized.onnx"
    obj = ONNXSurgery(pathToOnnxModel=modelsName, newOutputNodes=outputNodes, newModelName="SSDOPT")
    obj.cutting_head()