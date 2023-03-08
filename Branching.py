from ONNXUtilities import ONNXSurgery
import onnx
branchName = ["StatefulPartitionedCall/ssd_mobile_net_v2keras_feature_extractor/functional_1/block_12_expand_relu/Relu6:0"]
modelOnnx = "./SSDOptOutputHead.onnx"
modelToModify = onnx.load(modelOnnx)
ONNXSurgery.branch_generator(modelToModify, branchName, newName='SSDOptOutputHeadBranch.onnx')
