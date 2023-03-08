import torch
import onnxruntime
import numpy as np
import argparse

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-m", "--ONNXModel", help="Name of the reference ONNX model to inference.")
    parser.add_argument("-q", "--QuantizedModel", help="Name of the quantized ONNX model to inference.")

    args = parser.parse_args()
    if not args.ONNXModel:
        raise Exception("Please choose a model to inference!")
    if not args.QuantizedModel:
        raise Exception("Please choose a quantized model to inference!")
    qModel = args.QuantizedModel
    model = args.ONNXModel

    DummyOriginal = torch.rand(1, 320, 320, 3, dtype=torch.float32)
    DummyOriginal_Numpy = DummyOriginal.numpy().astype(np.uint8)
    #
    # # Step 1 ----> Get output of the original-model ----> Original_Output
    originalModelONNX = onnxruntime.InferenceSession(qModel)
    OriginalModelInputName = originalModelONNX.get_inputs()[0].name
    print(f"Input name is: {OriginalModelInputName}")
    originalOutput = originalModelONNX.run(None, {OriginalModelInputName: DummyOriginal_Numpy})
    print(f'Number of outputs: {len(originalOutput)}')

    Output_0 = originalOutput[0]
    Output_1 = originalOutput[1]
    Output_2 = originalOutput[2]
    Output_3 = originalOutput[3]
    Output_4 = originalOutput[4]
    Output_5 = originalOutput[5]
    Output_6 = originalOutput[6]
    Output_7 = originalOutput[7]

    # Step 1 ----> Get output of the original-model ----> Original_Output
    backBoneOptimizedONNX = onnxruntime.InferenceSession(model)
    backBoneOptimizedInputName = backBoneOptimizedONNX.get_inputs()[0].name
    print(f"Input name is: {backBoneOptimizedInputName}")
    backBoneOptimizedOutput = backBoneOptimizedONNX.run(None, {backBoneOptimizedInputName: DummyOriginal_Numpy})
    print(f'Number of outputs: {len(backBoneOptimizedOutput)}')
    backBoneOptimizedOutput_0 = backBoneOptimizedOutput[0]
    backBoneOptimizedOutput_1 = backBoneOptimizedOutput[1]
    backBoneOptimizedOutput_2 = backBoneOptimizedOutput[2]
    backBoneOptimizedOutput_3 = backBoneOptimizedOutput[3]
    backBoneOptimizedOutput_4 = backBoneOptimizedOutput[4]
    backBoneOptimizedOutput_5 = backBoneOptimizedOutput[5]
    backBoneOptimizedOutput_6 = backBoneOptimizedOutput[6]
    backBoneOptimizedOutput_7 = backBoneOptimizedOutput[7]

    MSE_0 = np.square((Output_0 - backBoneOptimizedOutput_0).flatten()).mean(axis=0)
    MSE_1 = np.square((Output_1 - backBoneOptimizedOutput_1).flatten()).mean(axis=0)
    MSE_2 = np.square((Output_2 - backBoneOptimizedOutput_2).flatten()).mean(axis=0)
    MSE_3 = np.square((Output_3 - backBoneOptimizedOutput_3).flatten()).mean(axis=0)
    MSE_4 = np.square((Output_4 - backBoneOptimizedOutput_4).flatten()).mean(axis=0)
    MSE_5 = np.square((Output_5 - backBoneOptimizedOutput_5).flatten()).mean(axis=0)
    MSE_6 = np.square((Output_6 - backBoneOptimizedOutput_6).flatten()).mean(axis=0)
    MSE_7 = np.square((Output_7 - backBoneOptimizedOutput_7).flatten()).mean(axis=0)

    OriginalModelOutputName_0 = originalModelONNX.get_outputs()[0].name
    OriginalModelOutputName_1 = originalModelONNX.get_outputs()[1].name
    OriginalModelOutputName_2 = originalModelONNX.get_outputs()[2].name
    OriginalModelOutputName_3 = originalModelONNX.get_outputs()[3].name
    OriginalModelOutputName_4 = originalModelONNX.get_outputs()[4].name
    OriginalModelOutputName_5 = originalModelONNX.get_outputs()[5].name
    OriginalModelOutputName_6 = originalModelONNX.get_outputs()[6].name
    OriginalModelOutputName_7 = originalModelONNX.get_outputs()[7].name
    print(Output_0.shape, backBoneOptimizedOutput_0.shape, MSE_0, OriginalModelOutputName_0)
    print(Output_1.shape, backBoneOptimizedOutput_1.shape, MSE_1, OriginalModelOutputName_1)
    print(Output_2.shape, backBoneOptimizedOutput_2.shape, MSE_2, OriginalModelOutputName_2)
    print(Output_3.shape, backBoneOptimizedOutput_3.shape, MSE_3, OriginalModelOutputName_3)
    print(Output_4.shape, backBoneOptimizedOutput_4.shape, MSE_4, OriginalModelOutputName_4)
    print(Output_5.shape, backBoneOptimizedOutput_5.shape, MSE_5, OriginalModelOutputName_5)
    print(Output_6.shape, backBoneOptimizedOutput_6.shape, MSE_6, OriginalModelOutputName_6)
    print(Output_7.shape, backBoneOptimizedOutput_7.shape, MSE_7, OriginalModelOutputName_7)

