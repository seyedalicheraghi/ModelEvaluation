from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.quantize import quantize_static
from onnxruntime.quantization.calibrate import create_calibrator, MinMaxCalibrater
from pathlib import Path
import argparse


class ModelQuantization:
    def __init__(self, pathToFP32Model, pathToIntModel):
        """
        :param pathToFP32Model: pathlib
        Path to the FP32 model.
        :param pathToIntModel: pathlib
        Path to the Int8 model for quantization.
        """
        self.pathToFP32Model = pathToFP32Model
        self.pathToIntModel = pathToIntModel

    def dynamic_quantization(self):
        return quantize_dynamic(self.pathToFP32Model, self.pathToIntModel)

    # def static_quantization(self):
    #     create_calibrator()
    #     quantize_static()


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    qModel = 'dqModel.onnx'
    # Adding optional argument
    parser.add_argument("-m", "--ONNXModel", help="Name of the ONNX model need to get quantized.")
    parser.add_argument("-q", "--QuantizedModel",
                        help="Name of the model you want to name after quantization. Default is called dqModel.onnx")

    args = parser.parse_args()
    if not args.ONNXModel:
        raise Exception("Please choose a model for quantization!")
    if args.QuantizedModel:
        qModel = args.QuantizedModel
    model = args.ONNXModel
    pathModel = Path(model)
    pathQuant = Path(qModel)
    qObj = ModelQuantization(pathToFP32Model=pathModel, pathToIntModel=pathQuant)
    dynamicQ = qObj.dynamic_quantization()
