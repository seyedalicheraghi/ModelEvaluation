from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.quantize import quantize_static
from onnxruntime.quantization.calibrate import create_calibrator, MinMaxCalibrater

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


from pathlib import Path
pathModel = Path('model.onnx')
pathQuant = Path('dqModel.onnx')
qObj = ModelQuantization(pathToFP32Model=pathModel, pathToIntModel=pathQuant)
dynamicQ = qObj.dynamic_quantization()
# MinMaxCalibrater(model='newModeOptimized.onnx')
