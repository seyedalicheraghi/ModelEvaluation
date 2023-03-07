import tensorflow as tf
import os

print(f"Tensorflow version is: {tf.__version__}")
saved_model_dir = './ssd_mobilenet_v2/saved_model'
ONNXName = 'model1.onnx'
opsetVersion = '13'
verboseFlag = True
verbose = ''
tag = ''
if verboseFlag:
    verbose = ' ' + '--verbose'

tag = ' ' + '--tag' + ' ' + 'serve'
os.system('pwd')
savedModelPath = ' ' + '--saved-model' + ' ' + saved_model_dir
outputCommand = ' ' + '--output' + ' ' + ONNXName
command = 'python -m tf2onnx.convert'
opsetCommand = ' ' + '--opset' + ' ' + opsetVersion + ' --target nhwc'

conv = command + savedModelPath + outputCommand + opsetCommand + verbose + tag
os.system(conv)
print(conv)


