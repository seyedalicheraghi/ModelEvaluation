import tensorflow as tf
import os
import argparse


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-o", "--Opset",
                        help="Select the version of Opset. Tensorflow accept version more that 13")
    parser.add_argument("-m", "--ONNXModel",
                        help="Name of the ONNX model converted from Saved_Model pb. Default is model.onnx")
    parser.add_argument("-t", "--SavedModel",
                        help="Path to the tensorflow saved_model.")
    parser.add_argument("-v", "--Verbose",
                        help="Please select True or False.")
    # Read arguments from command line
    args = parser.parse_args()
    opsetVersion = '13'
    ONNXName = 'model.onnx'
    verboseFlag = False
    if args.Opset:
        opsetVersion = str(args.Opset)
    if args.ONNXModel:
        ONNXName = str(args.ONNXModel)
    if args.Verbose:
        verboseFlag = str(args.Verbose)
    print(f"Opset selected for this conversion: {opsetVersion}")
    print(f"Name of the model selected in terminal: {ONNXName}")
    if not args.SavedModel:
        err = "Please select the path which includes the tensorflow saved model!"
        raise Exception(err)
    saved_model_dir = args.SavedModel
    print(f"Tensorflow version is: {tf.__version__}")

    verbose = ''
    tag = ''
    if verboseFlag:
        verbose = ' ' + '--verbose'

    tag = ' ' + '--tag' + ' ' + 'serve'
    os.system('pwd')
    savedModelPath = ' ' + '--saved-model' + ' ' + saved_model_dir
    outputCommand = ' ' + '--output' + ' ' + ONNXName
    command = 'python -m tf2onnx.convert'
    opsetCommand = ' ' + '--opset' + ' ' + opsetVersion # + ' --target nhwc'

    conv = command + savedModelPath + outputCommand + opsetCommand + verbose + tag
    os.system(conv)


