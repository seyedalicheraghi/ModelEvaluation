from ONNXUtilities import ONNXSurgery
import onnx
import yaml
import argparse
if __name__ == "__main__":
    outputNodes = []
    newName = 'sample'
    modelsName = "./model.onnx"
    yaml_file = ''
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-y", "--Yaml",
                        help="Path to the Yaml file which includes new output head!")
    parser.add_argument("-m", "--ONNXModel",
                        help="Name of the ONNX model to manipulate!")
    parser.add_argument("-n", "--Name",
                        help="Name of the new models for head, backbone, optimized model!")
    args = parser.parse_args()
    if args.Yaml:
        yaml_file = str(args.Yaml)
        with open(yaml_file, "r") as stream:
            try:
                outputNodes = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        err = 'Please select the Yaml file including new outputs!'
        raise Exception(err)
    if args.Name:
        newName = args.Name
    if args.ONNXModel:
        modelsName = args.ONNXModel
    print(f'Name of the new models to generate: {newName}')
    print(f'Name of the ONNX model to alter: {modelsName}')
    print(f'Name of the Yaml file: {yaml_file}')

    # Name of the ONNX file to get altered
    obj = ONNXSurgery(pathToOnnxModel=modelsName, newOutputNodes=outputNodes, newModelName=newName)
    obj.cutting_head()
    obj.bn_conv2d(onnx.load(newName+"Backbone.onnx"), pathToModelToSave='./'+newName+'Opt.onnx')
