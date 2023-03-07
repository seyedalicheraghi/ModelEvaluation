import onnx
import os.path


class ONNXSurgery:
    def __init__(self,
                 pathToOnnxModel,
                 newOutputNodes,
                 newModelName='newModel'
                 ):
        if not os.path.isfile(pathToOnnxModel):
            raise Exception("The ONNX file does not exist! Please check the directory or the path you provided!")
        self.pathToOnnxModel = pathToOnnxModel
        self.modelToOptimize = onnx.load(self.pathToOnnxModel)
        self.newOutputNodes = newOutputNodes
        self.newModelName = newModelName

    def find_nodes_cut(self, nodesToIgnore):
        """
        :param nodesToIgnore:
        Find nodes that need to get cut from the backbone and inserted for the head.
        :return:
        List of the nodes that are not in the backbone but in the head of the model.
        """
        for n in self.modelToOptimize.graph.node:
            for inputs in n.input:
                if inputs in nodesToIgnore:
                    for i in n.input:
                        if i not in nodesToIgnore:
                            nodesToIgnore.append(i)
                    for o in n.output:
                        if o not in nodesToIgnore:
                            nodesToIgnore.append(o)
        for items in self.newOutputNodes:
            if items in nodesToIgnore:
                nodesToIgnore.pop(nodesToIgnore.index(items))
        return nodesToIgnore

    @staticmethod
    def branch_generator(modelOnnx, branchName):
        """
        :param modelOnnx:
        ONNX model to create output branches.
        :param branchName:
        Operators output name which you want to check the output of.
        :return:
        A model which include the new model with added outputs to get checked.
        """
        shapeInf = onnx.shape_inference.infer_shapes(modelOnnx)
        opsetImport = shapeInf.opset_import
        # Get shapes of each layer
        shapes = {n.name: n for counter, n in enumerate(shapeInf.graph.value_info)}
        modelOutput = [shapes.get(items) for items in branchName]
        modelOriginalOutput = modelOnnx.graph.output
        print(f"The model has {len(modelOriginalOutput)} outputs!")
        print(f"You want to add {len(modelOutput)} output branches!")
        for items in modelOutput:
            modelOriginalOutput.append(items)
        modelOriginalInput = [items for items in modelOnnx.graph.input]
        FullGraph_Values = {n.name: n for n in modelOnnx.graph.initializer}
        nodes = []
        values = []
        for n in modelOnnx.graph.node:
            for v in n.input:
                if FullGraph_Values.get(v) is not None:
                    w = FullGraph_Values.get(v)
                    values.append(w)
            nodes.append(n)

        backBoneModelDef = onnx.helper.make_graph(nodes, 'newModel', modelOriginalInput, modelOriginalOutput,
                                                  initializer=values)
        backBoneModel = onnx.helper.make_model(backBoneModelDef, producer_name='ali', opset_imports=opsetImport)
        backBoneModel.ir_version = 7
        onnx.save(backBoneModel, './ModelWithNewOutputs.onnx')

    def cutting_head(self):
        """
        :param newModelName:
        The name of the model you need to assign for the head and backbone separated parts.
        :return:
        This method will generate two ONNX files:
            1. Backbone model: It will add "Backbone" to the end of the "newModelName" you initialized.
            2. Head model: It will add "OutputHead" to the end of the "newModelName" you initialized.
        """
        backboneNodes = []
        backboneValues = []
        outputHeadNodes = []
        outputHeadValues = []

        # Separate the backbone from the output head
        FullGraph_Values = {n.name: n for n in self.modelToOptimize.graph.initializer}
        nodesToGetCut = [items for items in self.newOutputNodes]
        nodesToGetCut = self.find_nodes_cut(nodesToGetCut)

        for n in self.modelToOptimize.graph.node:
            for outputs in n.output:
                if outputs not in nodesToGetCut:
                    for v in n.input:
                        if FullGraph_Values.get(v) is not None:
                            w = FullGraph_Values.get(v)
                            if w not in backboneValues:
                                backboneValues.append(w)
                    if n not in backboneNodes:
                        backboneNodes.append(n)
                else:
                    for v in n.output:
                        if FullGraph_Values.get(v) is not None:
                            w = FullGraph_Values.get(v)
                            if w not in outputHeadValues:
                                outputHeadValues.append(w)
                    if n not in outputHeadNodes:
                        outputHeadNodes.append(n)
            for inputs in n.input:
                if inputs in nodesToGetCut and inputs not in outputHeadNodes:
                    if FullGraph_Values.get(inputs) is not None:
                        w = FullGraph_Values.get(inputs)
                        if w not in outputHeadValues:
                            outputHeadValues.append(w)
                    if n not in outputHeadNodes:
                        outputHeadNodes.append(n)

        shapeInf = onnx.shape_inference.infer_shapes(self.modelToOptimize)
        opsetImport = shapeInf.opset_import
        # Get shapes of each layer
        shapes = {n.name: n for counter, n in enumerate(shapeInf.graph.value_info)}
        backBoneInput = [items for items in self.modelToOptimize.graph.input]
        backBoneOutput = [shapes.get(items) for items in self.newOutputNodes]
        headModelOutput = self.modelToOptimize.graph.output

        backBoneModelDef = onnx.helper.make_graph(backboneNodes, 'newModel', backBoneInput, backBoneOutput,
                                                  initializer=backboneValues)

        backBoneModel = onnx.helper.make_model(backBoneModelDef, producer_name='ali', opset_imports=opsetImport)
        backBoneModel.ir_version = 7
        onnx.save(backBoneModel, './' + self.newModelName + 'Backbone.onnx')

        headInput = [shapes.get(items) for items in self.newOutputNodes]
        headDef = onnx.helper.make_graph(outputHeadNodes, 'newModel', headInput, headModelOutput,
                                         initializer=outputHeadValues)

        headModel = onnx.helper.make_model(headDef, producer_name='ali', opset_imports=opsetImport)
        headModel.ir_version = 7
        onnx.save(headModel, './' + self.newModelName + 'OutputHead.onnx')

    @staticmethod
    def bn_conv2d(onnxModel, pathToModelToSave="./newModeOptimized.onnx"):
        """
        :param onnxModel:
        This is the ONNX model which is used for optimization.
        This is not the path but the onnx model.
        :param pathToModelToSave:
        The path to save the optimized model. If its not defined it would consider current directory as the destination.
        :return:
        It would not return any value.
        """
        import onnxoptimizer as op

        onnxModel.ir_version = 7
        model = op.optimize(onnxModel, ['fuse_bn_into_conv'])
        # save optimized model
        with open(pathToModelToSave, "wb") as f:
            f.write(model.SerializeToString())

    @staticmethod
    def check_onnx_version(modelPathToCheck):
        """
        :param modelPathToCheck:
        Need the path to the model. It must be the path to the model.
        :return:
        It would return the version of the ONNX model.
        """
        onnxModel = onnx.load(modelPathToCheck)
        return onnx.shape_inference.infer_shapes(onnxModel).opset_import

    def rbn_conv2D(self):
        # TODO
        pass

    def bn_depth_conv(self):
        # TODO
        pass

    @property
    def model_path(self):
        """
            This method return the path to the model for processing.
        """
        return self.pathToOnnxModel

    @property
    def model_onnx(self):
        """
        :return:
        Return the ONNX model. You could use it within the visualization method.
        """
        return self.modelToOptimize

    @staticmethod
    def attaching_head_body(onnxBackBone, onnxHead, inputData):
        pass

    @staticmethod
    def model_visualizer(onnxFile):
        """
        :param onnxFile:
        The ONNX you want to visualize. This must be a string about the path to the model.
        :return:
        This is a visualization method so it will not have anything to return.
        """
        import netron
        netron.start(onnxFile)
