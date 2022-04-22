# Copyright (c) 2021 OpenKS Authors, DCD Research Lab and Dlib Lab, Zhejiang University and Peking University. 
# All Rights Reserved.

import os
import numpy as np
import onnx
import onnxruntime as ort


class ModelLoader(object):

    def __init__(self, market_path='openks/market/trained_models'):
        self.market_path = market_path

    def check_model(self, model_name):
        # Load the ONNX model
        model = onnx.load(os.path.join(self.market_path, model_name))

        # Check that the model is well formed
        onnx.checker.check_model(model)

        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))


    def use_model(self, model_name, model_input):
        ort_session = ort.InferenceSession(os.path.join(self.market_path, model_name))
        outputs = ort_session.run(
            None,
            model_input,
        )
        return outputs[0]

    def list_models(self):
        print('Trained models in market:')
        for item in os.listdir(self.market_path):
            if item.endswith(".onnx"):
                print(item)
        print('-------------------------------')

    def get_model_shape(self, model_name):
        model = onnx.load(os.path.join(self.market_path, model_name))
        model_shape = []
        for input in model.graph.input:
            shape = []
            tensor_type = input.type.tensor_type
            if (tensor_type.HasField("shape")):
                for d in tensor_type.shape.dim:
                    if (d.HasField("dim_value")):
                        shape.append(d.dim_value)
                    elif (d.HasField("dim_param")):
                        shape.append(d.dim_param)
                    else:
                        print ("?", end=", ")
            else:
                print ("Unknown shape")
            model_shape.append(shape)
        return model_shape

    def get_model_io_names(self, model_name):
        model = onnx.load(os.path.join(self.market_path, model_name))
        input_all = [node.name for node in model.graph.input]
        input_initializer =  [node.name for node in model.graph.initializer]
        input_names = list(set(input_all)  - set(input_initializer))
        output_names =[node.name for node in model.graph.output]
        return input_names, output_names



