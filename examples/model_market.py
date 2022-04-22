# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import numpy as np
from openks.market import ModelLoader

model_name = 'TransE_FB-simple.onnx'

loader = ModelLoader(market_path='openks/market/trained_models')
# list all models in market
loader.list_models()

# check and output model's architecture
loader.check_model(model_name)

# get model's input variable shapes
model_shape = loader.get_model_shape(model_name)

# get model's input and output variable names
input_names, output_names = loader.get_model_io_names(model_name)

print(model_shape)
print(input_names, output_names)

# construct predict input
model_input = {}
for name, shape in zip(input_names, model_shape):
    model_input[name] = np.random.randn(*shape).astype(np.float32)
    
# execute model predict
print(loader.use_model(model_name, model_input))
