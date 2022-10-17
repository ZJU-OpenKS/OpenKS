# OpenKS模型市场

## 模型市场实现原理
模型市场采用ONNX作为训练后模型的通用存储格式，以实现对训练后的知识计算模型进行统一集成和标准化服务

## 模型市场使用方式
1. 开发者模式
* 分别在`models`模块中的`pytorch`、`paddle`、`tensorflow`中，在`kg_learn`、`ke_learn`等执行器脚本中实现`model_to_onnx`方法，方法实现逻辑可参考`models/pytorch/kg_learn.py`中的`model_to_onnx`，并在`run`方法中每次模型评估后调用该方法，以进行模型的持久化。
* 在模型训练脚本中，指定`market_path`参数为`openks/market/trained_models/xxx.onnx`，开始训练。

2. 使用者模式
* 声明`market`模块中的`ModelLoader`类，指定要使用的模型名称
* 构造待传入模型的输入数据
* 调用`ModelLoader`类中的`use_model`方法实现模型预测
（参考examples/model_market.py示例）