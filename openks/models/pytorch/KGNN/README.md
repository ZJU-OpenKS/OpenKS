这是发表于 WSDM21的 [论文](https://dl.acm.org/doi/pdf/10.1145/3488560.3498429)的代码实现，论文信息：
> Ju, Wei, et al. "Kgnn: Harnessing kernel-based networks for semi-supervised graph classification." *Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining*. 2022.

这篇论文提出了一种同时基于消息传递和图核方法的图表征学习方法，能够完成基于子图特征或全图特征的图推理。

如果使用了该项目的代码，请对上述论文进行引用。

### 环境配置
- python == 3.8.13
- pytorch == 1.11.0 
- torch_geometric == 2.0.4 
- pandas == 1.4.1
- sklearn == 0.23.2

### 示例
该模型的训练与测试基于多个通用的图分类数据集，如果要在  `PROTEINS`  数据集上测试，则使用以下命令：
```shell
python main.py --DS PROTEINS
```
更多参数请参考 `arguments.py` 或者使用以下命令：
```shell
python main.py -h
```

可选数据集包括 `PROTEINS` `DD` `Mutagenicity` `IMDB-BINARY` `IMDB-MULTI` `REDDIT-BINARY` `REDDIT-MULTI-5K` `COLLAB` `PTC_MR` `MUTAG` `NCI1` `ENZYMES` `FRANKENSTEIN`