这是发表于ICDM2022 的[论文](https://arxiv.org/abs/2210.03969)的实现，论文信息：
> Wei Ju, Yifang Qin, Ziyue Qiao, Xiao Luo, Yifan Wang, Yanjie Fu, and Ming Zhang(2022). Kernel-based Substructure Exploration
for Next POI Recommendation

在这篇论文中提出了一个  Kernel-Based Graph Neural Network (KBGNN) 用来做 next POI recommendation，这个方法结合了图形信息和时序信息。

如果使用了该项目的代码，请对上述论文进行引用。

### 环境配置
- python == 3.8.13
- pytorch == 1.11.0 
- torch_geometric == 2.0.4 
- pandas == 1.4.1
- sklearn == 0.23.2

###  示例

如果想要基于  `Foursquare-Tokyo` 数据训练和预测，首先应该下载 [原始数据](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) 并将其解压至  `~/raw_data/` 。

随后，运行命令
```shell
mkdir processed && cd processed
mkdir tky
cd ../utils
python process_data.py
```
这些命令会在 `~/processed/tky/` 目录下生成处理后的数据。

随后，可以通过以下命令在处理好的`Foursquare-Tokyo` 数据集上进行实验：
```shell
cd ./model
python main.py --data tky --batch 1024 --patience 10 --gcn_num 2 --max_step 2
```
对于更多运行参数的设置，请参考  `~/model/main.py`，或者使用以下命令：
```shell
python main.py -h
```