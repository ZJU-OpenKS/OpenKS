## 算法支持
| 算法名称 |   算法功能   | 实现框架 |
| :------: | :----------: | :------: |
| DisenCTR | 可解释图推理 | PyTorch  |


数据集：ml-1m

将数据集置于 ./raw_data/ml-1m目录

```python
# 处理数据
python process_ML.py
# DisenCTR模型训练
python main.py --data ML --batch 1024 --patience 10 --nConvs 2 --K 4
```