# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from openks.loaders import loader_config, SourceType, FileType, Loader
from openks.models import OpenKSModel

''' 文本载入与MMD数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/industry-text'
loader_config.data_name = 'my-data-set'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()

''' 文本信息抽取模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
args = {
	'gpu': True, 
	'embedding_dim':300,
	'hidden_dim':300, 
	'epoch': 20, 
	'batch_size': 32,
	'use_crf': True,
	'model_dir': './industry_ner_model/',
	'log_file_path': './industry_ner_model/run.log'
}
platform = 'TensorFlow'
executor = 'KELearn'
model = 'industry-entity-extract'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
industry_ner = executor(dataset=dataset, model=OpenKSModel.get_module(platform, model), args=args)
industry_ner.run()

'''
# 模型预测
# 数据是sentense的list, sentence中用空格隔开
sentences = ["三 是 深 圳 富 满 电 子 集 团 股 份 有 限 公 司 董 事 徐 浙 买 卖 公 司 股 票 行 为 构 成 短 线 交 易 。 本 所 发 出 年 报 问 询 函 6 6 份 、 重 组 问 询 函 4 份 、 关 注 函 6 份 、 其 他 函 件 3 9 份 。"]
result = industry_ner.predict(sentences)
print(result)
'''

print("-----------------------------------------------")
