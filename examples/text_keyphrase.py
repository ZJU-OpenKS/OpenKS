# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

from openks.loaders import loader_config, SourceType, FileType, Loader
from openks.models import OpenKSModel

''' 文本载入与MMD数据结构生成 '''
# 载入参数配置与数据集载入
loader_config.source_type = SourceType.LOCAL_FILE
loader_config.file_type = FileType.OPENKS
loader_config.source_uris = 'openks/data/patent-text'
loader_config.data_name = 'my-data-set'
loader = Loader(loader_config)
dataset = loader.dataset
dataset.info_display()

''' 文本信息抽取模型训练 '''
# 列出已加载模型
OpenKSModel.list_modules()
# 算法模型选择配置
args = {
    'extractor': 'topic-rake', 
    'finetuned': '/path/to/finetuned/word_embedding',
    'stopword': '/path/to/domain/stopwords.txt',
    'stopword_open': '/path/to/common/stopwords.txt', 
    'params': {
        'MIN_SCORE_TOTAL': 0.2,
        'MIN_WORD_LEN': 3,
        'SUFFIX_REMOVE': True,
        'STOPWORD_SINGLE_CHECK': True,
        'OPEN_STOPWORD': True,
        'WORD_SEPARATOR': True
    },
    'result_dir': loader_config.source_uris, 
    'rank': 'average'
}

platform = 'MLLib'
executor = 'KELearn'
model = 'keyphrase-rake-topic'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
text_keyphrase = executor(dataset=dataset, model=OpenKSModel.get_module(platform, model), args=args)
text_keyphrase.run()
print("-----------------------------------------------")
