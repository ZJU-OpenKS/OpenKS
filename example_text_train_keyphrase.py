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
    'pretrained': '/Users/zongchang/Downloads/embedding_model/baike_word_ngram_300d', 
    'finetuned': '/Users/zongchang/Downloads/embedding_model/mydomain_word_ngram_300d',
    'stopword': '/Users/zongchang/OneDrive/可泛化知识计算引擎的副本/code/get_phrase/stopwords/stopwords_cn.txt',
    'stopword_open': '/Users/zongchang/OneDrive/可泛化知识计算引擎的副本/code/get_phrase/stopwords/hit_stopwords.txt', 
    'params': {
        'MIN_SCORE': 0.0,
        'SIM_SCORE': 0.0,
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
executor = 'general'
model = 'keyphrase-rake-topic'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)
text_ner = executor(dataset=dataset, model=OpenKSModel.get_module(platform, model), args=args)
text_ner.run()
print("-----------------------------------------------")
