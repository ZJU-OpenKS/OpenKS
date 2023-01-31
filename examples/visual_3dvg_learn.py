# from openks.loaders import loader_config, SourceType, FileType, Loader
import sys
from openks.models import OpenKSModel

# 列出已加载模型
OpenKSModel.list_modules()

# 算法模型选择配置
args = {
        #"--gpu": 3,
        "--use_normal": True,
        "--use_multiview": True,
        "--coslr": True,
        #"--lang_num_max": 1,
        #######eval#######
        #"--folder": "2021-12-23_13-42-46",  #eval
        #"--reference": True,  #eval
        #"--no_nms": True,  #eval
        #"--force": True,  #eval
        #"--repeat": 1,  #eval
    }
platform = 'PyTorch'
executor = '3DVG'
model = 'pytorch-3DVisualGrounding'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)

text_ner = executor(args=args)
text_ner.run(mode="train")
#text_ner.run(mode="eval")
#text_ner.run(mode="visualize")
print("-----------------------------------------------")
