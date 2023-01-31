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
        #"--folder": "2021-12-23_13-42-46",  #eval&visualize
        #"--repeat": 1,  #ground eval
        #"--eval_reference": True,  #ground eval
        #"--no_nms": True,  #ground eval
        #"--force": True,  #ground eval
        #"--eval_caption": True,  #caption eval&visualize
        #"--min_iou": 0.5,  #caption eval
        #"--visualize_ground": True,  #ground visualize
        #"--visualize_caption": True,  #caption visualize
    }
platform = 'PyTorch'
executor = '3DJCG'
model = 'pytorch-3DJCGVisionLanguage'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)

text_ner = executor(args=args)
text_ner.run(mode="train")
#text_ner.run(mode="eval")
#text_ner.run(mode="visualize")
print("-----------------------------------------------")
