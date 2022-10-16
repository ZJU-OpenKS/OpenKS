# from openks.loaders import loader_config, SourceType, FileType, Loader
import sys
sys.path.append("..")
from openks.models import OpenKSModel

# 列出已加载模型
OpenKSModel.list_modules()

# 算法模型选择配置
args = {
        #"--gpu": 3,
        "--config_file": ".mmd_modules/Person_Relation/configs/MSMT17/vit_transreid_stride.yml",
    }
platform = 'PyTorch'
executor = 'Person_Relation'
model = 'pytorch-person_relation'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
executor = OpenKSModel.get_module(platform, executor)

text_ner = executor(args=args)
text_ner.run(mode="train")
#text_ner.run(mode="eval")
#text_ner.run(mode="visualize")
print("-----------------------------------------------")
