from openks.models import OpenKSModel

platform = 'Paddle'
executor = 'HypernymExtract'
model = 'HypernymExtract'
print("根据配置，使用 {} 框架，{} 执行器训练 {} 模型。".format(platform, executor, model))
print("-----------------------------------------------")
# 模型训练
entity = '苹果'
executor = OpenKSModel.get_module(platform, executor)
hypernym_extract = executor()
res = hypernym_extract.entity2hyper_lst(entity)
print(res)

print("-----------------------------------------------")
