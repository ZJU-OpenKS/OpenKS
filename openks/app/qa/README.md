# 知识问答demo

### 支持基于知识库匹配方式的问答搜索流程，以提供的测试数据集为例，效果如下：
#### 企业投融资与知识产权数据集（来自内部构建）
* ![ask_investor](../../../docs/pics/ask_investor.jpg)
* ![ask_investor](../../../docs/pics/ask_patent_num.jpg)
* ![ask_investor](../../../docs/pics/ask_patent.jpg)
#### 医疗健康数据集（来自：https://github.com/liuhuanyong/QASystemOnMedicalKG）
* ![ask_disease](../../../docs/pics/ask_disease1.jpg)
* ![ask_disease](../../../docs/pics/ask_disease2.jpg)

### 支持基于知识图谱向量表示的问答计算流程（待实现）

### 若要实现一个自定义的问答功能需实现以下工作：
1. 按照OPENKS的输入数据格式构建知识图谱数据集并放入openks/data/中，可参考[此文档](https://github.com/ZJU-OpenKS/OpenKS/blob/master/openks/data/README.md)
2. 需自定义编写问题解析脚本类，该类需继承QuestionParser，实现其中定义的处理模块，如实体抽取、关系抽取、模板识别、问题类型等，并通过parse函数组装各个模块并返回StrucQ结构化问题，可参考rule_parser.py中的解析示例
3. 在主方法main.py中声明问题解析类进行模型预加载，并调用自定义解析类的parse方法进行问题解析，之后调用AnswerFetcher的fetch_by_matching方法（直接匹配MTG图）或fetch_by_db_query方法（执行数据库查询）或fetch_by_similarity方法（向量相似度计算），即可实现一个简单的知识库匹配式问答