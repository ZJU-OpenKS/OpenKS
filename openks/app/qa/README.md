# 知识问答demo

### 支持基于知识库匹配方式的问答搜索流程，以提供的测试数据集为例，效果如下：
* ![ask_investor](../../../docs/pics/ask_investor.jpg)
* ![ask_investor](../../../docs/pics/ask_patent_num.jpg)
* ![ask_investor](../../../docs/pics/ask_patent.jpg)

### 支持基于知识图谱向量表示的问答计算流程（待实现）

### 若要实现一个自定义的问答功能需实现以下工作：
1. 按照OPENKS的输入数据格式构建知识图谱数据集并放入openks/data/中，可参考[此文档](https://github.com/ZJU-OpenKS/OpenKS/blob/master/openks/data/README.md)
2. 需自定义编写问题解析脚本类，该类需继承QuestionParser，实现其中定义的处理模块，如实体抽取、关系抽取、模板识别、问题类型等，并返回StrucQ结构化问题
3. 在主方法main.py中调用自定义解析类的parse方法和AnswerFetcher的fetch_by_matching方法，即可实现一个简单的知识库匹配式问答