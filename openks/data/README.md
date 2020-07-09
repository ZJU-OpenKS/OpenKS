# OpenKS知识图谱输入数据格式定义

* 将各数据文件放置于一个目录下，可将数据集名称设置为目录的名称（MMD和MTD将以目录名作为数据集和图谱名称）
* 需包含一个schema.json文件，指明图谱数据中包含的实体类型、关系类型、实体类型上下位关系、关系类型所连接的实体类型、实体和关系类型所具有的属性等，如下面例子中，包含三种实体类型和两种关系类型，以及它们的属性（properties），其中expert拥有上级概念person，同时指出了company->applies->patent，和expert->invents->patent关系，且关系具有属性和方向。举例如下：
```
[
	{
		"type": "entity",
		"concept": "person",
		"properties": [
			{
				"name": "name",
				"range": "str"
			}
		]
	},
	{
		"type": "entity",
		"concept": "expert",
		"properties": [
			{
				"name": "name",
				"range": "str"
			},
			{
				"name": "birthday",
				"range": "date"
			}
		],
		"parent": "person"
	},
	{
		"type": "entity",
		"concept": "patent",
		"properties": [
			{
				"name": "title",
				"range": "str"
			},
			{
				"name": "abstract",
				"range": "str"
			}
		]
	},
	{
		"type": "entity",
		"concept": "company",
		"properties": [
			{
				"name": "name",
				"range": "str"
			},
			{
				"name": "location",
				"range": "str"
			}
		]
	},
	{
		"type": "relation",
		"concept": "applies",
		"properties": [
			{
				"name": "apply_date",
				"range": "date"
			}
		],
		"members": ["company", "patent"]
	},
	{
		"type": "relation",
		"concept": "invents",
		"properties": [
			{
				"name": "invent_date",
				"range": "date"
			},
			{
				"name": "contribution",
				"range": "float"
			}
		],
		"members": ["expert", "patent"]
	}

]
```
* 包含一个完整实体及属性文件，文件名为entities，每行为一个实体数据，其中第一列为ID，其余为实体属性，顺序为schema.json中properties的顺序（若没有属性则只有一列ID），列之间以\t隔开，这里不涉及三元组信息。举例如下：
```
1	xxx有限公司	xx省xx市xx区xx街道xxx号
2	.........	.........
......
n	基于xxx的xx系统	本系统xxxxxxx
n+1	.........	.........
......
```
* 包含一个完整三元组及关系属性文件，文件名为triples，每行为一个三元组数据，其中前三列为：实体1ID-关系名称-实体2ID 三元组，后面为此关系的属性信息如权重等，顺序为schema.json中properties的顺序（若没有属性则只有三元组信息），列之间以\t隔开，举例如下：
```
1	applies	n	2020-05-21
1	applies	n+1	2019-02-12
......
m	invents	n	2017-11-23	0.6
......
```