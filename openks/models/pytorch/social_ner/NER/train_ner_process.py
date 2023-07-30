# 将train_ner.jsonl数据改写为输入数据的格式
# 划分训练集测试集

import os

import jsonlines
from config import DATA_Config


def _read_jsonl(path):
    texts = []
    labels = []

    # 读入文件 获取标签序列
    with open(path, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            # {"text": "（作者署名：石江月）",
            # "words": ["（", "作者", "署名", "：", "石江", "月", "）"],
            # "entities": [{"text": "作者", "type": "job", "offset": 1, "length": 2}, {"text": "石江月", "type": "per", "offset": 6, "length": 3}]}
            text = item["text"]
            entities = item["entities"]
            label = ["O"] * len(text)
            for each_entity in entities:
                start_pos = each_entity["offset"]
                end_pos = each_entity["offset"] + each_entity["length"]
                type = each_entity["type"]
                label[start_pos] = "B-" + type
                label[start_pos + 1 : end_pos] = ["I-" + type] * (
                    each_entity["length"] - 1
                )
            # 至此完成了一句话的label序列
            assert len(text) == len(label)
            texts.append(text)
            labels.append(label)

    # 写入文件
    train_num = int((len(texts) / 10) * DATA_Config.train_dev_test[0])
    dev_num = train_num

    # 这里将训练集和验证集保持一致
    with open(DATA_Config.TRAIN_PATH, "w", encoding="utf-8") as f:
        for i in range(train_num):
            text = texts[i]
            label = labels[i]  # list
            for j in range(len(text)):
                f.write(text[j] + "\t" + label[j] + "\n")
            f.write("\n")

    with open(DATA_Config.DEV_PATH, "w", encoding="utf-8") as f:
        for i in range(train_num):
            text = texts[i]
            label = labels[i]  # list
            for j in range(len(text)):
                f.write(text[j] + "\t" + label[j] + "\n")
            f.write("\n")

    # 写测试集
    with open(DATA_Config.TEST_PATH, "w", encoding="utf-8") as f:
        for i in range(train_num, len(texts)):
            text = texts[i]
            label = labels[i]  # list
            for j in range(len(text)):
                f.write(text[j] + "\t" + label[j] + "\n")
            f.write("\n")


if __name__ == "__main__":
    _read_jsonl(os.path.join(DATA_Config.DATA_PATH, "dataset.jsonl"))
