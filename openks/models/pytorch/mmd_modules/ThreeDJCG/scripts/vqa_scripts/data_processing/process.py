import os
import sys
import json

import xlrd
import re

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

OUTPUT_FILE = os.path.join("./data/", "ScanVQA_train.json")

organized = []
print("parsing...")
scene_count = 0
scene_ids = []
# 在本地文件夹下建立一个file文件; 会遍历那个file文件夹
# print('titles not right是文件第一行长度不够长')
# print('float no attribute lower是在回答时,回答了1234...的阿拉伯数字')
number = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen ' \
         'seventeen eighteen nineteen twenty'.split(' ')
# print(number)
for root, dirs, files in os.walk(".\\Files", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        if '.xlsx' not in filename:
            continue
        try:
            scene_count += 1
            scene_ids.append([int(name[6:9]), 0])
            now_scene_id = name[0:12]
            # if '0000' not in filename:  # 测试
            #     continue
            # 读取工作表
            excel = xlrd.open_workbook(filename)
            # table = excel.sheet_by_name('工作表1')
            table = excel.sheets()[0]
            entries = []
            rows = table.nrows  # 获取行数
            cols = table.ncols  # 获取列数; 应该是10
            if cols != 10:
                print(filename, 'cols != 10; titles are not right (title长度不对)')
            # print(rows, cols, 'in', filename)
            # print(table.row_values(0))  # titles;
            titles = ['scene_id', 'question_type', 'question', 'answer',
                      'related_object(type 1)', 'related_object(type 2)',
                      'related_object(type 3)', 'related_object(type 4)',
                      'rank(filter)', 'issue(filter)']

            scene_id_not_right = False  # 检查scene_id是否正确，如果错误只输出一次
            for i in range(1, rows):
                line = table.row_values(i)
                if "".join([str(item) for item in line[1:]]) == "":  # excel中存在一些别的东西
                    print("文件", filename, "处理错误, 存在空行", i+1)
                    break
                if line[0] != now_scene_id:
                    if scene_id_not_right:
                        print("文件", filename, "第", i+1, "行 scene id 错误", "现为", line[0], "应为", now_scene_id)
                        scene_id_not_right = True
                    line[0] = now_scene_id
                entry = {}
                # break
                qa_str_not_right = False
                for j, name in enumerate(titles):
                    if j >= cols:
                        value_str = ""
                    else:
                        value_str = line[j]
                    if 'related_object' in name:
                        if isinstance(value_str, float):
                            value_str = str(int(value_str))
                        value_str = re.split('[，,.。？? ]', value_str)
                        value_str = [int(value) for value in value_str if value != ""]
                    if name in ['question', 'answer']:
                        if isinstance(value_str, float):  # 回答了阿拉伯数字; do not remove
                            value_str = str(int(value_str)).lower()
                            print(now_scene_id, 'line {}'.format(i+1), '回答了阿拉伯数字, 已处理')
                        value_str = value_str.lower()
                        if value_str == "":
                            qa_str_not_right = True
                    entry[name] = value_str
                    # print(line)
                if qa_str_not_right:
                    print(filename, "line {} question或answer为空".format(i+1))
                    continue
                entries.append(entry)
            # print('file', filename, len(entries), 'entries')
            organized.extend(entries)
            scene_ids[-1][1] = len(entries)
            # break
        except Exception as e:
            print(filename, e, '存在问题; throw an Exception')

    for name in dirs:
        # print(os.path.join(root, name))
        pass

with open(OUTPUT_FILE, "w") as f:
    json.dump(organized, f, indent=4)

print('{} QAs in {} scenes processed'.format(len(organized), scene_count))
for i in range(0, 706, 100):
    print('{} scenes in range({}, {}), QA number {}'.format(
        len([x for x in scene_ids if (i <= x[0] < i + 100)]), i, i + 100,
        sum([x[1] for x in scene_ids if (i <= x[0] < i + 100)])))
print("done!")