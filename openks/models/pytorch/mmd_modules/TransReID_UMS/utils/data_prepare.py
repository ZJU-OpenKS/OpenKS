
import glob
import re

import os.path as osp

list_path = "/root/reId/data/XXX/XXX_ID_calibration_body_V1_0826_float.txt"
save_path = "/root/reId/data/XXX/data.txt"

with open(list_path, 'r') as txt:
    lines = txt.readlines()
img_path_list = []
pid_list = []
sum = 0
for img_idx, img_info in enumerate(lines):
    img_path, _, pid = img_info.split(' ')
    pid = pid#int(pid)  # no need to relabel
    camid = 0
    _, img_path = img_path.split('\\')

    #print("img_path", img_path)
    #print("pid", pid)
    img_path_list.append(img_path)
    pid_list.append(pid)
    sum += 1

print("sum", sum)
with open(save_path, "w") as f:
    for i in range(sum):
        out = img_path_list[i] + " " + pid_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close