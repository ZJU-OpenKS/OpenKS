
import glob
import re
import os
import os.path as osp

list_path = "/root/reId/data/XXX/XXX_without_hat_and_mask_clean_v2.txt"
img_data = "/root/reId/data/XXX/XXX/"
save_path = "/root/reId/data/XXX/head_clean_data.txt"

with open(list_path, 'r') as txt:
    lines = txt.readlines()
img_path_list = []
pid_list = []
sum = 0
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(' ')
    pid = pid#int(pid)  # no need to relabel
    camid = 0
    new_img_path = osp.join(img_data, img_path)
    #print("img_path", img_path)
    #print("pid", pid)
    if os.path.exists(new_img_path):
        img_path_list.append(img_path)
        pid_list.append(pid)
        sum += 1

print("sum", sum)
with open(save_path, "w") as f:
    for i in range(sum):
        out = img_path_list[i] + " " + pid_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close