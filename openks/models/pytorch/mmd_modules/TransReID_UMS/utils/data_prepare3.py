
import glob
import re
import os
import os.path as osp

list_path = "/root/reId/data/XXX/clean_data.txt"
save_train_path = "/root/reId/data/XXX/train_data.txt"
save_test_path = "/root/reId/data/XXX/test_data.txt"
save_query_path = "/root/reId/data/XXX/query_data.txt"
save_gallery_path = "/root/reId/data/XXX/gallery_data.txt"

with open(list_path, 'r') as txt:
    lines = txt.readlines()
img_path_train_list = []
pid_train_list = []
img_path_test_list = []
pid_test_list = []
img_path_query_list = []
pid_query_list = []
img_path_gallery_list = []
pid_gallery_list = []
pid_now = 0
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(' ')
    pid = pid#int(pid)  # no need to relabel
    camid = 0
    pid_num = int(pid)
    if pid_num < 47146:
        img_path_train_list.append(img_path)
        pid_train_list.append(pid)
    else:
        img_path_test_list.append(img_path)
        pid_test_list.append(pid)
        if pid_now != pid_num:
            pid_now = pid_num
            img_path_query_list.append(img_path)
            pid_query_list.append(pid)
        else:
            img_path_gallery_list.append(img_path)
            pid_gallery_list.append(pid)

print("##########################")

with open(save_train_path, "w") as f:
    for i in range(len(img_path_train_list)):
        out = img_path_train_list[i] + " " + pid_train_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close
with open(save_test_path, "w") as f:
    for i in range(len(img_path_test_list)):
        out = img_path_test_list[i] + " " + pid_test_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close
with open(save_query_path, "w") as f:
    for i in range(len(img_path_query_list)):
        out = img_path_query_list[i] + " " + pid_query_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close
with open(save_gallery_path, "w") as f:
    for i in range(len(img_path_gallery_list)):
        out = img_path_gallery_list[i] + " " + pid_gallery_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close

print("done!")