
import glob
import re
import os
import os.path as osp
import time
import random

begin_time=(2022,1,1,0,0,0,0,0,0)              #设置开始日期时间元组（1976-01-01 00：00：00）
end_time=(2022,1,1,23,59,59,0,0,0)    #设置结束日期时间元组（1990-12-31 23：59：59）

list_path = "/root/reId/data/MSMT17_V1/list_train.txt"
save_gallery_path = "/root/reId/data/MSMT17_V1/train_locus.txt"

start=time.mktime(begin_time)    #生成开始时间戳
end=time.mktime(end_time)      #生成结束时间戳

with open(list_path, 'r') as txt:
    lines = txt.readlines()
pid_gallery_list = []
time_gallery_list = []
cid_gallery_list = []
MAX_LEN = 20
for img_idx, img_info in enumerate(lines):
    #img_path, pid = img_info.split(' ')
    pid = str(random.randint(0, 10000))
    locus_time_list = []
    locus_cid_list = []
    num = random.randint(10, 30)
    for i in range(MAX_LEN):
        if i >= num:
            cid = "0"
        else:
            cid = str(random.randint(1, 50))
        locus_cid_list.append(cid)
        random_time = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
        date_touple = time.localtime(random_time)  # 将时间戳生成时间元组
        date = time.strftime("%Y-%m-%d:%H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
        locus_time_list.append(date)
    #locus_time_list.sort()
    pid_gallery_list.append(pid)
    cid_gallery_list.append(locus_cid_list)
    time_gallery_list.append(locus_time_list)

print("##########################")

with open(save_gallery_path, "w") as f:
    for i in range(len(pid_gallery_list)):
        out = pid_gallery_list[i] + ";"
        f.write(out)
        for j in range(MAX_LEN):
            out = time_gallery_list[i][j] + "_" + cid_gallery_list[i][j] + " "
            f.write(out)
        f.write("\n")

print("done!")
