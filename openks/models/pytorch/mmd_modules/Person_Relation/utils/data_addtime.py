
import glob
import re
import os
import os.path as osp
import time
import random
import numpy as np

begin_time=(2022,1,1,0,0,0,0,0,0)              #设置开始日期时间元组（1976-01-01 00：00：00）
middle_time=(2022,1,1,12,0,0,0,0,0)
end_time=(2022,1,1,23,59,59,0,0,0)    #设置结束日期时间元组（1990-12-31 23：59：59）
# train val query gallery
list_path = "/data5/caidaigang/data/MSMT17_V1/list_query.txt"
save_gallery_path = "/data5/caidaigang/data/MSMT17_V1/list_query_withtime.txt"

start=time.mktime(begin_time)    #生成开始时间戳
middle=time.mktime(middle_time)
end=time.mktime(end_time)      #生成结束时间戳
print(end-start)
"""
# add time
with open(list_path, 'r') as txt:
    lines = txt.readlines()
img_path_gallery_list = []
pid_gallery_list = []
time_gallery_list = []
pid_now = -1
pid_sum = -1
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(' ')
    pid = pid#int(pid)  # no need to relabel
    img_path_gallery_list.append(img_path)
    pid_gallery_list.append(pid)
    random_time = random.randint(start, end)    #在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(random_time)  # 将时间戳生成时间元组
    date = time.strftime("%Y-%m-%d:%H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
    time_gallery_list.append(date)

print("##########################")

with open(save_gallery_path, "w") as f:
    for i in range(len(img_path_gallery_list)):
        out = img_path_gallery_list[i] + " " + time_gallery_list[i] + " " + pid_gallery_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close
"""

# change time & add x,y（经纬度）
with open(list_path, 'r') as txt:
    lines = txt.readlines()
img_path_gallery_list = []
pid_gallery_list = []
time_gallery_list = []
x_gallery_list = []
y_gallery_list = []
locus_num = -1
pid_now = -1
time_now = 0
x_now = 0.
y_now = 0.
base_time = np.zeros(shape=[1000, 20], dtype=np.int64)
base_x = np.zeros(shape=[1000, 20], dtype=np.float64)
base_y = np.zeros(shape=[1000, 20], dtype=np.float64)
for img_idx, img_info in enumerate(lines):
    img_path, pid = img_info.split(' ')
    pid = pid#int(pid)  # no need to relabel
    pid_num = int(pid)
    if pid_num < 1000:
        if pid_now != pid_num:
            random_time = random.randint(start, middle)  # 在开始和结束时间戳中随机取出一个
            date_touple = time.localtime(random_time)  # 将时间戳生成时间元组
            date = time.strftime("%Y-%m-%d:%H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
            #x = random.uniform(118.35, 120.5)
            #y = random.uniform(29.18, 30.55)
            x = random.uniform(120.0, 120.5)
            y = random.uniform(30.0, 30.5)
            pid_now = pid_num
            time_now = random_time
            x_now = x
            y_now = y
            locus_num = 0
        else:
            random_time_add = random.randint(60, 600)
            x_add = random.uniform(-0.0050, 0.0050)
            y_add = random.uniform(-0.0050, 0.0050)
            random_time = time_now + random_time_add
            date_touple = time.localtime(random_time)  # 将时间戳生成时间元组
            date = time.strftime("%Y-%m-%d:%H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
            x = x_now + x_add
            y = y_now + y_add
            time_now = random_time
            x_now = x
            y_now = y
            locus_num += 1
        if locus_num < 20:
            base_time[pid_num, locus_num] = time_now
            base_x[pid_num, locus_num] = x_now
            base_y[pid_num, locus_num] = y_now
    else:
        circle_num = pid_num % 1000
        if pid_now != pid_num:
            locus_num = 0
        else:
            locus_num += 1
        if locus_num < 20:
            x = base_x[circle_num, locus_num]
            y = base_y[circle_num, locus_num]
            random_time = base_time[circle_num, locus_num]
            if random_time < 1.:
                x = x_now
                y = y_now
                random_time = time_now
        else:
            x = x_now
            y = y_now
            random_time = time_now
        if pid_num < 2000:#locus_num % 2 == 0:
            x_add = random.uniform(-0.0005, 0.0005)
            y_add = random.uniform(-0.0005, 0.0005)
            random_time_add = random.randint(-60, 60)
        else:
            x_add = random.uniform(-0.0020, 0.0020)
            y_add = random.uniform(-0.0020, 0.0020)
            random_time_add = random.randint(-300, 300)
        if pid_num == 1000:
            print("x",x)
            print("x_add", x_add)
            print("new_x", x + x_add)
            print("y", y)
            print("y_add", y_add)
            print("new_y", y + y_add)
        x = x + x_add
        y = y + y_add
        random_time = random_time + random_time_add
        date_touple = time.localtime(random_time)  # 将时间戳生成时间元组
        date = time.strftime("%Y-%m-%d:%H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）

        pid_now = pid_num
        time_now = random_time
        x_now = x
        y_now = y

    img_path_gallery_list.append(img_path)
    pid_gallery_list.append(pid)
    time_gallery_list.append(date)
    x = round(x, 5)
    y = round(y, 5)
    x_gallery_list.append(str(x))
    y_gallery_list.append(str(y))

print("##########################")

with open(save_gallery_path, "w") as f:
    for i in range(len(img_path_gallery_list)):
        out = img_path_gallery_list[i] + " " + time_gallery_list[i] + " " + x_gallery_list[i] + " " + y_gallery_list[i] + " " + pid_gallery_list[i]
        f.write(out)  # 自带文件关闭功能，不需要再写f.close

print("done!")
