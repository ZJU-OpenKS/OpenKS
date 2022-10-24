
import scipy.io
import glob
import re
import shutil
import os
import os.path as osp

anno_path = "/root/reId/data/HICO-DET/anno.mat"
print("anno_path", anno_path)
save_text_path = "/root/reId/data/HICO-DET/person_test.txt"
img_path = "/root/reId/data/HICO-DET/hico_20160224_det/images/test2015/"
img_save_path = "/root/reId/data/HICO-DET/hico_person/test/"


data = scipy.io.loadmat(anno_path)  # 读取mat文件
print(data.keys())   # 查看mat文件中的所有变量
print("__header__",data['__header__'])
print("__version__",data['__version__'])
print("__globals__",data['__globals__'])
print("list_action",data['list_action'].shape)
#print("list_action",data['list_action'][0])
print("anno_train",data['anno_train'].shape)
pic_list = []
anno_list = []
for i in range(data['anno_test'].shape[1]):
    j = 160
    if data['anno_test'][j, i] == 0 or data['anno_test'][j, i] == 1 or data['anno_test'][j, i] == -1:
        #print(str(i) + ":" + str(j + 1) + ":")
        #print(data['anno_train'][j:j + 10, i])
        id = str(i+1)
        s = id.zfill(8)
        pic = "HICO_test2015_" + s + ".jpg"
        pic_list.append(pic)
        anno_list.append(data['anno_test'][j:j + 10, i])
        print(os.path.join(img_path, pic))
        if os.path.exists(os.path.join(img_path, pic)):
            print("save pic")
            shutil.copy(os.path.join(img_path, pic), os.path.join(img_save_path, pic))
    """
    for j in range(600):
        if data['anno_train'][j, i] == 0 or data['anno_train'][j, i] == 1 or data['anno_train'][j, i] == -1:
            print(str(i)+":"+str(j+1)+":")
            print(data['anno_train'][j:j+20, i])
            break
    """
#print("anno_train",data['anno_train'][:, 0])
print("anno_test",data['anno_test'].shape)
print("list_train",data['list_train'].shape)
#print("list_train",data['list_train'][0:5])
print("list_test",data['list_test'].shape)
#print("list_test",data['list_test'][0:5])

"""
print("################################################")
anno_bbox_path = "/root/reId/data/HICO-DET/anno_bbox.mat"
print("anno_bbox_path", anno_bbox_path)
data = scipy.io.loadmat(anno_bbox_path)  # 读取mat文件
print(data.keys())   # 查看mat文件中的所有变量
print("__header__",data['__header__'])
print("__version__",data['__version__'])
print("__globals__",data['__globals__'])
print("bbox_train",data['bbox_train'].shape)
#print("bbox_train",data['bbox_train'][0, 0])
print("bbox_test",data['bbox_test'].shape)
print("list_action",data['list_action'].shape)
#print("list_action",data['list_action'][:2])
"""

with open(save_text_path, "w") as f:
    for i in range(len(pic_list)):
        out = pic_list[i] + " " + str(anno_list[i]) + "\n"
        f.write(out)  # 自带文件关闭功能，不需要再写f.close



#matrix1 = data['matrix1']
#matrix2 = data['matrix2']
#print(matrix1)
#print(matrix2)
#scipy.io.savemat('matData2.mat',{'matrix1':matrix1, 'matrix2':matrix2})  # 写入mat文件