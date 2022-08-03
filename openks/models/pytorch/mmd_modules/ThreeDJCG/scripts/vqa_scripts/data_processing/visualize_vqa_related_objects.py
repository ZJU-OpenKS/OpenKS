import numpy as np
import json
import sys
import os

import process   # re-run it

sys.path.append(os.path.join(os.getcwd()))

# from lib.config_vqa import CONF
from core_vision_utils import output_bounding_box

base_path = ""  # no use
# base_path = CONF.PATH.SCANNET_DATA
vqa_path = 'data\\ScanVQA_train.json'
output_path = 'output_vqa'

vqa = json.load(open(vqa_path, 'r'))

rels = ['related_object(type 1)', 'related_object(type 2)', 'related_object(type 3)', 'related_object(type 4)']

np.random.seed(0)
color = [np.random.rand(3), np.random.rand(3), np.random.rand(3), [1,1,0]]
# np.random.rand(3)

def save_related_objects(scene_id, data):
    print('processing', len(data), 'object in scene', scene_id)
    # scene_id = data[0]["scene_id"]
    save_path = os.path.join(output_path, scene_id)
    os.makedirs(save_path, exist_ok=True)

    base_data_npy = os.path.join(save_path, 'bboxes.npy')
    # base_data_npy = os.path.join(base_path, scene_id) + "_aligned_bbox.npy"
    target_data_npy = os.path.join('data\\bboxes', scene_id+'_bboxes.npy')

    if os.path.exists(target_data_npy):
        bbox = np.load(target_data_npy)
    else:
        print('Move Npy File From Dataset')
        bbox = np.load(base_data_npy)
        np.save(target_data_npy, bbox)

    for idx, qa in enumerate(data):
        cap = '[' + str(idx) + ']' + '_(Q)' + qa['question'] + '(A)' + qa['answer']
        cap = cap.replace(' ', '_')
        cap = cap.replace('?', '_')
        cap = cap.replace('.', '_')
        output_file = open(os.path.join(save_path, cap + '.obj'), 'w')
        # output_file = open(os.path.join(save_path, '1' + '.obj'), 'w')
        count = 0
        for rid, rel in enumerate(rels):
            for val in bbox:
                bbox_id = int(val[-1])
                # print(bbox_id, len(val), rid)
                if bbox_id in qa[rel]:
                    output_bounding_box(count, val[:6], color[rid], 1, output_file)
                    # output_bounding_box(count, val[:6], color[rid], 1, output_file, output_face=True, output_line=True)
                    count = count + 1
                # print(val)
        output_file.close()


# # copy bbox file
# scene_0 = open('data/scannet/meta_data/scannetv2_train.txt', 'r').readlines()
# scene_1 = open('data/scannet/meta_data/scannetv2_val.txt', 'r').readlines()
# scenes = scene_0 + scene_1
# scenes = [val.strip('\n') for val in scenes]
# for scene in scenes:
#     if '_00' not in scene:
#         continue
#     save_related_objects(scene, [])

scene_id = ''
scanvqa_tosave = []
for data in vqa:
    if scene_id != data["scene_id"]:
        if len(scanvqa_tosave) > 0:
            save_related_objects(scene_id, scanvqa_tosave)
        scene_id = data["scene_id"]
        scanvqa_tosave = []
    scanvqa_tosave.append(data)
save_related_objects(scene_id, scanvqa_tosave)
