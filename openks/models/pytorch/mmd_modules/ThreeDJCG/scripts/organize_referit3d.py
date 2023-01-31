import os
import sys
import json

import pandas as pd

from tqdm import tqdm
from ast import literal_eval

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF

RAW_REFERIT3D = os.path.join(CONF.PATH.DATA, "nr3d.csv")
PARSED_REFERIT3D = os.path.join(CONF.PATH.DATA, "nr3d_organized.json") # split
scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
PARSED_REFERIT3D_TRAIN = os.path.join(CONF.PATH.DATA, "nr3d_filtered_train.json") # split
PARSED_REFERIT3D_VAL = os.path.join(CONF.PATH.DATA, "nr3d_filtered_val.json") # split
train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))
print("parsing...")
organized = []
df = pd.read_csv(RAW_REFERIT3D)
df.tokens = df["tokens"].apply(literal_eval)
"""
for _, row in tqdm(df.iterrows()):
    entry = {}
    entry["scene_id"] = row["scan_id"]
    entry["object_id"] = str(row["target_id"])
    entry["object_name"] = row["instance_type"]
    entry["ann_id"] = str(row["assignmentid"])
    entry["description"] = row["utterance"].lower()
    entry["token"] = row["tokens"]

    scene_id = entry["scene_id"]
    object_id = entry["object_id"]
    ann_id = entry["ann_id"]

    # store
    if scene_id not in organized:
        organized[scene_id] = {}

    if object_id not in organized[scene_id]:
        organized[scene_id][object_id] = {}

    if ann_id not in organized[scene_id][object_id]:
        organized[scene_id][object_id][ann_id] = None

    organized[scene_id][object_id][ann_id] = entry

with open(PARSED_REFERIT3D, "w") as f:
    json.dump(organized, f, indent=4)
"""

for _, row in tqdm(df.iterrows()):
    entry = {}
    entry["scene_id"] = row["scan_id"]
    entry["object_id"] = str(row["target_id"])
    entry["object_name"] = row["instance_type"]
    entry["ann_id"] = str(row["assignmentid"])
    entry["description"] = row["utterance"].lower()
    entry["token"] = row["tokens"]

    scene_id = entry["scene_id"]
    object_id = entry["object_id"]
    ann_id = entry["ann_id"]

    # store
    # if scene_id not in organized:
    #     organized[scene_id] = {}
    #
    # if object_id not in organized[scene_id]:
    #     organized[scene_id][object_id] = {}
    #
    # if ann_id not in organized[scene_id][object_id]:
    #     organized[scene_id][object_id][ann_id] = None
    if scene_id in val_scene_list:
        organized.append(entry)

new_organized = sorted(organized, key=lambda e: e["scene_id"])

with open(PARSED_REFERIT3D_VAL, "w") as f:
    json.dump(new_organized, f, indent=4)

print("done!")