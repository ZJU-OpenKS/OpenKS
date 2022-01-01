"""
    parse Scan2CAD data to ScanNet instances

    format:
        {
            "scene_id": {
                "object_id": ...,
                "rotation_matrix": ...
            }
        }
"""

import os
import sys
import json
import quaternion

import numpy as np

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF

SCAN2CAD = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "full_annotations.json")))
ALIGNED_CAD2INST = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "aligned_cad2inst_id.json")))

print("parsing...")
parsed = {}
for scan2cad_data in tqdm(SCAN2CAD):
    scene_id = scan2cad_data["id_scan"]

    for cad_id, cad_data in enumerate(scan2cad_data["aligned_models"]):
        try:
            rotation_quaternion = np.quaternion(*cad_data["trs"]["rotation"])
            rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion).astype(np.float64)
            instance_id = ALIGNED_CAD2INST[scene_id][str(cad_id)]
            
            if scene_id not in parsed: parsed[scene_id] = {}
            parsed[scene_id][instance_id] = rotation_matrix.tolist()
        except KeyError:
            pass

print("number of scenes: {}".format(len(parsed.keys())))

# store
with open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json"), "w") as f:
    json.dump(parsed, f, indent=4)

print("done!")