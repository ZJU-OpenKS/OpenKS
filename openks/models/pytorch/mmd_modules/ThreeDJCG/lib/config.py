import os
import sys
from easydict import EasyDict

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CONF = EasyDict()

# path
CONF.PATH = EasyDict()
#CONF.PATH.BASE = "/rhome/dchen/Scan2CapRelease/" # TODO: change this
CONF.PATH.BASE = ROOT_DIR
#CONF.PATH.CLUSTER = "/cluster/balrog/dchen/Scan2CapRelease/" # TODO: change this
CONF.PATH.CLUSTER = "/data5/caidaigang/scanrefer/data/Scan2CapRelease/"
# CONF.PATH.BASE = "/home/davech2y/Scan2CapExp/" # TODO: change this
# CONF.PATH.CLUSTER = "/home/davech2y/Scan2CapExp/" # TODO: change this
#CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
#CONF.PATH.DATA = "/data4/caidaigang/caidaigang/model/scanrefer/data/"
CONF.PATH.DATA = "/data5/caidaigang/scanrefer/data/"
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# # append to syspath
# for _, path in CONF.PATH.items():
#     sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# Scan2CAD
#CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset") # TODO change this
#CONF.PATH.SCAN2CAD = "/data4/caidaigang/caidaigang/data/scan2CAD"
CONF.PATH.SCAN2CAD = "/data5/caidaigang/data/scan2CAD"

# data
#CONF.SCANNET_DIR =  os.path.join(CONF.PATH.BASE, "/data4/caidaigang/caidaigang/model/scanrefer/data/scannet/scans") # TODO change this
CONF.SCANNET_DIR =  os.path.join(CONF.PATH.BASE, "/data5/caidaigang/scanrefer/data/scannet/scans") # TODO change this
#CONF.SCANNET_FRAMES_ROOT = "/data4/caidaigang/caidaigang/data/frames_square/" # TODO change this
CONF.SCANNET_FRAMES_ROOT = "/data5/caidaigang/scanrefer/data/frames_square/" # TODO change this
#CONF.PROJECTION = "/home/davech2y/multiview_projection_scanrefer" # TODO change this
#CONF.ENET_FEATURES_ROOT = "/data4/caidaigang/caidaigang/data/enet_features" # TODO change this
CONF.ENET_FEATURES_ROOT = "/data5/caidaigang/scanrefer/data/enet_features" # TODO change this
#CONF.SCANNET_DIR =  os.path.join(CONF.PATH.BASE, "/data5/caidaigang/scanrefer/data/scannet/scans") # TODO change this
#CONF.SCANNET_FRAMES_ROOT = "/data5/caidaigang/scanrefer/data/frames_square/" # TODO change this
#CONF.PROJECTION = "/home/davech2y/multiview_projection_scanrefer" # TODO change this
#CONF.ENET_FEATURES_ROOT = "/data5/caidaigang/scanrefer/data/enet_features" # TODO change this
CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
# CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
# CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")
#CONF.PATH.AXIS_ALIGNED_MESH = os.path.join(CONF.PATH.OUTPUT, "ScanNet_axis_aligned_mesh")
CONF.PATH.AXIS_ALIGNED_MESH = "/data5/caidaigang/model/Scan2Cap-main/outputs/ScanNet_axis_aligned_mesh"

# pretrained
CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")

# Pretrained features
CONF.PATH.GT_FEATURES = os.path.join(CONF.PATH.CLUSTER, "gt_{}_features") # dataset
# CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_features")
#CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_{}_predictions") # dataset
CONF.PATH.VOTENET_FEATURES = os.path.join(CONF.PATH.CLUSTER, "votenet_ScanRefer_predictions") # dataset
