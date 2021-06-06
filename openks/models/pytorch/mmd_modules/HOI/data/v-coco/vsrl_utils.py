# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2016, Saurabh Gupta
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from pycocotools.coco import COCO
import os, json

coco = []

def get_data_dir():
  this_dir = os.path.dirname(__file__)
  dir_name = os.path.join(this_dir, 'data') 
  return dir_name

def load_coco(dir_name=None):
  global coco 
  if dir_name is None:
    dir_name = get_data_dir()
  if coco == []:
    coco = COCO(os.path.join(dir_name, 'instances_vcoco_all_2014.json'))
  return coco

def load_vcoco(imset, dir_name=None):
  if dir_name is None:
    dir_name = get_data_dir()
  with open(os.path.join(dir_name, 'vcoco', imset + '.json'), 'rt') as f:
    vsrl_data = json.load(f)
  vsrl_data = unicode_to_str(vsrl_data)
  for i in range(len(vsrl_data)):
    vsrl_data[i]['role_object_id'] = \
      np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']),-1)).T
    for j in ['ann_id', 'label', 'image_id']:
      vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1,1))
  return vsrl_data

def coco_ann_id_to_image_id(ann_id, coco):
  ann = coco.loadAnns(ann_id);
  ann_id_r = [a['id'] for a in ann]
  image_id_r = [a['image_id'] for a in ann]
  assert ann_id_r == ann_id, \
    'oops coco returned different ann_id''s in different order'
  return image_id_r


# Attach relevant boxes
def all_relevant_objects(vsrl_data_all, coco):
  vsrl_relevant_objects = []
  for i in xrange(len(vsrl_data_all)):
      v_i = vsrl_data_all[i]
      bbox = []; image_id = [];
      num_roles = len(v_i['role_name'])
      for j in range(num_roles):
          # print v_i['action_name'], v_i['include'][j]; 
          bbox_j = np.NaN*np.ones((0,4*num_roles)); 
          image_id_j = np.NaN*np.ones((0,1))
          if v_i['include'][j] != []:
              cat_ids = coco.getCatIds(catNms=v_i['include'][j])
              ann_list = coco.getAnnIds(imgIds=np.unique(v_i['image_id']*1).tolist(), 
                                        iscrowd=False, catIds=cat_ids)
              anns = coco.loadAnns(ann_list)
              bbox_j = np.NaN*np.zeros((len(anns), 4*num_roles))
              bbox_j[:,4*j:4*j+4] = xyhw_to_xyxy(np.vstack([np.array(a['bbox']).reshape((1,4)) for a in anns]))
              image_id_j = np.array(coco_ann_id_to_image_id(ann_list, coco)).reshape(-1,1)
          bbox.append(bbox_j)
          image_id.append(image_id_j)
      image_id = np.concatenate(image_id, axis=0)
      bbox = np.concatenate(bbox, axis=0)
      vsrl_relevant_objects.append({'image_id': image_id, 'bbox': bbox})
  return vsrl_relevant_objects

def attach_unlabelled(vsrl_data, coco):
  """
  def vsrl_data = attach_unlabelled(vsrl_data, coco)
  """
  anns = coco.loadAnns(
    coco.getAnnIds(imgIds=np.unique(vsrl_data['image_id']).tolist(), 
      iscrowd=False, catIds=1));
  ann_id                      = [a['id'] for a in anns]
  hard_ann_id                 = list(set(ann_id) - set(vsrl_data['ann_id'].ravel().tolist()))
  hard_image_id               = coco_ann_id_to_image_id(hard_ann_id, coco);
  
  vsrl_data['image_id']       = np.vstack((vsrl_data['image_id'], 
                                np.array(hard_image_id).reshape((-1,1))))
  vsrl_data['ann_id']         = np.vstack((vsrl_data['ann_id'],   
                                np.array(hard_ann_id).reshape((-1,1))))
  vsrl_data['role_object_id'] = np.vstack((vsrl_data['role_object_id'], 
                                np.zeros((len(hard_image_id), vsrl_data['role_object_id'].shape[1]))))
  vsrl_data['role_object_id'][vsrl_data['label'].shape[0]:,0] = np.array(hard_ann_id).reshape((-1))
  vsrl_data['label']          = np.vstack((vsrl_data['label'], 
                                -1*np.ones((len(hard_image_id), vsrl_data['label'].shape[1]))))
  return vsrl_data

def remove_negative(vsrl_data):
  """
  def vsrl_data = remove_negative(vsrl_data)
  Remove things that are labelled as a negative
  """
  to_keep = vsrl_data['label'] != 0 
  for i in vsrl_data.keys():
    if type(vsrl_data[i]) == np.ndarray:
      vsrl_data[i] = vsrl_data[i][to_keep, :]
  return vsrl_data

def xyhw_to_xyxy(bbox):
  out = bbox.copy()
  out[:, [2, 3]] = bbox[:, [0,1]] + bbox[:, [2,3]];
  return out

def attach_gt_boxes(vsrl_data, coco):
  """
  def vsrl_data = attach_gt_boxes(vsrl_data, coco)
  """
  anns = coco.loadAnns(vsrl_data['ann_id'].ravel().tolist());
  bbox = np.vstack([np.array(a['bbox']).reshape((1,4)) for a in anns])

  vsrl_data['bbox'] = xyhw_to_xyxy(bbox)
  vsrl_data['role_bbox'] = \
    np.nan*np.zeros((vsrl_data['role_object_id'].shape[0], \
      4*vsrl_data['role_object_id'].shape[1]), dtype=np.float)
  
  # Get boxes for the role objects
  for i in range(vsrl_data['role_object_id'].shape[1]):
    has_role = np.where(vsrl_data['role_object_id'][:,i] > 0)[0]
    if has_role.size > 0:
      anns = coco.loadAnns(vsrl_data['role_object_id'][has_role, i].ravel().tolist());
      bbox = np.vstack([np.array(a['bbox']).reshape((1,4)) for a in anns])
      bbox = xyhw_to_xyxy(bbox)
      vsrl_data['role_bbox'][has_role, 4*i:4*(i+1)] = bbox;
  return vsrl_data

def unicode_to_str(input):
  if isinstance(input, dict):
    return {unicode_to_str(key):unicode_to_str(value) for key,value in input.iteritems()}
  elif isinstance(input, list):
    return [unicode_to_str(element) for element in input]
  elif isinstance(input, unicode):
    return input.encode('utf-8')
  else:
    return input
