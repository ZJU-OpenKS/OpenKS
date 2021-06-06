import numpy as np
import json, copy, sys, os


if __name__ == "__main__":
  assert(len(sys.argv) == 2), \
    'Please specify coco annotation directory.'
  coco_annotation_dir  = sys.argv[1]

  this_dir = os.path.dirname(__file__)
  dir_name = os.path.join(this_dir, 'data') 
  vcoco_annotation_dir = dir_name

  print("%s, %s"%(coco_annotation_dir, vcoco_annotation_dir))
  
  # First merge annotations from train and val
  # Load the train and val annotations
  json_train_file = '{:s}/instances_{:s}.json'.format(coco_annotation_dir, 'train2014')
  print("Loading training annotations from %s"%(format(json_train_file)))
  json_train = json.load(open(json_train_file, 'r'))

  json_val_file = '{:s}/instances_{:s}.json'.format(coco_annotation_dir, 'val2014')
  print("Loading validating annotations from %s"%(format(json_val_file)))
  json_val = json.load(open(json_val_file, 'r'))

  # Copy and sanity check
  assert(json_train['info']       == json_val['info'])
  assert(json_train['licenses']   == json_val['licenses'])
  assert(json_train['categories'] == json_val['categories'])
  
  json_all = json_train
  json_all['images']      = json_train['images'] + json_val['images'];
  json_all['annotations'] = json_train['annotations'] + json_val['annotations'];

  # write out collected trainval annotations to a single file
  json_trainval_file = '{:s}/instances_{:s}.json'.format(coco_annotation_dir, 'trainval2014')
  with open(json_trainval_file, 'w') as outfile:
    json.dump(json_all, outfile)
  del json_train 
  del json_val 
  del json_all

  # Second, selct annotations needed for V-COCO
  json_trainval = json.load(open('{:s}/instances_{:s}.json'.format(coco_annotation_dir, 'trainval2014'), 'r'))

  vcoco_imlist = np.loadtxt(os.path.join(vcoco_annotation_dir, 'splits', 'vcoco_all.ids'))[:,np.newaxis]

  # select images that we need
  coco_imlist = [j_i['id'] for j_i in json_trainval['images']]
  coco_imlist = np.array(coco_imlist)[:,np.newaxis]
  in_vcoco = []
  for i in range(len(coco_imlist)):
    if np.any(coco_imlist[i] == vcoco_imlist):
      in_vcoco.append(i)
  j_images = [json_trainval['images'][ind] for ind in in_vcoco] 

  # select annotations that we need
  coco_imlist = [j_i['image_id'] for j_i in json_trainval['annotations']]
  coco_imlist = np.array(coco_imlist)[:,np.newaxis]
  in_vcoco = []
  for i in range(len(coco_imlist)):
    if np.any(coco_imlist[i] == vcoco_imlist):
      in_vcoco.append(i)
  j_annotations = [json_trainval['annotations'][ind] for ind in in_vcoco] 

  json_trainval['annotations'] = j_annotations
  json_trainval['images'] = j_images

  vcoco = os.path.join(vcoco_annotation_dir, 'instances_vcoco_all_2014.json')
  print("Writing COCO annotations needed for V-COCO to %s."%(format(vcoco)))
  with open(vcoco, 'wt') as f:
    json.dump(json_trainval, f)
