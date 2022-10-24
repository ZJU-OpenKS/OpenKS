from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import pickle
from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch.utils.data as data
import torch

def assert_eq(real, expected):
	assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

class hcstvg(data.Dataset):
	num_classes = 1
	default_resolution = [512, 512]
	mean = np.array([0.485, 0.456, 0.406],
									 dtype=np.float32).reshape(1, 1, 3)
	std  = np.array([0.229, 0.224, 0.225],
									 dtype=np.float32).reshape(1, 1, 3)



	def __init__(self, opt, split):
		super(hcstvg, self).__init__()
		self.data_dir = os.path.join(opt.data_dir, 'HC-STVG/data')
		self.img_dir = os.path.join(self.data_dir, 'frames')
		# if split == 'test':
		#   self.annot_path = os.path.join(
		#       self.data_dir, 'annotations', 
		#       'image_info_test-dev2017.json').format(split)
		# else:
		#   if opt.task == 'exdet':
		#     self.annot_path = os.path.join(
		#       self.data_dir, 'annotations', 
		#       'instances_extreme_{}2017.json').format(split)
		#   else:
		# split='val'
		self._tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		max_seq_length = 40
		max_region_num = 80
		self.interval = 8
		self._padding_index = 0
		self._max_seq_length = max_seq_length
		self.max_region_num = max_region_num
		self.annot_path = os.path.join(
			self.data_dir, 
			'{}_annotations.json').format(split)
		if split == 'test':
			self.annot_path = os.path.join(
			self.data_dir, 
			'{}_annotations_new.json').format(split)
		self.max_objs = 1
		self.class_name = [
			'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
			'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
			'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
			'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
			'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
			'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
			'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
			'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
			'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
			'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
			'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
			'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
			'scissors', 'teddy bear', 'hair drier', 'toothbrush']
		self._valid_ids = [
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
			14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
			24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
			37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
			48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
			58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
			72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
			82, 84, 85, 86, 87, 88, 89, 90]
		self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
		self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
											for v in range(1, self.num_classes + 1)]
		self._data_rng = np.random.RandomState(123)
		self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
														 dtype=np.float32)
		self._eig_vec = np.array([
				[-0.58752847, -0.69563484, 0.41340352],
				[-0.5832747, 0.00994535, -0.81221408],
				[-0.56089297, 0.71832671, 0.41158938]
		], dtype=np.float32)
		# self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
		# self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

		self.split = split
		self.opt = opt
		# with open(os.path.join(self.data_dir, 'annotations/voc.json'),'r') as f:
		# 	self.voc = json.load(f)

		print('==> initializing coco 2017 {} data.'.format(split))
		# self.coco = coco.COCO(self.annot_path)
		# self.images = self.coco.getImgIds()
		# self.num_samples = len(self.images)
		with open(os.path.join(self.annot_path), 'r') as f:
			self.images = json.load(f)
		with open(os.path.join(self.data_dir, '{}_full_annotations.pkl'.format(split)), 'rb') as f:
			self.annots = pickle.load(f)
		self.num_samples = len(self.images)
		if not os.path.exists(os.path.join(self.data_dir, "cache")):
			os.makedirs(os.path.join(self.data_dir, "cache"))

		cache_path = os.path.join(self.data_dir, "cache", 'tokens_'+split+'_'+str(max_seq_length)+ "_" + str(max_region_num) + '.pkl')
		if not os.path.exists(cache_path):
			self.entries = []
			self.tokenize()
			self.tensorize()
			pickle.dump(self.entries, open(cache_path, 'wb'))
		else:
			print('loading entries from %s' %(cache_path))
			self.entries = pickle.load(open(cache_path, "rb"))

		

		print('Loaded {} {} samples'.format(split, self.num_samples))

	def _to_float(self, x):
		return float("{:.2f}".format(x))

	def tokenize(self):
		"""Tokenizes the captions.

		This will add caption_tokens in each entry of the dataset.
		-1 represents nil, and should be treated as padding_idx in embedding.
		"""
		for im in self.images:
			entry={}
			tokens_list = []
			input_mask_list = []
			segment_ids_list = []
			# print(im['captions'] + im['questions'])
			for n in im['captions'] + im['questions']:
				sentence_tokens = self._tokenizer.tokenize(n["description"][:-1]+' '+n["description"][-1])
				sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

				tokens = [
					self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
					for w in sentence_tokens
				]

				tokens = tokens[:self._max_seq_length]
				segment_ids = [0] * len(tokens)
				input_mask = [1] * len(tokens)

				if len(tokens) < self._max_seq_length:
					# Note here we pad in front of the sentence
					padding = [self._padding_index] * (self._max_seq_length - len(tokens))
					tokens = tokens + padding
					input_mask += padding
					segment_ids += padding

				assert_eq(len(tokens), self._max_seq_length)
				tokens_list.append(tokens)
				input_mask_list.append(input_mask)
				segment_ids_list.append(segment_ids)
			entry["token"] = tokens_list
			entry["input_mask"] = input_mask_list
			entry["segment_ids"] = segment_ids_list
			self.entries.append(entry)

	def tensorize(self):

		for entry in self.entries:
			token = torch.from_numpy(np.array(entry["token"]))
			entry["token"] = token

			input_mask = torch.from_numpy(np.array(entry["input_mask"]))
			entry["input_mask"] = input_mask

			segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
			entry["segment_ids"] = segment_ids

	def convert_eval_format(self, all_bboxes):
		# import pdb; pdb.set_trace()
		detections = []
		for image_id in all_bboxes:
			for cls_ind in all_bboxes[image_id]:
				category_id = self._valid_ids[cls_ind - 1]
				for bbox in all_bboxes[image_id][cls_ind]:
					bbox[2] -= bbox[0]
					bbox[3] -= bbox[1]
					score = bbox[4]
					bbox_out  = list(map(self._to_float, bbox[0:4]))

					detection = {
							"image_id": int(image_id),
							"category_id": int(category_id),
							"bbox": bbox_out,
							"score": float("{:.2f}".format(score))
					}
					if len(bbox) > 5:
							extreme_points = list(map(self._to_float, bbox[5:13]))
							detection["extreme_points"] = extreme_points
					detections.append(detection)
		return detections

	def __len__(self):
		return self.num_samples

	def save_results(self, results, save_dir):
		json.dump(self.convert_eval_format(results), 
								open('{}/results.json'.format(save_dir), 'w'))
	
	def run_eval(self, results, save_dir):
		# result_json = os.path.join(save_dir, "results.json")
		# detections  = self.convert_eval_format(results)
		# json.dump(detections, open(result_json, "w"))
		self.save_results(results, save_dir)
		coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
		coco_eval = COCOeval(self.coco, coco_dets, "bbox")
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
