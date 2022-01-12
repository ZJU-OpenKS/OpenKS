from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torch import nn
import torch
import json

class LangFeats(nn.Module):
	def __init__(self):
		super(LangFeats, self).__init__()
		with open('/home/rusu5516/vidstg/annotations/voc.json', 'r')as f:
			self.voc = json.load(f)
		self.we = nn.Embedding(len(self.voc)+1, 1024, padding_idx=0)
		self.mlp1 = nn.Sequential(
			nn.Linear(in_features=1024, out_features=512),
			nn.ReLU())
		self.bilstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
		self.mlp2 = nn.Sequential(
			nn.Linear(in_features=512, out_features=64))

	def forward(self, sents):
		max_num = 0
		image_num = len(sents[0])
		word_idx_list=[]
		sents_per_image = []
		sents_start_idx = []
		s_idx = 0
		max_sents_num = 0
		# print(sents)
		for i in range(image_num):
			sents_start_idx.append(s_idx)
			sn_pi=0
			for j in range(3):
				if sents[j][i] != 'n/a':
					sn_pi += 1
					s_idx +=1
					words = sents[j][i][:-1].split(' ')
					word_idx = []
					for word in words:
						word_idx.append(self.voc[word])
					word_idx.append(self.voc[sents[j][i][-1]])
					word_idx_list.append(word_idx)
					if len(word_idx) > max_num:
						max_num=len(word_idx)
			sents_per_image.append(sn_pi)
		# for k in sents:
		# 	sents_start_idx.append(s_idx)
		# 	s_idx+=len(k)
		# 	sents_per_image.append(len(k))
		# 	if len(k)>max_sents_num:
		# 		max_sents_num=len(k)
		# 	for sent in k:
		# 		sent_list.append(sent)
		# 		if len(sent)>max_num:
		# 			max_num=len(sent)

		# batch_idx=[[] for i in range(max_sents_num)]

		# for k in range(max_sents_num): 
		# 	for i in range(image_num):
		# 		if k < sents_per_image[i]:
		# 			batch_idx[k].append(sents_start_idx[i]+k)

		# print( sents_per_image)
		for i in range(len(word_idx_list)):
			if len(word_idx_list[i]) < max_num:
				# print(word_idx_list[i])
				word_idx_list[i] = word_idx_list[i] + [0 for n in range(max_num - len(word_idx_list[i]))]
				# print(word_idx_list[i])
			word_idx_list[i] = torch.LongTensor(word_idx_list[i])

		# print(word_idx_list)

		sent_tensor = torch.stack(word_idx_list,0).cuda()

		# print(sent_tensor)

		embedding = self.we(sent_tensor)

		x = self.mlp1(embedding.view(-1,1024))

		outputs, _ = self.bilstm(x.view(-1, max_num, 512))

		langfeats = torch.cat([outputs[:,-1,:256], outputs[:,0,256:]], dim=1)

		langfeats = self.mlp2(langfeats)

		# print('lf:', langfeats)

		return langfeats, sents_per_image, sents_start_idx