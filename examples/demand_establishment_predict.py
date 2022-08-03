# %%
'''DATA PREPARE'''
dataset_path = 'openks/data/NSF'
global_kg = 'openks/data/global_kg'

import os
import numpy as np
import xml.etree.ElementTree as ET

def process_dataset(dataset_path):
	# Get file
	files = []
	g = os.walk(dataset_path)
	for path, _, file_list in g:
		for file_name in file_list:
			files.append(os.path.join(path, file_name))

	# Parse xml
	text_titles = []
	text_abs = []

	for file in files:
		tree = ET.parse(file)
		root = tree.getroot()
		text_title = root.find('Award').find('AwardTitle').text
		text_ab = root.find('Award').find('AbstractNarration').text
		text_titles.append(text_title)
		text_abs.append(text_ab)

	return text_titles, text_abs
print('--'*10)
print('PREPARE DATA...')
print('--'*10)
titles, abs = process_dataset(dataset_path)
labels = [1] * len(titles)

fake_size = 100
from faker import Faker
fake = Faker()
neg_titles = []
neg_abs = []

for i in range(fake_size):
	neg_titles.append(fake.sentence())
	neg_abs.append(fake.text())

titles.extend(neg_titles)
abs.extend(neg_abs)
labels.extend([0]*fake_size)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


# # %%
# '''PROCESS DATA'''
save_path = 'openks/data/NSF-parse'
create_mkdir(save_path)

max_len_title = 20
max_len_abs = 200
n_node = 2000

from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_emb = BertModel.from_pretrained('bert-base-uncased')

print('--'*10)
print('PROCESS DATA...')
print('--'*10)

input_title = tokenizer.batch_encode_plus(titles,
									max_length=max_len_title,
									padding=True,
									truncation=True,
									return_tensors='pt'
									)
with torch.no_grad():
	title_emb = bert_emb(**input_title)[-1]

input_ab = tokenizer.batch_encode_plus(titles,
									max_length=max_len_abs,
									padding=True,
									truncation=True,
									return_tensors='pt'
									)
with torch.no_grad():
	ab_emb = bert_emb(**input_title)[-1]

if not os.path.exists(save_path):
	os.mkdir(save_path)
np.save(f'{save_path}/title_emb.npy', np.array(title_emb))
np.save(f'{save_path}/ab_emb.npy', np.array(ab_emb))
np.save(f'{save_path}/label.npy', np.array(labels))

import pickle as pkl
with open(f'{global_kg}/kg_paper_paper', 'rb') as fl:
    paper_paper = pkl.load(fl)
with open(f'{global_kg}/kg_paper_title', 'rb') as fl:
    paper_title = pkl.load(fl)

global_kg_node = list(paper_title.values())[:n_node]
global_kg_edge = [[], []]
for key in paper_paper:
	if key >= n_node:
		break
	for node in paper_paper[key]:
		if node < n_node:
			global_kg_edge[0].append(key)
			global_kg_edge[1].append(node)

global_node_input = tokenizer.batch_encode_plus(global_kg_node,
												max_length=max_len_abs,
												padding=True,
												truncation=True,
												return_tensors='pt'
												)
with torch.no_grad():
	global_node_emb = bert_emb(**global_node_input)[-1]

np.save(f'{save_path}/global_node_emb.npy', np.array(global_node_emb))
np.save(f'{save_path}/global_edge_index.npy', np.array(global_kg_edge))

# %%
'''LOAD DATA AND MODEL'''

print('--'*10)
print('LOAD DATA AND MODEL...')
print('--'*10)

import torch
import numpy as np

input_texts = np.load(f'{save_path}/title_emb.npy', allow_pickle=True)
input_abs = np.load(f'{save_path}/ab_emb.npy', allow_pickle=True)
input_labels = np.load(f'{save_path}/label.npy', allow_pickle=True)
global_emb = np.load(f'{save_path}/global_node_emb.npy', allow_pickle=True)
global_edge_index = np.load(f'{save_path}/global_edge_index.npy', allow_pickle=True)

# print(input_labels)
text_num = input_texts.shape[0]
train_index = np.random.randint(0, text_num, size=int(0.7*text_num))
val_index = np.array(list(set(range(text_num)) - set(train_index)))

# %%
'''SPILT DATASET'''
train_texts = input_texts[train_index]
train_abs = input_abs[train_index]
train_label = input_labels[train_index]

val_texts = input_texts[val_index]
val_abs = input_abs[val_index]
val_label = input_labels[val_index]

print(train_label)
print(val_label)

from torch.utils.data import Dataset, DataLoader

class demand_data(Dataset):
	def __init__(self, text, kg, label):
		super().__init__()
		self.text = text
		self.kg = kg
		self.label = label
	
	def __len__(self):
		return self.text.shape[0]
	
	def __getitem__(self, index):
		return [self.text[index], self.kg[index], self.label[index]]

train_data = demand_data(train_texts, train_abs, train_label)
val_data = demand_data(val_texts, val_abs, val_label)

#%%
'''TRAINING MODEL'''

print('--'*10)
print('TRAINING MODEL...')
print('--'*10)

# from openks.models.pytorch.attn_inter import AttInter
from torch_geometric.data import Data
from openks.models import OpenKSModel

platform = 'PyTorch'
model_name = 'AttInter'
AttInter = OpenKSModel.get_module(platform, model_name)


import argparse
def parse_args(args=None):
	parser = argparse.ArgumentParser(
		description='Training and Testing Command Predictions Models',
	)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--feat_dim', default=768, type=int)
	parser.add_argument('--conv_emb_dim', default=300, type=int)
	parser.add_argument('--pred_hid_dim', default=84, type=int)
	parser.add_argument('--graph_pooling', default="mean", type=str)
	parser.add_argument('--gnn_type', default="gin", type=str)
	parser.add_argument('--conv_drop_ratio', default=0.0, type=float)
	parser.add_argument('--JK', type=str, default="last",
						help='how the node features across layers are combined. last, sum, max or concat')
	parser.add_argument('--num_conv_layer', type=int, default=3,
						help='number of GNN message passing layers (default: 3).')
	parser.add_argument('--in_channels', default=768, type=int)
	parser.add_argument('--ratio', default=0.1, type=float)
	parser.add_argument('--epoch', default=10, type=int)
	parser.add_argument('--device', default=0, type=int)
	return parser.parse_args(args)

att_config = parse_args()

train_loader = DataLoader(train_data, batch_size=att_config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=att_config.batch_size, shuffle=False)

model = AttInter(att_config=att_config)
device = att_config.device
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


def train():
	for i in range(att_config.epoch):
		model.train()
		acc = 0.0
		sum = 0.0
		loss_sum = 0
		for idx, input_data in enumerate(train_loader):
			# print(global_edge_index, global_edge_index.dtype)
			emb_list = [input_data[0].to(device),
						input_data[1].to(device).flatten(),
						Data(torch.tensor(global_emb), torch.tensor(global_edge_index, dtype=torch.long)).to(device)]
			out = model(emb_list)
			loss = criterion(out, input_data[2].to(device))

			loss.backward()
			optimizer.step()
			acc += torch.sum(torch.argmax(out, dim=1)==input_data[2].to(device)).item()
			sum+=len(input_data[2])
			loss_sum +=loss.item()

		print('train acc: %.2f%%, loss: %.4f'%(100*acc/sum, loss_sum/sum))
		
		model.eval()
		acc = 0.0
		sum = 0.0
		for idx, input_data in enumerate(val_loader):
			emb_list = [input_data[0].to(device), 
						input_data[1].to(device), 
						Data(torch.tensor(global_emb), torch.tensor(global_edge_index, dtype=torch.long)).to(device)]
			out = model(emb_list)
			loss = criterion(out, input_data[2].to(device))
			acc += torch.sum(torch.argmax(out, dim=1)==input_data[2].to(device)).item()
			sum+=len(input_data[2])
		print('valid acc: %.2f%%'%(100*acc/sum))

train()