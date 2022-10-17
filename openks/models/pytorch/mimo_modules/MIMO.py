# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert.modeling import *

class LSTM_encoder(nn.Module):
	"""docstring for LSTM_encoder"""
	def __init__(self, wordEmbedding, word2ID, pos2ID, cap2ID, word_dim, input_size, hidden_size, num_layers, bidirectional, lm_config, postag_config, cap_config, device):
		super(LSTM_encoder, self).__init__()

		self.device = device
		self.bidirectional = bidirectional

		self.lm_config = lm_config
		self.postag_config = postag_config
		self.cap_config = cap_config

		self.WordEmbedding = nn.Embedding(len(word2ID), word_dim)
		self.POSEmbedding = nn.Embedding(len(pos2ID), int(math.ceil(math.log(len(pos2ID),2))))
		self.CAPEmbedding = nn.Embedding(len(cap2ID), int(math.ceil(math.log(len(cap2ID),2))))

		self.w_lm = nn.Parameter(torch.randn(200, word_dim))
		self.w_pos = nn.Parameter(torch.randn(int(math.ceil(math.log(len(pos2ID),2))), word_dim))
		self.w_cap = nn.Parameter(torch.randn(int(math.ceil(math.log(len(cap2ID),2))), word_dim))

		self.WordEmbedding.weight = nn.Parameter(wordEmbedding, requires_grad=False)

		self.Word2ID = word2ID
		self.POS2ID = pos2ID
		self.CAP2ID = cap2ID

		self.input_size = input_size
		# self.batch_size = batch_size
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
		self.hidden = None

		#for weight in self.parameters():
			#print type(weight), weight.size(), weight.requires_grad

	def forward(self, tuple_batch):
		sentences_batch, pos_batch, cap_batch, lm_batch = tuple_batch

		length = []
		wordsIndex_batch = []
		posesIndex_batch = []
		capsIndex_batch = []
		lm_batch_new = []

		length_max = len(sentences_batch[0])

		for index in range(len(sentences_batch)):

			sentence = sentences_batch[index]
			poses = pos_batch[index]
			caps = cap_batch[index]

			assert len(sentence) == len(poses) == len(caps)

			length.append(len(sentence))

			wordsIndex = []
			posesIndex = []
			capsIndex = []

			for word in sentence:
				if word in self.Word2ID:
					wordsIndex.append(self.Word2ID[word])
				else:
					wordsIndex.append(self.Word2ID['<unk>'])
			wordsIndex += [0]*(length_max-len(wordsIndex))
			wordsIndex_batch.append(wordsIndex)

			for pos in poses:
				if pos not in self.POS2ID:
					posesIndex.append(self.POS2ID['SYM'])
				else:
					posesIndex.append(self.POS2ID[pos])
			posesIndex += [0]*(length_max-len(posesIndex))
			posesIndex_batch.append(posesIndex)

			for cap in caps:
				capsIndex.append(self.CAP2ID[cap])
			capsIndex += [0]*(length_max-len(capsIndex))
			capsIndex_batch.append(capsIndex)

		wordsIndex_batch = autograd.Variable(torch.LongTensor(wordsIndex_batch)).to(self.device)
		posesIndex_batch = autograd.Variable(torch.LongTensor(posesIndex_batch)).to(self.device)
		capsIndex_batch = autograd.Variable(torch.LongTensor(capsIndex_batch)).to(self.device)

		lmsEmb = autograd.Variable(lm_batch)

		sentencesEmb = self.WordEmbedding(wordsIndex_batch)
		posesEmb = self.POSEmbedding(posesIndex_batch)
		capsEmb = self.CAPEmbedding(capsIndex_batch)

		emb = sentencesEmb + 0

		if self.lm_config[0]:
			emb += torch.matmul(lmsEmb, self.w_lm)
		if self.postag_config[0]:
			emb += torch.matmul(posesEmb, self.w_pos)
		if self.cap_config[0]:
			emb += torch.matmul(capsEmb, self.w_cap)
		
		emb = emb.to(self.device)

		packed_sentencesEmb = pack_padded_sequence(emb, length, batch_first=True)
		
		packed_output, (ht, ct)  = self.lstm(packed_sentencesEmb, self.hidden)
		output, _ = pad_packed_sequence(packed_output, batch_first=True)

		return output.transpose(0,1).transpose(1,2), posesEmb.transpose(0,1).transpose(1,2), capsEmb.transpose(0,1).transpose(1,2), lmsEmb.transpose(0,1).transpose(1,2)
			
	def init_hidden(self, batch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_size)
		return (autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device))

class LSTM_decoder(nn.Module):
	"""docstring for LSTM_decoder"""
	def __init__(self, input_size, hidden_size, tagset_size, pos2ID, cap2ID, lm_config, postag_config, cap_config, device):
		super(LSTM_decoder, self).__init__()

		self.device = device
		self.tagset_size = tagset_size
		self.hidden_size = hidden_size
		#self.hidden = self.init_hidden()

		self.lm_config = lm_config
		self.postag_config = postag_config
		self.cap_config = cap_config

		self.w_ii = nn.Parameter(torch.randn(4 * hidden_size, input_size))
		self.w_hi = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
		self.w_ti = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
		self.w_co = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.w_ht = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.b_i = nn.Parameter(torch.randn(5 * hidden_size))

		self.w_y_fact = nn.Parameter(torch.randn(tagset_size, hidden_size))
		self.b_y_fact = nn.Parameter(torch.randn(tagset_size))
		
		self.w_fact = nn.Parameter(torch.randn(tagset_size, tagset_size))
		self.w_y_cond = nn.Parameter(torch.randn(tagset_size, hidden_size))
		self.b_y_cond = nn.Parameter(torch.randn(tagset_size))

		self.w_lmw = nn.Parameter(torch.randn(200, input_size))
		self.w_posw = nn.Parameter(torch.randn(int(math.ceil(math.log(len(pos2ID),2))), input_size))
		self.w_capw = nn.Parameter(torch.randn(int(math.ceil(math.log(len(cap2ID),2))), input_size))

		self.w_lmt = nn.Parameter(torch.randn(hidden_size, 200))
		self.w_post = nn.Parameter(torch.randn(hidden_size, int(math.ceil(math.log(len(pos2ID),2)))))
		self.w_capt = nn.Parameter(torch.randn(hidden_size, int(math.ceil(math.log(len(cap2ID),2)))))

		self.reset_parameters()

		self.hidden = None

		#for weight in self.parameters():
			#print type(weight), weight.size(), weight.requires_grad

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			# print type(weight), weight.size()
			weight.data.uniform_(-stdv, stdv)

	def forward(self, inputs, lmsEmb, posesEmb, capsEmb):
		border = self.hidden_size
		hs = [self.hidden[0][0].transpose(0, 1)]
		cs = [self.hidden[1][0].transpose(0, 1)]
		ts = [self.hidden[2][0].transpose(0, 1)]

		new_inputs = inputs + 0

		if self.lm_config[1]:
			new_inputs += torch.matmul(lmsEmb.transpose(1, 2), self.w_lmw).transpose(1, 2)
		if self.postag_config[1]:
			new_inputs += torch.matmul(posesEmb.transpose(1, 2), self.w_posw).transpose(1, 2)
		if self.cap_config[1]:
			new_inputs += torch.matmul(capsEmb.transpose(1, 2), self.w_capw).transpose(1, 2)

		hidden_out = []
		outputs_fact = []
		outputs_distrib_fact = []
		outputs_condition = []
		outputs_distrib_condition = []

		for index in range(len(new_inputs)):
			_input = new_inputs[index]
			posEmb = posesEmb[index]
			capEmb = capsEmb[index]
			lmEmb = lmsEmb[index]

			ii = torch.mm(self.w_ii, _input)
			hi = torch.mm(self.w_hi, hs[-1])
			ti = torch.mm(self.w_ti, ts[-1])
			i = torch.sigmoid(ii[:border] + hi[:border] + ti[:border] + self.b_i[:border].view(-1,1))
			f = torch.sigmoid(ii[border:2*border] + hi[border:2*border] + ti[border:2*border] + self.b_i[border:2*border].view(-1,1))
			z = torch.tanh(ii[2*border:3*border] + hi[2*border:3*border] + ti[2*border:3*border] + self.b_i[2*border:3*border].view(-1,1))
			c = f * cs[-1] + i * z
			o = torch.sigmoid(ii[3*border:4*border] + hi[3*border:4*border] + torch.mm(self.w_co, c) + self.b_i[3*border:4*border].view(-1,1))
			h = o * torch.tanh(c)
			_T = torch.mm(self.w_ht, h) + self.b_i[4*border:].view(-1,1)
			T = _T + 0

			if self.lm_config[-1]:
				T += torch.mm(self.w_lmt, lmEmb)
			if self.postag_config[-1]:
				T += torch.mm(self.w_post, posEmb)
			if self.cap_config[-1]:
				T += torch.mm(self.w_capt, capEmb)

			hs.append(h)
			cs.append(c)
			ts.append(T)

			y_fact = torch.mm(self.w_y_fact, T) + self.b_y_fact.view(-1,1)
			outputs_fact.append(F.log_softmax(y_fact, 0).view(1, self.tagset_size, -1))
			outputs_distrib_fact.append(y_fact.view(1, self.tagset_size, -1))

			y_condition = torch.mm(self.w_y_cond, T) + self.b_y_cond.view(-1,1)
			outputs_condition.append(F.log_softmax(y_condition, 0).view(1, self.tagset_size, -1))
			outputs_distrib_condition.append(y_condition.view(1, self.tagset_size, -1))

			hidden_out.append(T.view(1, self.hidden_size, -1))

		outputs_fact = torch.cat(outputs_fact).transpose(0,2).transpose(1,2)
		outputs_distrib_fact = torch.cat(outputs_distrib_fact).transpose(0,2).transpose(1,2)

		outputs_condition = torch.cat(outputs_condition).transpose(0,2).transpose(1,2)
		outputs_distrib_condition = torch.cat(outputs_distrib_condition).transpose(0,2).transpose(1,2)

		hidden_out = torch.cat(hidden_out).transpose(0,2).transpose(1,2)

		# print outputs.size(), type(outputs)
		return outputs_fact, outputs_condition, outputs_distrib_fact, outputs_distrib_condition, hidden_out

	def init_hidden(self, batch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_size)
		return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device))

class TAG_TF(nn.Module):
	"""docstring for TAG_TF"""
	def __init__(self, dim, num_attention_heads=3):
		super(TAG_TF, self).__init__()
		self.position_embeddings = nn.Embedding(512, dim)
		self.LayerNorm = BertLayerNorm(dim, eps=1e-12)

		config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=dim, num_attention_heads=num_attention_heads)
		self.fact_attention = BertAttention(config)
		self.cond_attention = BertAttention(config)

		self.fact_inter_attention = BertInterAttention(config)
		self.cond_inter_attention = BertInterAttention(config)
		# print(config.num_attention_heads)

	def forward(self, inputs, attention_mask):
		"""
		inputs: [size_b, seq_len, dim]
		attention_mask: [size_b, seq_len]
		"""
		seq_length = inputs.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
		# [size_b, seq_len]
		position_ids = position_ids.unsqueeze(0).expand(inputs.size()[:2])
		position_embeddings = self.position_embeddings(position_ids)

		embeddings = inputs + position_embeddings
		embeddings = self.LayerNorm(embeddings)

		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		
		fact_self_attention_output = self.fact_attention(embeddings, extended_attention_mask)
		cond_self_attention_output = self.cond_attention(embeddings, extended_attention_mask)

		fact_attention_output = self.fact_inter_attention(fact_self_attention_output, cond_self_attention_output, cond_self_attention_output, extended_attention_mask)
		cond_attention_output = self.cond_inter_attention(cond_self_attention_output, fact_self_attention_output, fact_self_attention_output, extended_attention_mask)

		return fact_attention_output, cond_attention_output

class BertInterAttention(nn.Module):
	def __init__(self, config):
		super(BertInterAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

		self.output = BertSelfOutput(config)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, q, k, v, attention_mask):
		mixed_query_layer = self.query(q)
		mixed_key_layer = self.key(k)
		mixed_value_layer = self.value(v)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		attention_output = self.output(context_layer, q)
		return attention_output

class BERT_Encoder(nn.Module):
	"""docstring for BERT_Encoder"""
	def __init__(self, num_labels=11, hidden_dropout_prob=0.1):
		super(BERT_Encoder, self).__init__()
		self.num_labels = num_labels
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(hidden_dropout_prob)
		
	def init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		sequence_output = self.dropout(sequence_output)
		return sequence_output

class Extractor(nn.Module):
	"""docstring for Extractor"""
	def __init__(self, hidden_dim, tagset_size, name=''):
		super(Extractor, self).__init__()
		self.tuple_layer = nn.Linear(hidden_dim, tagset_size)
		self.position_embeddings = nn.Embedding(300, hidden_dim)
		self.name = '_extractor_'+name

	def forward(self, inputs, position_ids):
		"""
		inputs: [size_b, seq_len, dim]
		position_ids: [size_b, seq_len]
		"""
		# print(inputs.size(), position_ids.size())
		position_embeddings = self.position_embeddings(position_ids)
		embeddings = inputs + position_embeddings

		logits = F.log_softmax(self.tuple_layer(embeddings), 2)

		return logits

class Multi_head_Net(nn.Module):
	"""docstring for Multi_head_Net"""
	def __init__(self, hidden_dim, tagset_size):
		super(Multi_head_Net, self).__init__()
		self.name = '_multi_head'

		self.w_lm = nn.Parameter(torch.randn(hidden_dim))
		self.w_pos = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)
		self.w_cap = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)
		# self.w = nn.Parameter(torch.randn(hidden_dim))
		self.b = nn.Parameter(torch.randn(hidden_dim))

		self.tagset_size = tagset_size

		self.fact_layer = nn.Linear(hidden_dim, tagset_size)
		self.cond_layer = nn.Linear(hidden_dim, tagset_size)

	def forward(self, tuple_batch):
		lm_input_batch, pos_input_batch, cap_input_batch = tuple_batch
		hidden_out_batch = 0

		if isinstance(lm_input_batch, torch.Tensor):
			hidden_out_batch += lm_input_batch*self.w_lm
			# hidden_out_batch.append(lm_input_batch)
		if isinstance(pos_input_batch, torch.Tensor):
			hidden_out_batch += pos_input_batch*self.w_pos
			# hidden_out_batch.append(pos_input_batch)
		if isinstance(cap_input_batch, torch.Tensor):
			hidden_out_batch += cap_input_batch*self.w_cap
			# hidden_out_batch.append(cap_input_batch)
		hidden_out_batch += self.b
		# hidden_out_batch = torch.cat(hidden_out_batch, 2)

		predict_fact_batch = F.log_softmax(self.fact_layer(hidden_out_batch), 2)
		predict_condition_batch = F.log_softmax(self.cond_layer(hidden_out_batch), 2)

		return predict_fact_batch, predict_condition_batch, hidden_out_batch

class Multi_head_Two_Net(nn.Module):
	"""docstring for Multi_head_Two_Net"""
	def __init__(self, hidden_dim, tagset_size):
		super(Multi_head_Two_Net, self).__init__()
		self.name = '_multi_head_all'

		self.w1 = nn.Parameter(torch.randn(hidden_dim))
		self.w2 = nn.Parameter(torch.randn(hidden_dim))

		self.b = nn.Parameter(torch.randn(hidden_dim))

		self.tagset_size = tagset_size

		self.fact_layer = nn.Linear(hidden_dim, tagset_size)
		self.cond_layer = nn.Linear(hidden_dim, tagset_size)

	def forward(self, tuple_batch):
		first_input_batch, second_input_batch = tuple_batch
		hidden_out_batch = first_input_batch*self.w1 + second_input_batch*self.w2 + self.b
		# hidden_out_batch = torch.cat(hidden_out_batch, 2)

		predict_fact_batch = F.log_softmax(self.fact_layer(hidden_out_batch), 2)
		predict_condition_batch = F.log_softmax(self.cond_layer(hidden_out_batch), 2)

		return predict_fact_batch, predict_condition_batch, hidden_out_batch

class MIMO_LSTM(nn.Module):
	"""docstring for Tagger"""
	def __init__(self, wordEmbedding, word2ID, pos2ID, cap2ID, embedding_dim, input_dim, hidden_dim, tagset_size_fact, tagset_size_condition, lm_config, postag_config, cap_config, device):
		super(MIMO_LSTM, self).__init__()
		self.name = ''

		self.hidden_dim = hidden_dim

		self.model_LSTM_encoder = LSTM_encoder(wordEmbedding, word2ID, pos2ID, cap2ID, embedding_dim, input_dim, hidden_dim, num_layers = 1, bidirectional=True, lm_config=lm_config, postag_config=postag_config, cap_config=cap_config, device=device)

		self.model_LSTM_decoder = LSTM_decoder(hidden_dim * 2, hidden_dim * 2, tagset_size_fact, pos2ID, cap2ID, lm_config, postag_config, cap_config, device)


	def forward(self, tuple_batch, batch_size, attention_mask=None):
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		fact_batch, condition_batch, outputs_distrib_fact, outputs_distrib_condition, hidden_out = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb)

		return fact_batch, condition_batch, outputs_distrib_fact, outputs_distrib_condition, hidden_out

class MIMO_BERT(nn.Module):
	"""docstring for MIMO"""
	def __init__(self, pretrained_model_name, num_labels=11, hidden_dropout_prob=0.1):
		super(MIMO_BERT, self).__init__()
		self.bert = BERT_Encoder(num_labels, hidden_dropout_prob)
		self.hidden_size = 768 if pretrained_model_name.startswith('bert-base') else 1024
		self.classifier_fact = nn.Linear(self.hidden_size, num_labels)
		self.classifier_cond = nn.Linear(self.hidden_size, num_labels)
		self.apply(self.init_weights)

	def init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		sequence_output = bert(input_ids, token_type_ids, attention_mask, labels)
		# [batch_size, sequence_length, num_labels]
		logits_fact = F.log_softmax(self.classifier_fact(sequence_output), 2)
		logits_cond = F.log_softmax(self.classifier_cond(sequence_output), 2)

		return logits_fact, logits_cond, None, None, sequence_output

class MIMO_LSTM_TF(nn.Module):
	"""docstring for MIMO_LSTM_TF"""
	def __init__(self, wordEmbedding, word2ID, pos2ID, cap2ID, embedding_dim, input_dim, hidden_dim, tagset_size_fact, tagset_size_condition, lm_config, postag_config, cap_config, device):
		super(MIMO_LSTM_TF, self).__init__()

		self.hidden_dim = hidden_dim

		self.model_LSTM_encoder = LSTM_encoder(wordEmbedding, word2ID, pos2ID, cap2ID, embedding_dim, input_dim, hidden_dim, num_layers = 1, bidirectional=True, lm_config=lm_config, postag_config=postag_config, cap_config=cap_config, device=device)

		self.model_LSTM_decoder = LSTM_decoder(hidden_dim * 2, hidden_dim * 2, tagset_size_fact, pos2ID, cap2ID, lm_config, postag_config, cap_config, device)

		self.tag_tf = TAG_TF(hidden_dim * 2)

		self.fact_layer = nn.Linear(hidden_dim * 2, tagset_size_fact)
		self.cond_layer = nn.Linear(hidden_dim * 2, tagset_size_condition)

	def forward(self, tuple_batch, batch_size, attention_mask):
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		_, _, _, _, hidden_out = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb)

		fact_att_out, cond_att_out = self.tag_tf(hidden_out, attention_mask)

		y_fact = self.fact_layer(fact_att_out)
		y_cond = self.cond_layer(cond_att_out)

		outputs_fact = F.log_softmax(y_fact, 2)
		outputs_condition = F.log_softmax(y_cond, 2)

		return outputs_fact, outputs_condition






class MIMO_BERT_LSTM(nn.Module):
	"""docstring for MIMO_BERT_LSTM"""
	def __init__(self, pos2ID, cap2ID, hidden_dim, tagset_size_fact, tagset_size_condition, lm_config, postag_config, cap_config, device):
		super(MIMO_BERT_LSTM, self).__init__()
		self.name = ''
		self.POS2ID = pos2ID
		self.CAP2ID = cap2ID
		self.hidden_dim = hidden_dim
		self.POSEmbedding = nn.Embedding(len(pos2ID), int(math.ceil(math.log(len(pos2ID),2))))
		self.CAPEmbedding = nn.Embedding(len(cap2ID), int(math.ceil(math.log(len(cap2ID),2))))
		self.device = device

		self.model_BERT_encoder = BERT_Encoder(tagset_size_fact)

		self.model_LSTM_decoder = LSTM_decoder(hidden_dim, hidden_dim, tagset_size_fact, pos2ID, cap2ID, lm_config, postag_config, cap_config, device)

	def get_embs(self, tuple_batch):
		pos_batch, cap_batch, lm_batch = tuple_batch

		posesIndex_batch = []
		capsIndex_batch = []

		length_max = len(pos_batch[0])

		for index in range(len(pos_batch)):
			poses = pos_batch[index]
			caps = cap_batch[index]

			assert len(poses) == len(caps)
			posesIndex = []
			capsIndex = []
			
			for pos in poses:
				if pos not in self.POS2ID:
					posesIndex.append(self.POS2ID['SYM'])
				else:
					posesIndex.append(self.POS2ID[pos])
			posesIndex += [0]*(length_max-len(posesIndex))
			posesIndex_batch.append(posesIndex)

			for cap in caps:
				capsIndex.append(self.CAP2ID[cap])
			capsIndex += [0]*(length_max-len(capsIndex))
			capsIndex_batch.append(capsIndex)

		posesIndex_batch = autograd.Variable(torch.LongTensor(posesIndex_batch)).to(self.device)
		capsIndex_batch = autograd.Variable(torch.LongTensor(capsIndex_batch)).to(self.device)

		lmsEmb = autograd.Variable(lm_batch)

		posesEmb = self.POSEmbedding(posesIndex_batch)
		capsEmb = self.CAPEmbedding(capsIndex_batch)

		return lmsEmb.transpose(0,1).transpose(1,2), posesEmb.transpose(0,1).transpose(1,2), capsEmb.transpose(0,1).transpose(1,2), 

	def forward(self, tuple_batch, batch_size, input_ids, token_type_ids=None, attention_mask=None):
		encoder = self.model_BERT_encoder(input_ids, token_type_ids, attention_mask)
		encoder = encoder.transpose(0,1).transpose(1,2)
		lmsEmb, posesEmb, capsEmb = self.get_embs(tuple_batch)
		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		fact_batch, condition_batch, outputs_distrib_fact, outputs_distrib_condition, hidden_out = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb)

		return fact_batch, condition_batch, outputs_distrib_fact, outputs_distrib_condition, hidden_out

class MIMO_BERT_LSTM_TF(nn.Module):
	"""docstring for MIMO_BERT_LSTM"""
	def __init__(self, pos2ID, cap2ID, hidden_dim, tagset_size_fact, tagset_size_condition, lm_config, postag_config, cap_config, device):
		super(MIMO_BERT_LSTM_TF, self).__init__()

		self.POS2ID = pos2ID
		self.CAP2ID = cap2ID
		self.hidden_dim = hidden_dim
		self.POSEmbedding = nn.Embedding(len(pos2ID), int(math.ceil(math.log(len(pos2ID),2))))
		self.CAPEmbedding = nn.Embedding(len(cap2ID), int(math.ceil(math.log(len(cap2ID),2))))
		self.device = device

		self.model_BERT_encoder = BERT_Encoder(tagset_size_fact)

		self.model_LSTM_decoder = LSTM_decoder(hidden_dim, hidden_dim, tagset_size_fact, pos2ID, cap2ID, lm_config, postag_config, cap_config, device)

		self.tag_tf = TAG_TF(hidden_dim)

		self.fact_layer = nn.Linear(hidden_dim, tagset_size_fact)
		self.cond_layer = nn.Linear(hidden_dim, tagset_size_condition)

	def get_embs(self, tuple_batch):
		pos_batch, cap_batch, lm_batch = tuple_batch

		posesIndex_batch = []
		capsIndex_batch = []
		lm_batch_new = []

		length_max = len(pos_batch[0])

		for index in range(len(pos_batch)):
			poses = pos_batch[index]
			caps = cap_batch[index]
			lms = lm_batch[index]

			assert len(poses) == len(caps) == len(lms)

			posesIndex = []
			capsIndex = []
			
			for pos in poses:
				if pos not in self.POS2ID:
					posesIndex.append(self.POS2ID['SYM'])
				else:
					posesIndex.append(self.POS2ID[pos])
			posesIndex += [0]*(length_max-len(posesIndex))
			posesIndex_batch.append(posesIndex)

			for cap in caps:
				capsIndex.append(self.CAP2ID[cap])
			capsIndex += [0]*(length_max-len(capsIndex))
			capsIndex_batch.append(capsIndex)

			# print lms.size(), length_max
			lms_pad = torch.randn((length_max-len(lms), len(lms[0])), dtype=torch.float32)
			lms = torch.cat([lms.to(self.device), lms_pad.to(self.device)])
			lm_batch_new.append(lms.view(1, lms.size(0), lms.size(1)))

		posesIndex_batch = autograd.Variable(torch.LongTensor(posesIndex_batch)).to(self.device)
		capsIndex_batch = autograd.Variable(torch.LongTensor(capsIndex_batch)).to(self.device)

		lm_batch_new = torch.cat(lm_batch_new, 0)
		lmsEmb = autograd.Variable(lm_batch_new).to(self.device)

		posesEmb = self.POSEmbedding(posesIndex_batch)
		capsEmb = self.CAPEmbedding(capsIndex_batch)

		return lmsEmb.transpose(0,1).transpose(1,2), posesEmb.transpose(0,1).transpose(1,2), capsEmb.transpose(0,1).transpose(1,2), 

	def forward(self, tuple_batch, batch_size, input_ids, token_type_ids=None, attention_mask=None):
		encoder = self.model_BERT_encoder(input_ids, token_type_ids, attention_mask)
		encoder = encoder.transpose(0,1).transpose(1,2)

		lmsEmb, posesEmb, capsEmb = self.get_embs(tuple_batch)
		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		fact_batch, condition_batch, outputs_distrib_fact, outputs_distrib_condition, hidden_out = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb)

		fact_att_out, cond_att_out = self.tag_tf(hidden_out, attention_mask)

		y_fact = self.fact_layer(fact_att_out)
		y_cond = self.cond_layer(cond_att_out)

		outputs_fact = F.log_softmax(y_fact, 2)
		outputs_condition = F.log_softmax(y_cond, 2)

		return fact_batch, condition_batch

# 		