import torch
import numpy as np
from transformers import GPT2Tokenizer
from torch import nn
from transformers import GPT2Model
import json
import os

class GPTClassifier(nn.Module):

    def __init__(self, n_token=50259, n_class=281, dropout=0.5):

        super(GPTClassifier, self).__init__()

        self.bert = GPT2Model.from_pretrained("gpt2")
        self.bert.resize_token_embeddings(n_token)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768 * 2, n_class)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(768 * 2, n_class)

    def forward(self, input_id_h, input_id_t, mask_h, mask_t, context, mask_context):
        pooled_output_t = self.bert(input_ids= input_id_t, attention_mask=mask_t,return_dict=False)[0][:,-1,:]
        pooled_output_h = self.bert(input_ids= input_id_h, attention_mask=mask_h,return_dict=False)[0][:,-1,:]
        pooled_output_c = self.bert(input_ids= context, attention_mask=mask_context,return_dict=False)[0][:,-1,:]
        pooled_output_h = torch.cat((pooled_output_h, pooled_output_c), 1)
        pooled_output_t = torch.cat((pooled_output_t, pooled_output_c), 1)
        dropout_output_h = self.dropout(pooled_output_h)
        dropout_output_t = self.dropout(pooled_output_t)
        linear_output_h = self.linear(dropout_output_h)
        linear_output_t = self.linear2(dropout_output_t)
        final_layer_h = self.relu(linear_output_h)
        final_layer_t = self.relu(linear_output_t)

        return final_layer_h, final_layer_t
