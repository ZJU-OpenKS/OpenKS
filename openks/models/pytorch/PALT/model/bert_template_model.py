from model.AdapterBert import AdapterBertModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import re
from transformers import BertPreTrainedModel, BertModel
from model.BertWithLinearModel import MyBertLinearModel

class PromptMLPEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, index_size):
        super().__init__()
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.index_size = index_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0]))))
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.index_size)
        self.mlp_head = nn.Sequential(nn.Linear(self.index_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        seq_indices = self.seq_indices.to(self.embedding.weight.device)
        input_embeds = self.embedding(seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(input_embeds).squeeze()
        return output_embeds


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size):
        super().__init__()
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0]))))
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        seq_indices = self.seq_indices.to(self.embedding.weight.device)
        input_embeds = self.embedding(seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class PTuneNSP(BertPreTrainedModel):

    def __init__(self, config, template, pseudo_token_id,
                 pad_token_id, unk_token_id,
                 use_mlm_finetune=False,
                 use_head_finetune=True,
                 index_size=512,
                 use_mlpencoder=False,
                 word_embedding_type=False,
                 word_embedding_hidden_size=None,
                 word_embedding_dropout=False,
                 word_embedding_layernorm=False,
                 top_additional_layer_type=None,
                 top_additional_layer_hidden_size=None,
                 top_use_dropout=False,
                 dropout_ratio=None,
                 top_use_layernorm=False,
                 top_layer_nums=None,
                 adapter_type=None,
                 adapter_size=None,):
        super().__init__(config)
        if top_additional_layer_type:
            self.bert = MyBertLinearModel(
                config,
                use_dropout=top_use_dropout,
                dropout_ratio=dropout_ratio,
                use_layernorm=top_use_layernorm,
                top_additional_layer_type=top_additional_layer_type,
                top_additional_layer_hidden_size=top_additional_layer_hidden_size,
                top_layer_nums=top_layer_nums,
            )
        elif adapter_type:
            assert adapter_size is not None
            print("Using adapter model:\n\tadapter_type: %s\n\tadapter_size: %d" % (
                adapter_type,
                adapter_size
            ))
            self.bert = AdapterBertModel(config, adapter_type, adapter_size)
        else:
            self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.index_size = index_size
        self.use_mlpencoder = use_mlpencoder
        self.embed_size = self.bert.embeddings.word_embeddings.embedding_dim
        self.top_use_dropout = top_use_dropout
        self.dropout_ratio = dropout_ratio
        self.top_use_layernorm = top_use_layernorm
        self.adapter_type = adapter_type
        self.adapter_size = adapter_size
        self.word_embedding_type = word_embedding_type
        self.word_embedding_hidden_size = word_embedding_hidden_size
        self.word_embedding_dropout = word_embedding_dropout
        self.word_embedding_layernorm = word_embedding_layernorm
        self.top_additional_layer_type = top_additional_layer_type
        self.top_additional_layer_hidden_size = top_additional_layer_hidden_size
        self.init_weights()
        print("========Using dropout in word embedding layer: \033[31m%s\033[0m========" % self.word_embedding_dropout)
        print("========Using layernorm in word embedding layer: \033[31m%s\033[0m========" % self.word_embedding_layernorm)
        if self.word_embedding_type:
            if self.word_embedding_type == "linear":
                print("\033[31m========Using word linear mapping========\033[0m")
                self.word_embedding = nn.Sequential(
                    nn.Linear(self.embed_size, self.embed_size),
                    nn.Dropout(p=self.dropout_ratio) if self.word_embedding_dropout else nn.Identity()
                )
                self.word_embedding[0].weight.data.zero_()
                self.word_embedding[0].bias.data.zero_()
            elif self.word_embedding_type == "double-linear":
                print("\033[31m========Using word double linear mapping========\033[0m")
                assert self.word_embedding_hidden_size is not None
                print("hidden size: %d" % self.word_embedding_hidden_size)
                self.word_embedding = nn.Sequential(
                    nn.Linear(self.embed_size, self.word_embedding_hidden_size),
                    nn.Dropout(p=self.dropout_ratio) if self.word_embedding_dropout else nn.Identity(),
                    nn.LayerNorm(self.word_embedding_hidden_size, eps=config.layer_norm_eps) \
                        if self.word_embedding_layernorm else nn.Identity(),
                    nn.Linear(self.word_embedding_hidden_size, self.embed_size),
                    nn.Dropout(p=self.dropout_ratio) if self.word_embedding_dropout else nn.Identity()
                )
                self.word_embedding[0].weight.data.zero_()
                self.word_embedding[0].bias.data.zero_()
                self.word_embedding[3].weight.data.zero_()
                self.word_embedding[3].bias.data.zero_()
            else:
                print("\033[31m========Using word mlp mapping========\033[0m")
                assert self.word_embedding_hidden_size is not None
                print("hidden size: %d" % self.word_embedding_hidden_size)
                self.word_embedding = nn.Sequential(
                    nn.Linear(self.embed_size, self.word_embedding_hidden_size),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_ratio) if self.word_embedding_dropout else nn.Identity(),
                    nn.LayerNorm(self.word_embedding_hidden_size, eps=config.layer_norm_eps) \
                        if self.word_embedding_layernorm else nn.Identity(),
                    nn.Linear(self.word_embedding_hidden_size, self.embed_size),
                    nn.Dropout(p=self.dropout_ratio) if self.word_embedding_dropout else nn.Identity()
                )
                self.word_embedding[0].weight.data.zero_()
                self.word_embedding[0].bias.data.zero_()
                self.word_embedding[4].weight.data.zero_()
                self.word_embedding[4].bias.data.zero_()
            if self.word_embedding_layernorm:
                self.layernorm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        if self.top_additional_layer_type:
            for num in self.bert.encoder.top_layer_nums:
                if self.top_additional_layer_type == "linear":
                    self.bert.encoder.layer[num].additional_layer[0].weight.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[0].bias.data.zero_()
                elif self.top_additional_layer_type == "double-linear":
                    self.bert.encoder.layer[num].additional_layer[0].weight.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[0].bias.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[3].weight.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[3].bias.data.zero_()
                elif self.top_additional_layer_type == "mlp":
                    self.bert.encoder.layer[num].additional_layer[0].weight.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[0].bias.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[4].weight.data.zero_()
                    self.bert.encoder.layer[num].additional_layer[4].bias.data.zero_()
                else:
                    self.bert.encoder.layer[num].init_new_parameters()
        if self.adapter_type:
            self.bert.init_new_parameters()

        if use_mlm_finetune:
            print("========Using mlm finetune========")
        if use_head_finetune:
            print("========Using head finetune========")
        for param in self.bert.parameters():
            param.requires_grad = use_mlm_finetune
        for param in self.cls.parameters():
            param.requires_grad = use_head_finetune
        if self.top_additional_layer_type:
            self.bert.activate_add_layer_grad()
        if self.adapter_type:
            self.bert.activate_adapter_grad()

        self.template = template

        # load prompt encoder
        self.hidden_size = self.bert.embeddings.word_embeddings.embedding_dim
        self.spell_length = sum(self.template)
        if self.use_mlpencoder:
            self.prompt_encoder = PromptMLPEncoder(self.template, self.hidden_size, self.index_size)
        else:
            self.prompt_encoder = PromptEncoder(self.template, self.hidden_size)

        # special tokens
        self.pseudo_token_id = pseudo_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

    def embed_input(self, input_ids):
        bz = input_ids.shape[0]
        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[(input_ids == self.pseudo_token_id)] = self.unk_token_id
        raw_embeds = self.bert.embeddings.word_embeddings(input_ids_for_embedding)
        if self.prompt_encoder.spell_length == 0:
            return raw_embeds

        blocked_indices = (input_ids == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        ):
        assert input_ids is not None and inputs_embeds is None, "only supports token id input"
        inputs_embeds = self.embed_input(input_ids)
        if self.word_embedding_type:
            inputs_embeds = inputs_embeds + self.word_embedding(inputs_embeds)
            if self.word_embedding_layernorm:
                inputs_embeds = self.layernorm(inputs_embeds)
        outputs = self.bert(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)
