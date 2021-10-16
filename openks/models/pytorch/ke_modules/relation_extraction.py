import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F
from ...model import TorchModel

BertLayerNorm = torch.nn.LayerNorm

@TorchModel.register("RelationExtraction", "PyTorch")
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[2][1] + outputs[2][-1]

        sub_idx = sub_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        obj_idx = obj_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, sequence_output.shape[2])
        sub_output = torch.gather(sequence_output,1,sub_idx).squeeze(1)
        obj_output = torch.gather(sequence_output,1,obj_idx).squeeze(1)
        
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits