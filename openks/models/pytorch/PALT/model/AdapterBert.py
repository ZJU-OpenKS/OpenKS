import torch.nn as nn
from transformers.modeling_bert import (
    BertEncoder,
    BertModel,
    BertLayer,
    BertAttention,
    BertIntermediate,
    BertLayerNorm,
    BertEmbeddings,
    BertPooler
)


class AdapterBertSelfOutput(nn.Module):
    def __init__(self, config, adapter_type, adapter_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_type = adapter_type

        if adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(config.hidden_size, adapter_size),
                nn.GELU(),
                nn.Linear(adapter_size, config.hidden_size)
            )
        elif adapter_type == "linear":
            self.adapter = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            raise ValueError("No such adapter type")

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states) + hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def activate_adapter_grad(self):
        for param in self.adapter.parameters():
            param.requires_grad = True

    def init_new_parameters(self):
        if self.adapter_type == "mlp":
            self.adapter[0].weight.data.zero_()
            self.adapter[0].bias.data.zero_()
            self.adapter[2].weight.data.zero_()
            self.adapter[2].bias.data.zero_()
        else:
            self.adapter.weight.data.zero_()
            self.adapter.bias.data.zero_()


class AdapterBertAttention(BertAttention):
    def __init__(self, config, adapter_type, adapter_size):
        super().__init__(config)
        self.output = AdapterBertSelfOutput(config, adapter_type, adapter_size)

    def activate_adapter_grad(self):
        self.output.activate_adapter_grad()
    
    def init_new_parameters(self):
        self.output.init_new_parameters()


class AdapterBertOutput(nn.Module):
    def __init__(self, config, adapter_type, adapter_size):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter_type = adapter_type
        if adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(config.hidden_size, adapter_size),
                nn.GELU(),
                nn.Linear(adapter_size, config.hidden_size)
            )
        elif adapter_type == "linear":
            self.adapter = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            raise ValueError("No such adapter type")

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states) + hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def activate_adapter_grad(self):
        for param in self.adapter.parameters():
            param.requires_grad = True
    
    def init_new_parameters(self):
        if self.adapter_type == "mlp":
            self.adapter[0].weight.data.zero_()
            self.adapter[0].bias.data.zero_()
            self.adapter[2].weight.data.zero_()
            self.adapter[2].bias.data.zero_()
        else:
            self.adapter.weight.data.zero_()
            self.adapter.bias.data.zero_()


class AdapterBertLayer(BertLayer):
    def __init__(self, config, adapter_type="mlp", adapter_size=None):
        # ref: https://github.com/google-research/adapter-bert/blob/1a31fc6e92b1b89a6530f48eb0f9e1f04cc4b750/modeling.py#L907
        super().__init__(config)
        self.attention = AdapterBertAttention(
            config, adapter_type, adapter_size)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = AdapterBertAttention(
                config, adapter_type, adapter_size)
        self.intermediate = BertIntermediate(config)
        self.output = AdapterBertOutput(config, adapter_type, adapter_size)

    def activate_adapter_grad(self):
        self.attention.activate_adapter_grad()
        if self.is_decoder:
            self.crossattention.activate_adapter_grad()
        self.output.activate_adapter_grad()
    
    def init_new_parameters(self):
        self.attention.init_new_parameters()
        if self.is_decoder:
            self.crossattention.init_new_parameters()
        self.output.init_new_parameters()


class AdapterBertEncoder(BertEncoder):
    def __init__(self, config, adapter_type, adapter_size):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([
            AdapterBertLayer(config, adapter_type, adapter_size) for _ in range(config.num_hidden_layers)
        ])

    def activate_adapter_grad(self):
        for layer in self.layer:
            layer.activate_adapter_grad()
        
    def init_new_parameters(self):
        for layer in self.layer:
            layer.init_new_parameters()


class AdapterBertModel(BertModel):
    def __init__(self, config, adapter_type, adapter_size):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        print(adapter_type)
        print(adapter_size)
        self.encoder = AdapterBertEncoder(config, adapter_type, adapter_size)
        self.pooler = BertPooler(config)
        self.init_weights()

    def activate_adapter_grad(self):
        self.encoder.activate_adapter_grad()
    
    def init_new_parameters(self):
        self.encoder.init_new_parameters()
