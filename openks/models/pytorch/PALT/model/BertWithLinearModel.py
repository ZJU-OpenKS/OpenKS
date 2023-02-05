from model.AdapterBert import AdapterBertLayer
import torch
import torch.nn as nn
from transformers.modeling_bert import (
    BertEncoder,
    BertModel,
    BertLayer,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayerNorm,
    BertEmbeddings,
    BertPooler
)
# import pdb

class MyBertEncoder(BertEncoder):
    def __init__(self, config, use_dropout,
                 dropout_ratio, use_layernorm,
                 top_additional_layer_type,
                 top_additional_layer_hidden_size,
                 top_layer_nums):
        super().__init__(config)
        self.config = config
        self.top_additional_layer_type = top_additional_layer_type
        self.top_additional_layer_hidden_size = top_additional_layer_hidden_size
        for num in top_layer_nums:
            assert num < config.num_hidden_layers
        self.top_layer_nums = top_layer_nums \
            if top_layer_nums is not None \
                else [config.num_hidden_layers - 1, config.num_hidden_layers // 2]
        if self.top_additional_layer_type == "adapter-module":
            print("\033[31m========Using adapter for the last layer of bert.========\033[0m")
            self.layer = nn.ModuleList([
                MyBertLayer(
                    config,
                    use_dropout=use_dropout,
                    dropout_ratio=dropout_ratio,
                    use_layernorm=use_layernorm,
                    add_layer=False,
                    add_layer_type=top_additional_layer_type,
                    add_hidden_size = top_additional_layer_hidden_size
                ) if (i not in self.top_layer_nums) else AdapterBertLayer(
                    config,
                    adapter_type="linear",
                    adapter_size=self.top_additional_layer_hidden_size
                ) for i in range(config.num_hidden_layers)
            ])
        else:
            self.layer = nn.ModuleList([
                MyBertLayer(
                    config,
                    use_dropout=use_dropout,
                    dropout_ratio=dropout_ratio,
                    use_layernorm=use_layernorm,
                    add_layer=(i in self.top_layer_nums),
                    add_layer_type=top_additional_layer_type,
                    add_hidden_size = top_additional_layer_hidden_size
                ) for i in range(config.num_hidden_layers)
            ])
    
    def activate_add_layer_grad(self):
        if self.top_additional_layer_type == "adapter-module":
            # Currently this should not happen
            assert False
            self.layer[-1].activate_adapter_grad()
        else:
            for layer in self.layer:
                layer.activate_add_layer_grad()


class MyBertLinearModel(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config, use_dropout: bool, dropout_ratio: float, use_layernorm: bool,
                 top_additional_layer_type=None, top_additional_layer_hidden_size=None,
                 top_layer_nums=None):
        super().__init__(config)
        self.config = config
        print("Top dropout: \033[31m%s\033[0m, \033[31m%s\033[0m" % (str(use_dropout), str(dropout_ratio)))
        print("Top layernorm: \033[31m%s\033[0m" % (str(use_layernorm)))

        self.embeddings = BertEmbeddings(config)
        print("Additional layer type: %s" % str(top_additional_layer_type))
        print("Additional layer hidden size: %s" % str(top_additional_layer_hidden_size))
        self.encoder = MyBertEncoder(
            config, use_dropout=use_dropout, dropout_ratio=dropout_ratio, use_layernorm=use_layernorm,
            top_additional_layer_type=top_additional_layer_type,
            top_additional_layer_hidden_size=top_additional_layer_hidden_size,
            top_layer_nums=top_layer_nums
        )
        self.pooler = BertPooler(config)

        self.init_weights()
    
    def activate_add_layer_grad(self):
        self.encoder.activate_add_layer_grad()


class MyBertLayer(BertLayer):
    def __init__(self, config, use_dropout: bool, dropout_ratio: float, use_layernorm: bool,
                 add_layer=False, add_layer_type=None, add_hidden_size=None):
        super().__init__(config)
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.add_layer = add_layer
        self.use_layernorm = use_layernorm
        if self.add_layer:
            if add_layer_type == "linear":
                print("\033[31m========Using linear after this bert layer========\033[0m")
                self.additional_layer = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Dropout(p=dropout_ratio) if use_dropout else nn.Identity(),
                )
            elif add_layer_type == "double-linear":
                print("\033[31m========Using double linear after this bert layer========\033[0m")
                assert add_hidden_size is not None
                print("Hidden size of this layer: %d" % add_hidden_size)
                self.additional_layer = nn.Sequential(
                    nn.Linear(config.hidden_size, add_hidden_size),
                    nn.Dropout(p=dropout_ratio) if use_dropout else nn.Identity(),
                    BertLayerNorm(add_hidden_size, eps=config.layer_norm_eps) \
                        if self.use_layernorm else nn.Identity(),
                    nn.Linear(add_hidden_size, config.hidden_size),
                    nn.Dropout(p=dropout_ratio) if use_dropout else nn.Identity(),
                )
            elif add_layer_type == "mlp":
                print("\033[31m========Using MLP after this bert layer========\033[0m")
                assert add_hidden_size is not None
                print("Hidden size of this layer: %d" % add_hidden_size)
                self.additional_layer = nn.Sequential(
                    nn.Linear(config.hidden_size, add_hidden_size),
                    nn.Tanh(),
                    nn.Dropout(p=dropout_ratio) if use_dropout else nn.Identity(),
                    BertLayerNorm(add_hidden_size, eps=config.layer_norm_eps) \
                        if self.use_layernorm else nn.Identity(),
                    nn.Linear(add_hidden_size, config.hidden_size),
                    nn.Dropout(p=dropout_ratio) if use_dropout else nn.Identity(),
                )
            else:
                raise ValueError()
        if self.use_layernorm:
            self.layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        if self.add_layer:
            # pdb.set_trace()
            linear_output = self.additional_layer(attention_output)
            attention_output = attention_output + linear_output
            if self.use_layernorm:
                attention_output = self.layernorm(attention_output)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
    
    def activate_add_layer_grad(self):
        if self.add_layer:
            for param in self.additional_layer.parameters():
                param.requires_grad = True
