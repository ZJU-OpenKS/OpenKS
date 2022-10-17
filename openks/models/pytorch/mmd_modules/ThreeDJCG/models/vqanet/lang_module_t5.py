import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from openks.models.pytorch.mmd_modules.ThreeDJCG.models.transformer.attention import MultiHeadAttention
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from .modified_t5 import T5Model

class LangModule(nn.Module):
    # def __init__(self, num_text_classes, answer_class_number=2000, model_name='t5', pretrained_path='/mnt/lustre/zhaolichen/codes/3dvl/t5_base',
    def __init__(self, num_text_classes, answer_class_number=2000, model_name='t5', pretrained_path='t5-base',
                 use_lang_classifier=True, hidden_size=768):
        super().__init__()

        self.model_name = model_name
        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier

        if model_name in ['t5']:
            self.model = T5Model(pretrained_path)
        else:
            raise NotImplementedError(model_name)

        # language classifier
        if use_lang_classifier:
            self.query_cls = nn.Linear(hidden_size, num_text_classes)
            # self.answering = nn.GRU(
            #     input_size = hidden_size,
            #     hidden_size = hidden_size,
            #     batch_first=True,
            #     bidirectional=True
            # )
            self.answering_fc = nn.Linear(hidden_size, answer_class_number)
            self.lang_sem_cls_class = nn.Linear(hidden_size, answer_class_number)

    def encode(self, data_dict):
        """
        encode the input descriptions
        """
        word_embs = data_dict["vqa_question_embedding"]  # B * 32 * MAX_DES_LEN * LEN(300)
        lang_len = data_dict["vqa_question_embedding_length"]
        device = word_embs.device

        # todo: there are xx objects in the scene (more info)
        query, answer = data_dict['vqa_question'], data_dict['vqa_answer']
        batch_size, lang_num_max = len(query), len(query[0])
        query = [x for y in query for x in y]
        answer = [x for y in answer for x in y]

        tokens = self.model.tokenize_forward(query, answer, device)
        encoder_outputs = self.model.encoder_forward(input_ids=tokens['input_ids'], return_dict=True)
        hidden_state = encoder_outputs.last_hidden_state  # same as decoder_outputs.encoder_last_hidden_state
        data_dict['tokens'] = tokens
        data_dict['t5_encoder_outputs'] = encoder_outputs
        data_dict['vqa_question_attention_mask'] = None
        data_dict['vqa_question_lang_fea'] = hidden_state
        return data_dict

    def decode(self, data_dict):
        query, answer = data_dict['vqa_question'], data_dict['vqa_answer']
        batch_size, lang_num_max = len(query), len(query[0])
        tokens = data_dict['tokens']
        encoder_outputs = data_dict['t5_encoder_outputs']
        updated_lang_fea = data_dict['updated_lang_fea']
        # use the updated language feature
        encoder_outputs.hidden_state = updated_lang_fea

        # Note: The input tokens are trained with an seq-to-seq model, so we could only use the hidden_state[:, 0, :]
        outputs = self.model.decoder_forward(labels=tokens['labels'], return_dict=True, encoder_outputs=encoder_outputs, output_hidden_states=True)

        # embs = self.model.generate(**tokens)
        # import ipdb; ipdb.set_trace();
        # lang_feat_mask = torch.zeros_like(query_tokens).bool()
        # lang_feat_mask[query_tokens == self.tokenzier.tokenizer.pad_token_id] = True

        loss, logits, hidden_state = outputs.loss, outputs.logits, outputs.decoder_hidden_states[-1]
        prediction_answer = [[x for x in y] for y in logits.argmax(-1)]
        prediction_answer = [self.model.tokenizer.decode(x) for x in prediction_answer]  # string

        # import ipdb; ipdb.set_trace()
        # cap_emb = hidden_state[:, 0, :]
        # cap_emb = hidden_state[:, 0, :]
        # store the encoded language features
        data_dict["vqa_question_lang_decoded_fea"] = hidden_state  # B, hidden_size

        # classify
        if self.use_lang_classifier:
            # We Only Use The Feature[0] for classification (seq2seq, cls token)
            data_dict["vqa_question_lang_scores"] = self.query_cls(hidden_state[:, 0, :]).reshape(batch_size, lang_num_max, -1)
            last_feat = hidden_state[:, 0, :].reshape(batch_size, lang_num_max, -1)
            pred_answer = self.answering_fc(last_feat)
            pred_lang_sem_cls = self.lang_sem_cls_class(last_feat).reshape(batch_size, lang_num_max, -1, 4)
            data_dict["vqa_pred_answer"] = pred_answer
            data_dict["vqa_pred_lang_sem_cls"] = pred_lang_sem_cls

        return data_dict

    def forward(self, data_dict):
        raise NotImplementedError('Cross-Modal-Attention is needed!')

