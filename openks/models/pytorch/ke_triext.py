import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from ..model import TripleExtractionModel
import os

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
    tokenizer.add_tokens(new_tokens)
            
def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, unused_tokens=True):
    """
    Build batch from a input text and corresponding entities.
    """
    def get_special_token(w, special_tokens, unused_tokens):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]  

    CLS = "[CLS]"
    SEP = "[SEP]"
    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = [CLS]
        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'],special_tokens,unused_tokens)
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'],special_tokens,unused_tokens)
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'],special_tokens,unused_tokens)
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'],special_tokens,unused_tokens)

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx >= max_seq_length:
                sub_idx = 0
            if obj_idx >= max_seq_length:
                obj_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              sub_idx=sub_idx,
                              obj_idx=obj_idx))
    return features

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1

        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


@TripleExtractionModel.register("TripleExtraction", "PyTorch")
class TripleExtractionTorch(TripleExtractionModel):
	def __init__(self, name='pytorch-default', model=None, args=None):
		self.name = name
		self.args = args
		self.model = model
		self.tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
		self.model.bert.resize_token_embeddings(len(self.tokenizer))

	def data_reader(self):
        #TODO
        return NotImplemented
        # return train_data, dev_data, test_data, label_list, task_ner_labels
    
	def embedding_initial_from_prompt(self, model, tokenizer, special_tokens):
        return NotImplemented

	def save_model(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)
    
    
	def evaluate(self,device, eval_dataloader, eval_label_ids, num_labels):
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        for eval_batch in eval_dataloader:
            eval_batch = tuple(t.to(device) for t in eval_batch)
            input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = eval_batch
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logits = preds[0]
        preds = np.argmax(preds[0], axis=1)
        result = compute_f1(preds, eval_label_ids.numpy())
        result['eval_loss'] = eval_loss
        return preds, result, logits

	def get_features(self, dataset, label2id, special_tokens):
        raw_features = convert_examples_to_features(
            dataset, label2id, self.args.max_seq_length, self.tokenizer, special_tokens, unused_tokens=not(self.args.add_new_tokens))
        all_input_ids = torch.tensor([f.input_ids for f in raw_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in raw_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in raw_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in raw_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in raw_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in raw_features], dtype=torch.long)
        features = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        return features, all_label_ids

	def run(self):
        train_dataset, eval_dataset, test_dataset, label_list, task_ner_labels = self.data_reader()
        setseed(self.args.seed)
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}
        num_labels = len(label_list)
        add_marker_tokens(self.tokenizer, task_ner_labels)
        special_tokens = {}
        train_data, _= self.get_features(train_dataset, label2id, special_tokens)
        self.embedding_initial_from_prompt(self.model,self.tokenizer,special_tokens)
        
        train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size)
        eval_data, eval_label_ids= self.get_features(eval_dataset, label2id)
        eval_dataloader = DataLoader(eval_data, batch_size=self.args.eval_batch_size)
        
        train_batches = [batch for batch in train_dataloader]
        num_train_optimization_steps = len(train_dataloader) * self.args.num_train_epochs
        best_result = None
        eval_step = max(1, len(train_batches) // self.args.eval_per_epoch)
        lr = self.args.learning_rate
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        self.model.to(device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * self.args.warmup_proportion), num_train_optimization_steps)
        global_step = 0
        tr_loss = 0
        nb_tr_steps = 0

        for epoch in range(int(self.args.num_train_epochs)):
            self.model.train()
            random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = batch
                targets = {"relation": label_ids, "sub_idx": sub_idx, "obj_idx": obj_idx}
                loss = self.model(input_ids, input_mask, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    if self.args.do_eval:
                        preds, result, logits = self.evaluate(self.model, device, eval_dataloader, eval_label_ids, num_labels)
                        self.model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = self.args.train_batch_size
                        if (best_result is None) or (result[self.args.eval_metric] > best_result[self.args.eval_metric]):
                            best_result = result
                            self.save_model(self.args.output_dir)


        
        

        

        

    