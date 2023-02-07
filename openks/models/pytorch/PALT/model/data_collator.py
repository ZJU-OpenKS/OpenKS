import torch

class TempCollator:
    def __init__(self, tokenizer, pseudo_token_id, prompt_temp, mlm_probability=0.15, nsp=False):
        self.predict_mode = False
        self.tokenizer = tokenizer
        self.pseudo_token_id = pseudo_token_id
        self.prompt_temp = prompt_temp #consists of three numbers, corresponding to num of prompts before, between and after
        self.begin_prompt = torch.tensor([pseudo_token_id] * prompt_temp[0], dtype=torch.int64)
        self.mid_prompt = torch.tensor([pseudo_token_id] * prompt_temp[1], dtype=torch.int64)
        self.end_prompt = torch.tensor([pseudo_token_id] * prompt_temp[2], dtype=torch.int64)
        self.prompt_len = sum(prompt_temp)
        self.nsp = nsp

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def process_label(self, batch_data):
        if self.predict_mode:
            keys = list(batch_data.keys())
            for k in keys:
                if 'label' in k:
                    batch_data.pop(k)
        return batch_data

    def __call__(self, examples):
        max_seq_length = examples[0]['input_ids'].size(0)
        for (i, example) in enumerate(examples):
            tmp_input_ids = example['input_ids'].clone()
            assert tmp_input_ids.size(0) == max_seq_length, 'all input length should be the same'
            pos_indicator = example['pos_indicator']
            tmp_length = pos_indicator[-1].item() + 1
            head_length = pos_indicator[1].item() - 1
            rel_length = pos_indicator[3].item() - pos_indicator[2].item() - 1
            head_ids = tmp_input_ids[pos_indicator[0] + 1 : pos_indicator[3]]
            tail_ids = tmp_input_ids[pos_indicator[4] : pos_indicator[5]]
            example['input_ids'] = torch.cat([tmp_input_ids[0:1], self.begin_prompt, \
                                   head_ids, self.mid_prompt, tail_ids, \
                                   self.end_prompt, tmp_input_ids[pos_indicator[5]:]])

            example['attention_mask'] = torch.tensor(
                [1] * (tmp_length + self.prompt_len) + [0] * (max_seq_length - tmp_length),
                dtype=torch.long
            )
            example['token_type_ids'] = torch.tensor(
                [0] * (1 + head_length + rel_length + self.prompt_temp[0] + self.prompt_temp[1]) + [1] * (max_seq_length + self.prompt_temp[2] - 1 - head_length - rel_length),
                dtype=torch.long
            )
            assert len(example['input_ids']) == len(example['attention_mask']) == len(example['token_type_ids'])

        batch_data = {}
        input_keys = list(set(examples[0].keys()) - {'pos_indicator', 'corrupted_part'})
        for k in input_keys:
            batch_data[k] = []
        for (i, example) in enumerate(examples):
            for k in input_keys:
                batch_data[k].append(example[k])
        for k in input_keys:
            batch_data[k] = torch.stack(batch_data[k])
        if self.nsp and 'labels' in batch_data:
            batch_data['next_sentence_label'] = batch_data.pop('labels')

        return self.process_label(batch_data)



class PromptCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, nsp=False):
        self.predict_mode = False
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.nsp = nsp

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def process_label(self, batch_data):
        if self.predict_mode:
            for k in ['labels', 'lm_labels', 'masked_lm_labels', 'mlm_labels', 'span_labels', 'next_sentence_label']:
                if k in batch_data:
                    batch_data.pop(k)
        return batch_data

    def __call__(self, examples):
        max_seq_length = examples[0]['input_ids'].size(0)
        for (i, example) in enumerate(examples):
            tmp_input_ids = example['input_ids'].clone()
            assert tmp_input_ids.size(0) == max_seq_length, 'all input length should be the same'
            pos_indicator = example['pos_indicator']
            tmp_length = pos_indicator[-1].item() + 1
            head_length = pos_indicator[1].item() - 1
            rel_length = pos_indicator[3].item() - pos_indicator[2].item() - 1

            example['attention_mask'] = torch.tensor(
                [1] * tmp_length + [0] * (max_seq_length - tmp_length),
                dtype=torch.long
            )
            example['token_type_ids'] = torch.tensor(
                [0] * (1 + head_length + rel_length) + [1] * (max_seq_length - 1 - head_length - rel_length),
                dtype=torch.long
            )

        batch_data = {}
        input_keys = list(set(examples[0].keys()) - {'pos_indicator', 'corrupted_part'})
        for k in input_keys:
            batch_data[k] = []
        for (i, example) in enumerate(examples):
            for k in input_keys:
                batch_data[k].append(example[k])
        for k in input_keys:
            batch_data[k] = torch.stack(batch_data[k])
        if self.nsp and 'labels' in batch_data:
            batch_data['next_sentence_label'] = batch_data.pop('labels')
        return self.process_label(batch_data)


class PoolingCollator:

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.predict_mode = False
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def set_train_mode(self):
        self.predict_mode = False

    def set_predict_mode(self):
        self.predict_mode = True

    def process_label(self, batch_data):
        if self.predict_mode:
            for k in ['labels', 'lm_labels', 'masked_lm_labels', 'mlm_labels', 'span_labels']:
                if k in batch_data:
                    batch_data.pop(k)
        return batch_data

    def span_mask(self, inputs, mask_left, mask_right):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.')
        labels = inputs.clone()
        if mask_left is None or mask_right is None:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
        else:
            probability_matrix = torch.full(labels.shape, 0.0)
            probability_matrix[mask_left:mask_right] = self.mlm_probability
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        out_of_word_mask = torch.tensor(labels >= len(self.tokenizer), dtype=torch.bool)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix.masked_fill_(out_of_word_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return (inputs, labels)

    def __call__(self, examples):
        max_seq_length = examples[0]['input_ids'].size(0)
        for (i, example) in enumerate(examples):
            tmp_input_ids = example['input_ids'].clone()
            assert tmp_input_ids.size(0) == max_seq_length, 'all input length should be the same'
            pos_indicator = example['pos_indicator']
            tmp_length = pos_indicator[-1].item() + 1
            head_length = pos_indicator[1].item() - 1
            rel_length = pos_indicator[3].item() - pos_indicator[2].item() - 1
            tail_length = pos_indicator[5].item() - pos_indicator[4].item() - 1
            example['attention_mask'] = torch.tensor(
                [1] * tmp_length + [0] * (max_seq_length - tmp_length),
                dtype=torch.long
            )
            example['token_type_ids'] = torch.zeros(max_seq_length, dtype=torch.long)
            example['pooling_head_mask'] = torch.tensor(
                [0] + [1] * head_length + [0] * (max_seq_length - head_length - 1),
                dtype=torch.long
            )
            example['pooling_rel_mask'] = torch.tensor(
                (
                    [0] * (pos_indicator[2].item() + 1)
                    + [1] * rel_length
                    + [0] * (max_seq_length - pos_indicator[2].item() - rel_length - 1)
                ),
                dtype=torch.long
            )
            example['pooling_tail_mask'] = torch.tensor(
                (
                    [0] * (pos_indicator[4].item() + 1)
                    + [1] * tail_length
                    + [0] * (max_seq_length - pos_indicator[4].item() - tail_length - 1)
                ),
                dtype=torch.long
            )
            assert (
                example['pooling_head_mask'].size(0)
                == example['pooling_rel_mask'].size(0)
                == example['pooling_tail_mask'].size(0)
                == max_seq_length
            )
        batch_data = {}
        input_keys = list(set(examples[0].keys()) - {'pos_indicator', 'corrupted_part'})
        for k in input_keys:
            batch_data[k] = []
        for (i, example) in enumerate(examples):
            for k in input_keys:
                batch_data[k].append(example[k])
        for k in input_keys:
            batch_data[k] = torch.stack(batch_data[k])
        return self.process_label(batch_data)
