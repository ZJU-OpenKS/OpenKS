from __future__ import absolute_import, division, print_function

import sys

from social_ner.common.singleton import SingletonMeta

sys.path.append("./social_ner/NER/")

import logging
import os
import pickle

import torch
from models import BERT_BiLSTM_CRF
from pytorch_transformers import BertTokenizer

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


class predict(metaclass=SingletonMeta):
    def __init__(self):
        output_dir = "./social_ner/NER/model/"
        args = torch.load(os.path.join(output_dir, "training_args.bin"))
        args.output_dir = output_dir

        device = torch.device("cuda")
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
        args.device = device
        n_gpu = torch.cuda.device_count()

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        logger.info(f"args: {args}")

        # logger.info(f"cuda是否可用：{torch.cuda.is_available()}")
        logger.info(f"device: {device} n_gpu: {n_gpu}")

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    args.gradient_accumulation_steps
                )
            )

        with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
            label_list = pickle.load(f)

        with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
            label2id = pickle.load(f)

        id2label = {value: key for key, value in label2id.items()}

        tokenizer = BertTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        model = BERT_BiLSTM_CRF.from_pretrained(
            args.output_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim
        )
        model.to(device)
        model.eval()

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label

    @torch.no_grad()
    def predict(self, x: str):
        sent_list = self.__split_sentence(x)
        result = []
        global_offset = 0

        for sent in sent_list:
            pred_labels = []
            (
                input_tokens,
                input_ids,
                input_mask,
                segment_ids,
            ) = self.__preprocess_sentence(sent)

            input_ids = input_ids.unsqueeze(dim=0).to(self.args.device)
            input_mask = input_mask.unsqueeze(dim=0).to(self.args.device)
            segment_ids = segment_ids.unsqueeze(dim=0).to(self.args.device)

            logits = self.model.predict(input_ids, segment_ids, input_mask)[0]

            for idx in logits:
                pred_labels.append(self.id2label[idx])

            ner_token = ""
            ner_pos_begin = -1
            ner_pos_end = -1
            ner_label = ""
            pos = 0

            for token, label in zip(input_tokens[1:-1], pred_labels[1:-1]):
                logger.debug(f"{pos}: {token} {label}")
                if label == "O":
                    if ner_token:
                        result.append(
                            {
                                "text": ner_token,
                                "type": ner_label,
                                "offset": global_offset + ner_pos_begin,
                                "length": ner_pos_end - ner_pos_begin + 1,
                            }
                        )
                    # 清空实体
                    ner_token = ""
                    ner_pos_begin = -1
                    ner_pos_end = -1
                    ner_label = ""
                else:
                    ner_pos_token, true_label = label.split("-", maxsplit=2)
                    if ner_pos_token == "B":
                        # 保存
                        if ner_token:
                            result.append(
                                {
                                    "text": ner_token,
                                    "type": ner_label,
                                    "offset": global_offset + ner_pos_begin,
                                    "length": ner_pos_end - ner_pos_begin + 1,
                                }
                            )
                        # 新建实体
                        ner_token = token
                        ner_pos_begin = pos
                        ner_pos_end = pos
                        ner_label = true_label
                    elif ner_pos_token == "I":
                        ner_token += token
                        ner_pos_end = pos
                    else:
                        logger.error(f"ner_pos_token {ner_pos_token} must be B/I/O!")
                        raise ValueError
                pos += 1

            global_offset += len(sent)

        return result

    def __split_sentence(self, x: str):
        seps = "。！？.!?\n"
        sep_token = "[SEPTOKEN]"
        for sep in seps:
            x = x.replace(sep, sep + sep_token)
        sent_list = x.split(sep_token)
        return [sent for sent in sent_list if sent]

    def __preprocess_sentence(self, sent: str):
        ori_token_list = [ch for ch in sent]
        
        ori_token_list = ["[CLS]"] + ori_token_list + ["[SEP]"]

        token_list = [self.tokenizer.tokenize(x)[0] for x in ori_token_list]
        token_id_list = self.tokenizer.convert_tokens_to_ids(token_list)

        seq_len = len(token_id_list)

        input_ids = torch.zeros(self.args.max_seq_length, dtype=torch.long)
        input_ids[:seq_len] += torch.tensor(token_id_list, dtype=torch.long)

        input_mask = torch.zeros(self.args.max_seq_length, dtype=torch.long)
        input_mask[:seq_len] = 1

        segment_ids = torch.zeros(self.args.max_seq_length, dtype=torch.long)

        return ori_token_list, input_ids, input_mask, segment_ids


if __name__ == "__main__":
    p = predict()
    result = p.predict("如果要问目前人类武库中有哪种武器可算是“顶级”？答案非美海军俄亥俄级弹道导弹核潜艇莫属。")
    print(result)
