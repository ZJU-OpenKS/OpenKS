import contextlib
from typing import Optional, List

import numpy as np
from dataclasses import dataclass, field


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    pooling_model: bool = field(
        default=False, metadata={"help": "Whether to use mean pooling of text encoding for triplet modeling"}
    )
    text_loss_weight: float = field(
        default=0.1, metadata={"help": "The weight of text loss"}
    )
    pos_weight: Optional[float] = field(
        default=None, metadata={
            "help": "The weight of positive labels in knowledge loss. This should be equal to the number of pos-neg pairs (not the number of negative samples)"
        }
    )
    model_type: Optional[str] = field(default=None)
    use_NSP: bool = field(default=False)
    prompt_len: int = field(default=5)
    use_mlm_finetune: bool = field(default=False)
    use_head_finetune: bool = field(default=False)
    use_mlpencoder: bool = field(default=False)
    word_embedding_type: Optional[str] = field(default=None)
    word_embedding_hidden_size: Optional[int] = field(default=None)
    word_embedding_dropout: bool = field(default=False)
    word_embedding_layernorm: bool = field(default=False)
    top_additional_layer_type: Optional[str] = field(default=None)
    top_additional_layer_hidden_size: Optional[int] = field(default=None)
    top_use_dropout: bool = field(default=False)
    top_layer_nums: Optional[List[int]] = field(default=None)
    dropout_ratio: Optional[float] = field(default=None)
    top_use_layernorm: bool = field(default=False)
    adapter_type: str = field(default=None)
    adapter_size: int = field(default=-1)
    load_checkpoint: bool = field(default=False)
    checkpoint_dir: Optional[str] = field(default=None)
    model_checkpoint_num: Optional[int] = field(default=None)
    not_print_model: bool = field(default=False)


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the examples and features"}
    )
    num_neg: int = field(
        default=1, metadata={"help": "The number of negative samples."}
    )
    margin: float = field(
        default=1., metadata={"help": "The margin of knowledge loss"}
    )
    data_debug: bool = field(
        default=False, metadata={"help": "Whether use only a small part of data for debugging"}
    )
    max_seq_length: int = field(
        default=128, metadata={
            "help": '''The maximum total input sequence length after WordPiece tokenization. 
                                          Sequences longer than this will be truncated, and sequences shorter 
                                          than this will be padded.'''
        }
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
    mask_ratio: float = field(
        default=0.15, metadata={"help": "The ratio of examples to be masked"}
    )
    group_shuffle: bool = field(
        default=False, metadata={
            "help": "Whether use group shuffle such that the positive and negative samples are always in the same batch"
        })
    test_ratio: float = field(
        default=1.0, metadata={"help": "The ratio of test data used to evaluate the performance"}
    )
    type_constrain: bool = field(
        default=False
    )
    data_split: bool = field(default=False)
    num_split: int = field(default=5)
    rank: int = field(default=0)
    only_corrupt_entity: bool = field(default=False)
    pseudo_token: str = field(default='<prompt>')
    begin_temp: int = field(default=5)
    mid_temp: int = field(default=5)
    end_temp: int = field(default=5)
    exp_info: str = field(default="")

    test_count: Optional[int] = field(default=None)
    test_worker_id: Optional[int] = field(default=None)
    filename_info: str = field(default="")
