from typing import List, Union

import numpy as np
import paddle
from paddle.utils import download
from paddle.vision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                      ToTensor)

from .model import CLIP
from .simple_tokenizer import SimpleTokenizer

__all__ = ["transform", "tokenize", "build_model"]

MODEL_NAMES = ["RN50", "RN101", "VIT"]

URL = {
    "RN50": "https://bj.bcebos.com/paddleaudio/examples/clip/RN50.pdparams",
    "RN101": "https://bj.bcebos.com/paddleaudio/examples/clip/RN101.pdparams",
    "VIT": "https://bj.bcebos.com/paddleaudio/examples/clip/ViT-B-32.pdparam",
}

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
_tokenizer = SimpleTokenizer()

transform = Compose(
    [
        Resize(224, interpolation="bicubic"),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
        lambda t: t.unsqueeze_(0),
    ]
)


def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = paddle.zeros((len(all_tokens), context_length), dtype="int64")

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(
                f"Input {texts[i]} is too long for context length {context_length}"
            )
        result[i, : len(tokens)] = paddle.Tensor(np.array(tokens))

    return result


def build_model(name="RN50"):
    assert name in MODEL_NAMES, f"model name must be one of {MODEL_NAMES}"
    name2model = {
        "RN101": build_rn101_model,
        "VIT": build_vit_model,
        "RN50": build_rn50_model,
    }
    model = name2model[name]()
    weight = download.get_weights_path_from_url(URL[name])
    sd = paddle.load(weight)
    model.load_dict(sd)
    model.eval()
    return model


def build_vit_model():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    return model


def build_rn101_model():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 23, 3),
        vision_width=64,
        vision_patch_size=0,  # Not used in resnet
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    return model


def build_rn50_model():
    model = CLIP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    return model
