# PALT: Parameter-Lite Transfer of Language Models for Knowledge Graph Completion

The source code repo for paper [PALT: Parameter-Lite Transfer of Language Models for Knowledge Graph Completion](https://arxiv.org/abs/2210.13715).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/palt-parameter-lite-transfer-of-language/link-prediction-on-umls)](https://paperswithcode.com/sota/link-prediction-on-umls?p=palt-parameter-lite-transfer-of-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/palt-parameter-lite-transfer-of-language/link-prediction-on-wn18rr)](https://paperswithcode.com/sota/link-prediction-on-wn18rr?p=palt-parameter-lite-transfer-of-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/palt-parameter-lite-transfer-of-language/link-prediction-on-fb15k-237)](https://paperswithcode.com/sota/link-prediction-on-fb15k-237?p=palt-parameter-lite-transfer-of-language)

## Prepare the data
```bash
cd data
bash prepare_data.sh
cd ..
```

## Prepare the environment
```bash
conda create -n palt python=3.6
conda activate palt
conda install pytorch=1.6.0 cudatoolkit=10.1 -c pytorch
conda install tqdm matplotlib
pip install transformers==3.0.2
```

## Run our model
```bash
bash WN11.sh
bash FB13.sh
bash umls.sh
bash WN18RR.sh
bash FB15k237.sh
```

## Results
### Triplet Classification
![](./imgs/triplet_cla.png)
### Link Prediction
![](./imgs/link_pred.png)
## Citation
```bibtex
@inproceedings{shen-etal-2022-lass,
    title = "PALT: Parameter-Lite Transfer of Language Models for Knowledge Graph Completion",
    author = "Jianhao Shen and Chenguang Wang and Ye Yuan and Jiawei Han and Heng Ji and Koushik Sen and Ming Zhang and Dawn Song",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```
