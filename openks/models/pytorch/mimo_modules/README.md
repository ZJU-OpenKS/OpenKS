
# Joint Extraction of Fact and Condition Tuples from Sceintific Text


## Introduction
This repository contains source code for the EMNLP 2019 paper " "Multi-Input Multi-Output Sequence Labeling for Joint Extraction of Fact and Condition Tuples from Scientific Text" ([Paper](http://www.meng-jiang.com/pubs/mimo-emnlp19/mimo-emnlp19-paper.pdf)).

## Usage


### 1.Download External Resources

* The `dumped MIMO` can be found [here](https://www.dropbox.com/s/lc1bvoxc2wbut9t/dumped_models.pt?dl=0).

* The `word embedding` we use can be found [here](https://www.dropbox.com/sh/6yx1l8euehgw12k/AAB9mWc3m8H7niuEF7NBYUdRa?dl=0).

* The `pre-trained language model` we use can be found [here](https://www.dropbox.com/sh/q1kehix8q58sxmh/AADU35QFu1ZMuNQFTiEYWSxUa?dl=0).

put these files into `./resources` folder
### 2.Install Requirements
This repo is tested on Python 3.6, PyTorch 1.2.0

Create Environment (Optional): Ideally, you should create an environment for the project.
```bash
conda create -n mimo python=3.6

conda activate mimo

pip install -r requirments.txt
```

### 3.Start a demo application 
```bash
cd MIMO_service

python mimo_server.py #Start a MIMO service

python client.py 
```

The output of the demo is shown below.

```bash
{
	'statements': {
		'stmt 1': {
			'text': 'Histone deacetylase inhibitor valproic acid ( VPA ) has been used to increase the reprogramming efficiency of induced pluripotent stem cell ( iPSC ) from somatic cells , yet the specific molecular mechanisms underlying this effect is unknown .',
			'fact tuples': [
				['Histone deacetylase inhibitor valproic acid', 'NIL', 'has been used to increase', 'induced pluripotent stem cell', 'reprogramming efficiency'],
				['VPA', 'NIL', 'has been used to increase', 'induced pluripotent stem cell', 'reprogramming efficiency'],
				['Histone deacetylase inhibitor valproic acid', 'NIL', 'has been used to increase', 'induced pluripotent stem cell', 'reprogramming'],
				['specific molecular mechanisms', 'NIL', 'is unknown', 'NIL', 'NIL']
			],
			'condition tuples': [
				['iPSC', 'reprogramming efficiency', 'from', 'somatic cells', 'NIL'],
				['induced pluripotent stem cell', 'reprogramming efficiency', 'from', 'somatic cells', 'NIL'],
				['specific molecular mechanisms', 'NIL', 'underlying', 'NIL', 'effect']
			],
			'concept_indx': [0, 1, 2, 3, 4, 6, 17, 18, 19, 20, 22, 25, 26, 30, 31, 32],
			'attr_indx': [14, 15, 35],
			'predicate_indx': [8, 9, 10, 11, 12, 24, 33, 36, 37]
		}
	}
}

```

### 4. Train Your Own MIMO

example commands for pretrain:

(all gates for LM, pretrain)
```bash
python train.py --cuda --config 111000000 --model_name MIMO_BERT_LSTM --pretrain
```


(all gates for POS, pretrain)
```bash
python train.py --cuda --config 000111000 --model_name MIMO_BERT_LSTM --pretrain
```
(all gates for LM and POS, pretrain)
```bash
python train.py --cuda --config 111111000 --model_name MIMO_BERT_LSTM --pretrain
```
example commands with multi-output:

(all gates for LM with multi-output)
```bash
python train.py --cuda --config 111000000 --model_name MIMO_BERT_LSTM
```


(all gates for POS with multi-output)
```bash
python train.py --cuda --config 000111000 --model_name MIMO_BERT_LSTM
```
(all gates for LM and POS, with multi-output)
```bash
python train.py --cuda --config 111111000 --model_name MIMO_BERT_LSTM
```


## Reference
```
@inproceedings{jiang-mimo,
    title = "Multi-Input Multi-Output Sequence Labeling for Joint Extraction of Fact and Condition Tuples from Scientific Text",
    author = "Jiang, Tianwen and Zhao, Tong and Qin, Bing and Liu, Ting and Chawla, Nitesh V and Jiang, Meng",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
}
```

