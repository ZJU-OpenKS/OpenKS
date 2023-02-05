## AD-AUG

This is the pytorch implementation for our ECML-PKDD 2022 [paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_896.pdf):
> Yifan Wang, Yifang Qin, Yu Han, Mingyang Yin, Jingren Zhou, Hongxia Yang, and Ming Zhang(2022). AD-AUG: Adversarial Data Augmentation for
Counterfactual Recommendation

In this paper, we propose a novel counterfactual data augmentation framework, AD-AUG, to mitigate the impact of the imperfect training data and empower
CF model.

Please cite our paper if you use the code.

### Environment Requirement

The code has been tested running under Python 3.9.7. The required packages are as follows:

- pytorch == 1.7.1 
- sklearn == 1.0.2 
- pandas == 1.3.5
- numpy == 1.21.2

### File Structure


```
AD-AUG
├── data
└── model
    ├── AD-CDAE
    ├── AD-MacridVAE
    ├── AD-MultVAE
    ├── CDAE
    ├── MacridVAE
    ├── MultVAE
    ├── SLIM
    └── WMF
```

As listed above, there are two main directories that contain the codes for data preprocessing and model experiments respectively.

Under `./model`, we implement all the compared baselines and the AD-AUG augmented version of the AE-based baselines `X` under the directory named by `AD-X`,

### Running Example

For example, to generate `ML-1M` data for AD-AUG models, run:
```shell
cd ./data && mkdir ml-1m
python process_ml1m.py
```
which will generate three `.csv` data under the directory `./data/ml-1m`.

To conduct experiment on the Data-oriented MacridVAE, run:
```shell
cd ./model/AD-MacridVAE
python main.py --data ../../data/ml-1m --batch 1024 --intern 10 --type data --rg_aug 1e-2
```
Normally when running Model-oriented methods, AD-AUG requires a larger `rg_aug` (e.g. 1e4) to limit the output of the augmenter.

