# Multi-Event-Combine Model


>
> Overall description：this framework is to identify abnormal nodes based on multi-event-combine modeling
> 1) data input：The basic characteristics and event characteristics of nodes in the KG are selected as heterogeneous event input, and the type of nodes provides labels
> 2) network structure：three Pre-mode of heterogeneous behaviors（`CatEmbdConcat`、`CatOneHotConcat`、`AllEmbdSum`），two kinds of sequence model（`BiLSTM`、`Transformer`）
> 3) output：the score of test dataset，embedding result
>

## 1、Guide
This section describes the steps required to invoke the framework. The framework directory is described below：

| file name           | illustration                             |
| ------------- | ------------------------------ |
| `config.yml`      | Configuration file, modify the required parameters |
| `multi_event_run.py`      | The main function of the framework |
| `utils.py`      | Tools |
| `event_combine_model.py` | Network structure. For the main function call, do not need to change  |
| `event_dataset.py` | Data loader function  |
| `schema.json` | The schema of data  |

### 1） Test sample data set
`testdata\multi_event` contains the following subdirectories：
- `feature` : Used to store heterogeneous event data，divided into training set and test set. If it is a prediction task, it only needs to provide heterogeneous event data of the prediction set
- `label` : Used to store label，divided into training set and test set. If it is a prediction task, only the prediction list needs to be provided
- `model` : Used to store models
- `result` : Used to store results

Specific input path：
- heterogeneous event table path of train set：`data_path` +"/feature/train_feature.csv"，the name of file is "train_feature.csv"
- labels table of train set：`data_path` +"/label/train_label.csv"
- heterogeneous event table path of test set：`data_path` +"/feature/valid_feature.csv"，the name of file is "valid_feature.csv"
- labels table of test set：`data_path` +"/label/valid_label.csv"
- heterogeneous event table path of predict set：`data_path` +"/feature/test_feature.csv"，the name of file is "test_feature.csv"

### 2） heterogeneous event data introduction
Different event data come from different data sources and have different table structures. Unify the following heterogeneous multi event data formats：

| name           | illustration                             |
| ------------- | ------------------------------ |
| `ID`      | Unique number |
| `event_type` | The type of event  |
| `start_time` | The start time of the event. If this field is missing, it is considered an invalid event  |
| `end_time` | The end time of the event  |
| `place` | The place of the event  |
|              Optional                                     |
| `place_1` | district |


The format of heterogeneous event table is as follows:

| name           | illustration                             |
| ------------- | ------------------------------ |
| `ID`      | Unique number |
| `event_time` | The start time of the event, timestamp(ms). Size corresponds to time sequence  |
| `Cat1`~`CatN` | Category columns  |
| `Dense1`~`DenseM` | Numerical columns  |

The format of label table is as follows:

| name           | illustration                             |
| ------------- | ------------------------------ |
| `ID`      | Unique number |
| `label` | label, 0 or 1 |


### 3） Parameter configuration
Modify the required parameters of the framework in the `config.yml` file, as follows：

| Parameter name                    | Default                         | illustration                             |
| ---------------------- | ------------------------ | ------------------------------ |
| `data_path.train_feature_path`      | / |The path of the training set "heterogeneous event table" mentioned above (Note: it can not be set during testing)|
| `data_path.train_label_path`      | / |The training set "label table" path mentioned above (Note: it can not be set during testing)|
| `data_path.valid_feature_path`      | / |The path of the verification set "heterogeneous event table" mentioned above (Note: it can not be set during testing)|
| `data_path.valid_label_path`      | / |The path of the verification set "label table" mentioned above (Note: it can not be set during testing)|
| `data_path.test_feature_path`      | / |The path of heterogeneous event table of prediction set mentioned above (Note: it can not be set during training)|
| `config["cat_num"]` | / | Identify the number of categories in the event details field, as shown in table structure example N above |
| `config["cat_cols"]` | / | Identify the category field name in the event details. The table structure example above is `Cat1`~ `CatN` |
| `config["dense_num"]` | / | Identify the number of numerical features in the event details field, as shown in table structure example M above |
| `config["dense_cols"]` | / | Identify the name of the numerical features in the event details field, as shown in the table structure example above `Dense1`~ `DenseM` |
| `config["seq_len"]` | 24 | The truncation length of the event sequence, generally twice the overall statistical average |
| `config["Epochs"]` | 200 | Maximum number of iterations |
| `config["batch_size"]` | 256 | The size of batch |
| `config["num_workers"]` | 8 | Number of parallels for Dataloader |
| `config["embd_dim"]` | 64 | Embedding dimension of input embedding lookup |
| `config["hidden_size"]` | 128 | The output dimension of the middle end |
| `config["dropout_prob"]` | 0.5 | Dropout probability value in LSTM or transformer |
| `config["num_layers"]` | 2 | The number of layers in LSTM |
| `config["label_dim"]` | 2 | Output dimension. The second category is 2 |
| `config["loss_weight"]` | [0.5,0.5] | Manually set sample weight, [a, b], a represents the weight of 0, B represents the weight of 1, and a + B = 1.0 |
| `config["dropTime"]` | True | Whether to discard time column |
| `config["isStandScaler"]` | True | Whether to use standardscaler for numeric features |
| `config["time_num"]` | 0 | The number of time fields. When config ["droptime"] is true, it defaults to 0 |
| `config["latentSpaces"]` | 1 | Number of hidden spaces. Whether to map to multiple hidden spaces during the conversion of the middle end. Multiple hidden spaces will reduce the effect in the actual measurement, so one is the default |
| `config["seed"]` | 2021 | Random seed |
| `config["EarlyStop_rounds"]` | 20 | Number of early stops |
| `config["num_heads"]` | 4 | For the number of multiple heads used in heterogeneous fusion of attention mask, multiple heads may improve the effect of fusion |
| `config["device"]` | 'cuda:0' | Specifies the GPU card to use |
| `config["Pre_Mode"]` | 'AllEmbdSum' | Three heterogeneous preprocessing modes. By default, 'AllEmbdSum' adopts the AttentionMask mode |
| `config["model_Mode"]` | 'Transformer' | Network structure mode, each preprocessing corresponds to two sequence modeling of BiLSTM[without Attention] and Transformer[with Attention] |
| `config["is_train"]` | True | Identify whether training is required. True indicates that training set is required for training and then prediction; False means that the model has been trained and loaded directly for prediction |
| `config["MaskOrNot"]` | False | Identify whether the AllEmbedSum method needs to mask the alignment part of heterogeneous filling  |

> Additional instructions：
>
> 1. `config["Pre_Mode"]`: there are three pre-processing modes['CatEmbdConcat'  'CatOneHotConcat', 'AllEmbdSum']，Corresponding to concat value after embedding of category, concat value after onehot of category, and aggregation of attention after embedding of all fields
> 2. `config["model_Mode"]`：overall network mode, three pre-processing modes and two post-processing modes['BiLSTM'(without Attention),'Transformer'（Attention）]Six network structures are combined：['CatEmbdConcatModel','CatEmbdConcatAttModel','CatOnehotConcatModel','CatOnehotConcatAttModel','AllEmbdSumModel','AllEmbdSumAttModel']
> 3. The pre-processing mode and network mode should match：the corresponding network modes of 'CatEmbdConcat' are：'CatEmbdConcatModel', 'CatEmbdConcatAttModel'
>

### 4） Operating instruction
Run `multi_event_run.py` to start the framework running
```
example：nohup python -u multi_event_run.py > xxxx.log 2>&1 &
```
log output：：
- 1、After each epoch is executed, the loss result of the training set is output
- 2、Output the loss, AUC, TOPK and other results of the verification set every three epochs
- 3、After triggering the stop condition, output the optimal number of iteration rounds, the optimal verification set AUC, and the average time of each epoch
- 4、Output the TOPK (100,500,2000,10000) index and AUC of the  model in the test set

### 5） Output
After the program runs, the relevant results will be automatically stored in the configured address and can be accessed on demand：
- 1、 Model file：in the `model` folder of `data_path`, stores the mapping dictionary (vocab_dic、StandardScaler model、OneHot model)；And the corresponding network model
- 2、 Rating list：in the `result` folder of `data_path`，stores score list of the test set, named as：`test_watch_time`+"_" +`config["model_Mode"]`
- 3、 Embedding intermediate result：in the `result` of `data_path`，stores the embedded expression of the last layer after the training set and test set pass through the network

### 6） Example
1. Build the KG data set according to the input data format of OPENKS(https://github.com/ZJU-OpenKS/OpenKS/blob/master/openks/data/README.md) and put it into `openks/data/`
2. Using the relevant processing modules supported by OPENKS, like entity extraction, relationship extraction, event recognition, and return the structured event data through KG construction, KG representation completion and KG reasoning
3. Modify the required parameters of the framework in the `config.yml` file
4. Run `multi_event_run.py`
