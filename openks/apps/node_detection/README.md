# Abnormal Node Detection Model


>
> Overall description：this framework is to identify abnormal nodes based on KG
> 1) data input：select the node table and edge table of the KG as input, and type of nodes provides labels
> 2) modeling：The basic graph features extracted according to the input data, and the abnormal nodes identified by random forest model
> 3) output：predict list, model
>

## 1、Guide
This section describes the steps required to invoke the framework. The framework directory is described below：

| file name           | illustration                             |
| ------------- | ------------------------------ |
| `config.yml`      | Configuration file, modify the required parameters |
| `get_abnormal_node_run.py`      | The main function of the framework  |
| `featuresGraph.py` | Graph basic feature generation class. For the main function call, do not need to change  |
| `utils.py`      | Tools |
| `dataset.py` | Data loader function  |
| `schema.json` | The schema of data  |

### 1） Test sample data set
`testdata\abnormal_node_detect` contains the following subdirectories：
- `data` : used to store data
- `model` : used to store model files. After model training, the corresponding model files will be stored at this address
- `result` : used to store the result file

### 2） Parameter configuration
Modify the required parameters of the framework in the `config.yml` file, as follows：

| Parameter name                    | Default                         | illustration                             |
| ---------------------- | ------------------------ | ------------------------------ |
| `dataset_dir`      | / |the path of dataset|
| `model_path`      | / |the path of model|
| `result_path`      | / |the path of result|
| `config["seed"]`      | 1234 |random seed|
| `config["targetAccuracy"]`      | 0.8 |Expected model accuracy|
| `config["TrainTimes"]` | 无 | Expected maximum number of model cycles |
| `config["is_train"]` | 无 | Identify whether training is required. True indicates that training set is required for training and then prediction; False means that the model has been trained and loaded directly for prediction |


### 3） Operating instruction
Run `get_abnormal_node_run.py` to start the framework running
```
example：nohup python -u get_abnormal_node_run.py > xxxx.log 2>&1 &
```

### 4） Output
After the program runs, the relevant results(predict list and model) will be automatically stored in the configured address and can be accessed on demand

### 5） Example
1. Build the KG data set according to the input data format of OPENKS(https://github.com/ZJU-OpenKS/OpenKS/blob/master/openks/data/README.md) and put it into `openks/data/`
2. Using the relevant processing modules supported by OPENKS, like entity extraction, relationship extraction, behavior recognition, and return the structured data through KG construction, KG representation completion and KG reasoning
3. Modify the required parameters of the framework in the `config.yml` file
4. Run `get_abnormal_node_run.py`
