# Abnormal Community Identify Model


>
> Overall description：this framework is to identify abnormal communities based on user relationship network modeling
> 1) data input：select a variety of relationship point pairs in the KG as input
> 2) modeling：GEMSECWithRegularization, GEMSEC, DeepWalkWithRegularization, DeepWalk
> 3) output：embedding，cluster_mean，log，assignment，group_score
>

## 1、Guide
This section describes the steps required to invoke the framework. The framework directory is described below：

| file name           | illustration                             |
| ------------- | ------------------------------ |
| `config.yml`      | Configuration file, modify the required parameters |
| `abnormal_communiyu_run.py`      | The main function of the framework, including relational network acquisition, algorithm modeling, etc |
| `abnormal_community_model.py` | GEMSECWithRegularization, GEMSEC, DeepWalkWithRegularization, DeepWalk. For the main function call, do not need to change  |
| `utils.py`      | Tools |
| `layers.py` | Deepwalk，Clustering，Regularization. For the model call, do not need to change  |
| `schema.json` | The schema of data  |

### 1） Test sample data set
`testdata\abnormal_community_identify` contains the following subdirectories：
- `relation` : Used to store relational data
- `embeddings` : Used to store embedding results
- `cluster_means` : Used to store cluster-means results
- `logs` : Used to store logs
- `assignments` : Used to store assignments results
- `score` : Used to store community score

### 2） Relationship data introduction

| name           | illustration                             |
| ------------- | ------------------------------ |
| `ID`      | Unique identification |
| `concept` | The type of relation  |
|              Optional                                     |
| `weight` | The weight of relation  |

### 3） Parameter configuration
Modify the required parameters of the framework in the `config.yml` file, as follows：

| Parameter name                    | Default                         | illustration                             |
| ---------------------- | ------------------------ | ------------------------------ |
| `input_path`      | / |the input path|
| `embedding_output`      | / |the output path for embedding result |
| `cluster_mean_output`      | / |the output path for cluster-means result|
| `log_output`      | / |the output path for log, saved as json file|
| `assignment_output`      | / |the output path for result, saved as json file|
| `abnormal_group_score_output`      | / |the output path for group score result, saved as json file|
| `config["dump_matrices"]` | True | Save the embeddings to disk or not. Default is not. |
| `config["model"]` | GEMSEC | The model type |
| `config["P"]` | 1 | Return hyperparameter |
| `config["Q"]` | 1 | In-out hyperparameter |
| `config["walker"]` | 'first' | Random walker order |
| `config["dimensions"]` | 16 | Number of dimensions |
| `config["random_walk_length"]` | 80 | Length of random walk per source |
| `config["num_of_walks"]` | 5 | Number of random walks per source |
| `config["window_size"]` | 5 | Window size for proximity statistic extraction |
| `config["distortion"]` | 0.75 | Downsampling distortion |
| `config["negative_sample_number"]` | 10 | Number of negative samples to draw |
| `config["initial_learning_rate"]` | 0.01 | Initial learning rate |
| `config["minimal_learning_rate"]` | 0.001 | Minimal learning rate |
| `config["annealing_factor"]` | 1 | Annealing factor |
| `config["initial_gamma"]` | 0.1 | Initial clustering weight |
| `config["lambd"]` | 0.0625 | Smoothness regularization penalty |
| `config["cluster_number"]` | 20 | Number of clusters |
| `config["overlap_weighting"]` | 'normalized_overlap' | Weight construction technique for regularization |
| `config["regularization_noise"]` | 1e-8 | Uniform noise max and min on the feature vector distance |


### 4） Operating instruction
Run `abnormal_community_run.py` to start the framework running
```
example：nohup python -u abnormal_community_run.py > xxxx.log 2>&1 &
```
log output：
- 1、Loss and Modularity
- 1、The abnormal score of each community

### 5） Output
After the program runs, the relevant results(embedding, cluster_mean, log, assignment, group_score) will be automatically stored in the configured address and can be accessed on demand

### 6） Example
1. Build the KG data set according to the input data format of OPENKS(https://github.com/ZJU-OpenKS/OpenKS/blob/master/openks/data/README.md) and put it into `openks/data/` 
2. Using the relevant processing modules supported by OPENKS, like entity extraction, relationship extraction, event recognition, and return the structured relation data through KG construction, KG representation completion and KG reasoning
3. Modify the required parameters of the framework in the `config.yml` file
4. Run `abnormal_community_run.py`
