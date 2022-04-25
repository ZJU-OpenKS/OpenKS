
# Quick start examples for Fleet API

## Collective Training

```
python -m paddle.distributed.launch collective_train.py

python -m openKS_launcher --mode gpu collective_train.py

python -m openKS_launcher --mode gpu openKS_GPU_test.py


```

## Parameter Server Training

```
python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 distributed_train.py

python -m openKS_launcher --mode cpu --worker_num 2 --server_num 2 distributed_train.py

python -m openKS_launcher --mode cpu --worker_num 2 --server_num 2 openKS_CPU_test.py

```

