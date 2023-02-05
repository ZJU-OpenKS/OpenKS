import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("start", type=int)
parser.add_argument("root", type=str)
args = parser.parse_args()

gpus = [i for i in range(8)]
workers = [args.start + i for i in range(8)]

assert len(gpus) == len(workers)
for gpu, worker in zip(gpus, workers):
    os.system("export CUDA_VISIBLE_DEVICES=%d; export WORKER_ID=%d; export EXP_ROOT=%s; "
              "nohup bash FB15k237_test.sh >%s 2>&1 &"
              % (gpu, worker, args.root, args.root + "/shell-output-" + str(worker)))
