import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, nargs="+")
parser.add_argument("--workers", type=int, nargs="+")
parser.add_argument("--root", type=str)
args = parser.parse_args()

print(args.gpus)
print(args.workers)

gpus = args.gpus
workers = args.workers

assert len(gpus) == len(workers)
for gpu, worker in zip(gpus, workers):
    os.system("export CUDA_VISIBLE_DEVICES=%d; export WORKER_ID=%d; export EXP_ROOT=%s; "
              "nohup bash WN18RR_test.sh >%s 2>&1 &"
              % (gpu, worker, args.root, args.root + "/nohup-shell-output-" + str(worker)))
