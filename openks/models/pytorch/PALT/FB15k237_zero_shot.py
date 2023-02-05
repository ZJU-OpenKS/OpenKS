import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, nargs="+")
parser.add_argument("--workers", type=int, nargs="+")
parser.add_argument("--root", type=str)
parser.add_argument("--model_type", type=str)
args = parser.parse_args()

print(args.gpus)
print(args.workers)
assert args.model_type in ["base", "large"]

gpus = args.gpus
workers = args.workers

assert len(gpus) == len(workers)
for gpu, worker in zip(gpus, workers):
    os.system("export CUDA_VISIBLE_DEVICES=%d; export WORKER_ID=%d; export EXP_ROOT=%s; export MODEL_TYPE=%s; "
              "nohup bash FB15k237_zero_shot.sh >%s 2>&1 &"
              % (gpu, worker, args.root, args.model_type, args.root + \
                    ("/nohup-shell-output-FB15k237-%s-worker-%s" % (args.model_type, str(worker)))))
