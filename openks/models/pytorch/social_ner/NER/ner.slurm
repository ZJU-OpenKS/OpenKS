#!/bin/bash
#SBATCH -J test_ner                           # 作业名
#SBATCH -o ner_20230112.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                 # 作业申请 1 个节点
#SBATCH -t 12:00:00                           # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点

source ~/.bashrc

# 设置运行环境
conda activate ss-ner

# 输入要执行的命令，例如 ./hello 或 python test.py 等
BERT_BASE_DIR=bert-base-chinese
DATA_DIR=./data/
OUTPUT_DIR=./model/20230112

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 30 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn False \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR


