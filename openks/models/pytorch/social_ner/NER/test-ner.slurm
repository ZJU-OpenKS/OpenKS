#!/bin/bash
#SBATCH -J test-ner
#SBATCH -o test-ner.out
#SBATCH -p compute

#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:titan_xp:1

source ~/.bashrc

source activate ss-ner

BERT_BASE_DIR=/users6/jcchen/pretrained/bert-base-chinese
DATA_DIR=./data/
OUTPUT_DIR=./model/

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn False \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR
