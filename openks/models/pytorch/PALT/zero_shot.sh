export EXP_NAME=EMPTY
export DATA_NAME=FB13_data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

python run_triplet_classification.py \
--do_predict \
--data_dir ./data/FB13 \
--per_device_eval_batch_size 64 \
--data_cache_dir ${EXP_ROOT}/cache_${DATA_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path bert-base-cased \
--model_type raw_bert \
--use_NSP \
--dropout_ratio 0.1 \
--begin_temp 0 \
--mid_temp 0 \
--end_temp 0 \
--num_neg 1 \
--only_corrupt_entity \
--margin 7 \
--max_seq_length 192 \
--learning_rate 1.5e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 0 \
--output_dir ${EXP_ROOT}/out_${EXP_NAME} \
--gradient_accumulation_steps 1 \
--save_steps 2417 \
--warmup_steps 2417 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1.
