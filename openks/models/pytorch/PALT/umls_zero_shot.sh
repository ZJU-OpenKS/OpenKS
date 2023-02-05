export EXP_NAME=umls_zero_shot
export DATA_NAME=umls_data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

python run_link_prediction.py \
--do_predict \
--data_dir ./data/umls \
--per_device_eval_batch_size 512 \
--data_cache_dir ${EXP_ROOT}/cache_${DATA_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path bert-base-cased \
--model_type raw_bert \
--use_NSP \
--top_layer_nums 6 11 \
--dropout_ratio 0.1 \
--begin_temp 0 \
--mid_temp 0 \
--end_temp 0 \
--num_neg 5 \
--margin 7 \
--max_seq_length 192 \
--learning_rate 5e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 10 \
--output_dir ${EXP_ROOT}/THIS_SHOULD_BE_EMPTY \
--gradient_accumulation_steps 1 \
--save_steps 100 \
--warmup_steps 100 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1. \
--not_print_model
