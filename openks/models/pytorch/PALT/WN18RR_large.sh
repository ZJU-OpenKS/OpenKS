export EXP_NAME=WN18RR_large_wordlinear_toplinear_12_23_epoch10_bs_8_lr_3e_5
export DATA_NAME=WN18RR_data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

python run_link_prediction.py \
--do_train \
--data_dir ./data/WN18RR \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--data_cache_dir ${EXP_ROOT}/cache_${DATA_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path bert-large-cased \
--model_type template \
--use_NSP \
--use_mlpencoder \
--word_embedding_type linear \
--top_additional_layer_type linear \
--top_layer_nums 12 23 \
--dropout_ratio 0.1 \
--begin_temp 2 \
--mid_temp 0 \
--end_temp 0 \
--num_neg 5 \
--margin 7 \
--max_seq_length 192 \
--learning_rate 3e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 10 \
--output_dir ${EXP_ROOT}/WN18RR_exps/out_${EXP_NAME} \
--gradient_accumulation_steps 1 \
--save_steps 10854 \
--warmup_steps 32563 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1.
