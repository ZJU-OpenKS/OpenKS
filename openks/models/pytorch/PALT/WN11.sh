export EXP_NAME=WN11_base_wordlinear_6_11_linear_epoch40_bs16_8_lr_1_5_e_4_prompt_2_0_0
export DATA_NAME=WN11_data
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

python run_triplet_classification.py \
--do_train \
--do_predict \
--data_dir ./data/WN11 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 64 \
--data_cache_dir ${EXP_ROOT}/cache_${DATA_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path bert-base-cased \
--model_type template \
--use_NSP \
--use_mlpencoder \
--word_embedding_type linear \
--top_additional_layer_type linear \
--top_layer_nums 6 11 \
--dropout_ratio 0.1 \
--begin_temp 2 \
--mid_temp 0 \
--end_temp 0 \
--num_neg 1 \
--only_corrupt_entity \
--margin 7 \
--max_seq_length 192 \
--learning_rate 1.5e-4 \
--adam_epsilon 1e-6 \
--num_train_epochs 40 \
--output_dir ${EXP_ROOT}/WN11_exps/out_${EXP_NAME} \
--gradient_accumulation_steps 1 \
--save_steps 880 \
--warmup_steps 880 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1. \
--not_print_model
