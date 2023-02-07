export EXP_NAME=THIS_SHOULD_BE_EMPTY
export DATA_NAME=FB13_data
export CUDA_VISIBLE_DEVICES=0,2,5,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

python run_triplet_classification.py \
--do_predict \
--data_dir ./data/FB13 \
--per_device_eval_batch_size 768 \
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
--begin_temp 10 \
--mid_temp 10 \
--end_temp 10 \
--num_neg 1 \
--only_corrupt_entity \
--margin 7 \
--max_seq_length 192 \
--learning_rate 1.5e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 10 \
--output_dir ${EXP_ROOT}/THIS_SHOULD_BE_EMPTY \
--gradient_accumulation_steps 1 \
--save_steps 2471 \
--warmup_steps 2471 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1. \
--not_print_model
--load_checkpoint \
--checkpoint_dir ${EXP_ROOT}/FB13_base \
