export EXP_NAME=FB13_bert_prompt_NSP
export DATA_NAME=FB13_prompt
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

mkdir -p ${EXP_ROOT}/cache_${EXP_NAME}
python run_triplet_classification.py \
--do_train \
--do_predict \
--data_dir ./data/FB13 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--data_cache_dir ${EXP_ROOT}/cache_${DATA_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path bert-base-cased \
--model_type prompt \
--prompt_len 10 \
--use_NSP \
--num_neg 1 \
--only_corrupt_entity \
--margin 7 \
--max_seq_length 192 \
--learning_rate 3e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 5 \
--output_dir ${EXP_ROOT}/out_${EXP_NAME} \
--gradient_accumulation_steps 1 \
--save_steps 5000 \
--warmup_steps 5000 \
--weight_decay 0.01 \
--text_loss_weight 0.0 \
--test_ratio 1. \
--overwrite_output_dir
