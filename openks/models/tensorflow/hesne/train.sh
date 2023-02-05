#!/usr/bin/env bash
####training
###delicious dataset
#CUDA_VISIBLE_DEVICES=1 python train.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_delicious.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/delicious/h_128_b_64_l_0.005_d_0.98_ds_200.0_k_1.0_random_50_n_5_ns_1_witht'

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/delicious/h_128_b_64_l_0.005_d_0.98_ds_200.0_k_0.8_random_50_n_5_ns_witht'

###lastfm dataset
#CUDA_VISIBLE_DEVICES=2 python train.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_lastfm.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/lastfm/h_128_b_64_l_0.008_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht'

###movielens dataset
#CUDA_VISIBLE_DEVICES=3 python train.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_movielens.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/movielens/h_128_b_32_l_0.01_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht_init'

###mag dataset
CUDA_VISIBLE_DEVICES=2 python train.py --config_file='/home1/wyf/Projects/ecai/code/config_mag.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_random_30_n_5_ns_1_witht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_init'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_random_10_n_5_ns_1_witht'

#########################

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_random_50_n_5_with_attention_nswitht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_random_50_n_5_with_attention_nswitht'

####recomendation
###delicious dataset
#CUDA_VISIBLE_DEVICES=0 python reco.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_delicious_reco.txt' \
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/delicious/h_128_b_64_l_0.005_d_0.98_ds_300.0_k_1.0_random_50_n_5_ns_0_witht_onens1'
###############
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/delicious/h_128_b_64_l_0.008_d_0.98_ds_200.0_k_1.0_random_50_n_5_ns_1_witht'

###lastfm dataset
#CUDA_VISIBLE_DEVICES=2 python reco.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_lastfm_reco.txt' \
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/lastfm/h_128_b_64_l_0.008_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht_onens'
############
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/lastfm/h_128_b_64_l_0.005_d_0.98_ds_500.0_k_1.0_random_20_n_1_ns_0_witht'

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/lastfm/h_128_b_64_l_0.008_d_0.98_ds_500.0_k_1.0_random_50_n_5_ns_1_witht'

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/lastfm/h_128_b_64_l_0.008_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht'

###movielens dataset
#CUDA_VISIBLE_DEVICES=1 python reco.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_movielens_reco.txt' \
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/movielens/h_128_b_32_l_0.01_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht_onens'
#############
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/movielens/h_128_b_32_l_0.015_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/movielens/h_128_b_32_l_0.01_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/movielens/h_128_b_32_l_0.01_d_0.98_ds_500.0_k_1.0_random_50_n_1_ns_0_witht_init'

###mag dataset
#CUDA_VISIBLE_DEVICES=2 python reco.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_mag_reco.txt' \
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_random_50_n_5_ns_1_witht_onens'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_last_20_n_5_ns_1_witht_onens'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_init'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_onens'
##########
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.008_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_init'

####case
###delicious dataset
#CUDA_VISIBLE_DEVICES=0 python case.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_delicious.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/delicious/h_128_b_64_l_0.005_d_0.98_ds_200.0_k_1.0_random_50_n_5_ns_1_witht'

###lastfm dataset
#CUDA_VISIBLE_DEVICES=0 python case.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_lastfm.txt'\

###movielens dataset
#CUDA_VISIBLE_DEVICES=3 python case.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_movielens.txt'\

###mag dataset
#CUDA_VISIBLE_DEVICES=0 python case.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_mag.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0__n_5_ns_1_witht_his'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_update'

###mag dataset reco
#CUDA_VISIBLE_DEVICES=0 python case.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_mag_reco.txt' \

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_0.8_last_10_n_5_ns_1_witht_update'

#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0__n_5_ns_1_witht_his_one'
#--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/mag/h_128_b_64_l_0.005_d_0.98_ds_100.0_k_1.0_random_50_n_5_ns_1_witht_update'







