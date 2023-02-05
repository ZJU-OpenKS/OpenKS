#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python train.py --config_file='/home1/wyf/Projects/dynamic_network_embedding/code/config_learnsuc_delicious.txt'\
                                       #--log_dir='/home1/wyf/Projects/dynamic_network_embedding/log/yelp'\
                                       #--data_dir='/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv'\
                                       #--save_dir='/home1/wyf/Projects/dynamic_network_embedding/save/yelp'\
                                       #--init_from='/home1/wyf/Projects/dynamic_network_embedding/save/yelp/h_128_b_200_l_0.0005_d_0.98_ds_500.0_k_0.8_learnsuc'
