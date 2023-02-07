#CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/alishop-7c
#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/ml-1m
#CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/ml-100k-by_user
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/ml-100k_1 --save_name saved_model_data.pth
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/ml-100k_1 --save_name saved_model_model.pth --type model --rg_aug 1e3 --mode tst
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/ml-100k_1 --save_name saved_model_model.pth --type model --rg_aug 1e3 --mode trn
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/ml-100k_1 --save_name saved_model.pth --type data --rg_aug 1e1 --mode tst
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/ml-100k_1 --save_name saved_model.pth --type model --rg_aug 1e3 --mode trn

#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/ml-1m_1
#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/amazon_1 --batch 2048


CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/amazon_beauty --batch 1024 --intern 10 --save_name saved_data_beauty.pth --type data --rg_aug 1e-2 --mode trn
#CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/amazon_beauty --batch 1024 --intern 10 --save_name saved_data_beauty.pth --type model --rg_aug 1e4 --mode trn
#CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/amazon_beauty --batch 2048 --save_name saved_model_beauty.pth --type model --rg_aug 1e3 --mode trn
