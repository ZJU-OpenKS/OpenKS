#CUDA_VISIBLE_DEVICES=6 python main.py --data ../../data/alishop-7c
#CUDA_VISIBLE_DEVICES=5 python main.py --data ../../data/ml-1m/pro_sg
#CUDA_VISIBLE_DEVICES=5 python main.py --data ../../data/ml-1m
#CUDA_VISIBLE_DEVICES=5 python main.py --data ../../data/ml-100k-by_user
#CUDA_VISIBLE_DEVICES=6 python main.py --data ../../data/ml-100k_1 --save_name saved_model.pth

CUDA_VISIBLE_DEVICES=0 python main.py --data ../../data/amazon_beauty --batch 1024 --intern 10

#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/ml-1m_1
#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/amazon_1 --batch 2048

#