#CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/ml-100k-by_user
#CUDA_VISIBLE_DEVICES=3 python main.py --data ../../data/ml-100k_1
#CUDA_VISIBLE_DEVICES=2 python main.py --data ../../data/ml-1m_1

CUDA_VISIBLE_DEVICES=3 python main.py --data ../../data/amazon_beauty --batch 1024 --intern 10

#CUDA_VISIBLE_DEVICES=5 python main.py --data ../../data/amazon_1 --batch 1024
#CUDA_VISIBLE_DEVICES=1 python main.py --data ../../data/ml-20m --batch 256 --intern 1
