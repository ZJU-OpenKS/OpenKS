import argparse
import pickle
from torch.utils.data import DataLoader
from utils import CrossDomainDataset
from train import *

def parse_args():
    parser = argparse.ArgumentParser(description='our_model')
    parser.add_argument('--cur_dim', type=int, default=100,
                        help='dimension of embeddings')
    parser.add_argument('--dataset', default='meituan',
                        help='dataset name, meituan, amazon, douban can choose')
    parser.add_argument('--n_domain', type=int, default=2,
                        help='number of domains')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of layers for the model')
    parser.add_argument('--n_neigh', nargs='?', default=[10, 5],
                        help='number of neighbor to sample')
    # parser.add_argument('--n_neigh', nargs='?', default=[10],
    #                     help='number of neighbor to sample')
    parser.add_argument('--n_cross', type=int, default=1,
                        help='number of neigh layer cross times')
    parser.add_argument('--n_head_layer', type=int, default=2,
                        help='number of heads for attention per layer')
    parser.add_argument('--n_head_cross', type=int, default=2,
                        help='number of heads for attention cross layers')
    parser.add_argument('--orthloss_weight', type=float, default=1.0,
                        help='weight for the orth loss')
    # parser.add_argument('--kl_weight', type=float, default=1.0,
    #                     help='weight for the kl loss')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for the soft label loss')
    parser.add_argument('--labelloss_weight', type=float, default=1.0,
                        help='weight for the soft label loss')
    parser.add_argument('--hintloss_weight', type=float, default=1.0,
                        help='weight for the hint loss')
    # parser.add_argument('--item_sample', choices=['domain', 'common', 'seperate'], default='seperate', type=str,
    #                     help='item domain sample type')
    parser.add_argument('--n_neg', type=int, default=5,
                        help='number of negative instances to pair with a positive instance')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='dropout rate (1-keep probability)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--patience', type=int, default=20,
                        help='extra iterations before early-stopping')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, metavar='N',
                        help='eval_batch_size')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')
    parser.add_argument('--topk', type=int, default=10,
	                    help="compute metrics@top_k")
    parser.add_argument('--mode', choices=['train', 'test'], default='test', type=str,
                        help='test mode or train mode')

    args = parser.parse_args()
    args.save = args.save + args.dataset
    args.save = args.save + '_batch{}'.format(args.batch_size)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_emb{}'.format(args.cur_dim)
    args.save = args.save + '_layer{}'.format(args.n_layer)
    args.save = args.save + '_neigh{}'.format(' '.join(str(n) for n in args.n_neigh))
    args.save = args.save + '_cross{}'.format(args.n_cross)
    args.save = args.save + '_head_layer{}'.format(args.n_head_layer)
    args.save = args.save + '_head_cross{}'.format(args.n_head_cross)
    # args.save = args.save + '_kl_weight{}'.format(args.kl_weight)
    args.save = args.save + '_orthloss_weight{}'.format(args.orthloss_weight)
    args.save = args.save + '_tempearture{}'.format(args.temperature)
    args.save = args.save + '_labelloss_weight{}'.format(args.labelloss_weight)
    args.save = args.save + '_hintloss_weight{}'.format(args.hintloss_weight)
    # args.save = args.save + '_item_sample{}'.format(args.item_sample)
    args.save = args.save + '_drop{}'.format(args.dropout)
    args.save = args.save + '_negsize{}'.format(args.n_neg)
    args.save = args.save + '_decay{}'.format(args.decay)
    args.save = args.save + '_decaystep{}'.format(args.decay_step)
    args.save = args.save + '_patience{}'.format(args.patience)
    return args

if __name__ == '__main__':
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.device = device
    if args.dataset == 'amazon':
        datapath = '/home/wyf/project/DGML/amazon/'
        # datapath = '/home2/wyf/Projects/DGML/amazon1/'
    elif args.dataset == 'meituan':
        datapath = '/home/wyf/project/DGML/meituan1/'
        # datapath = '/home2/wyf/Projects/DGML/meituan1/'
    elif args.dataset == 'douban':
        # datapath = '/home/wyf/project/DGML/douban/'
        datapath = '/home2/wyf/Projects/DGML/douban/'
    with open(datapath + 'user_item_number.pkl', 'rb') as f:
        args.n_users = pickle.load(f)
        args.n_items = pickle.load(f)
        args.n_domain_items = pickle.load(f)
        args.n_domain_users = pickle.load(f)
    with open(datapath + 'user_item_index.pkl', 'rb') as f:
        alluser_index_table = pickle.load(f)
        allitem_index_table = pickle.load(f)
        args.item_domain_index_tables = pickle.load(f)
        args.user_domain_index_tables = pickle.load(f)
    if args.mode == 'train':
        print('data loading')
        train_data_loader = DataLoader(
            dataset=CrossDomainDataset(datapath, args.n_layer, args.n_neigh, args.n_neg, args.n_domain,
                                       args.user_domain_index_tables, args.item_domain_index_tables, 'train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True)

        valid_data_loader = DataLoader(
            dataset=CrossDomainDataset(datapath, args.n_layer, args.n_neigh, args.n_neg, args.n_domain,
                                       args.user_domain_index_tables, args.item_domain_index_tables, 'valid'),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=True)
        
        test_data_loader = DataLoader(
            dataset=CrossDomainDataset(datapath, args.n_layer, args.n_neigh, args.n_neg, args.n_domain,
                                       args.user_domain_index_tables, args.item_domain_index_tables, 'test'),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=True)
        data_loader = (train_data_loader, valid_data_loader, test_data_loader)
        trainer = Trainer(args, data_loader)
        # trainer.load_model()
        trainer.train()
    else:
        print('data loading')
        test_data_loader = DataLoader(
            dataset=CrossDomainDataset(datapath, args.n_layer, args.n_neigh, args.n_neg, args.n_domain,
                                       args.user_domain_index_tables, args.item_domain_index_tables, 'test'),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=20,
            pin_memory=True)
        trainer = Trainer(args, test_data_loader)
        trainer.load_model()
        trainer.test(visual_att=False, visual_emb=False)