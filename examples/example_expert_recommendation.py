import argparse
from openks.models import OpenKSModel
from openks.models import ExpertRecModel
import logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Training HGT')

    parser.add_argument('--n_epoch', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--clip', type=int, default=5.0)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--max_project', type=int, default=3,
                        help='max number of projects the project related to')
    parser.add_argument('--n_max_neigh', type=int, default=[5, 10],
                        help='max number of neighs for each layer')
    parser.add_argument('--n_neigh_layer', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='number of head')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--topk', type=int, default=10,
                        help="compute metrics@top_k")
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--patience', type=int, default=10,
                        help='extra iterations before early-stopping')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')

    args = parser.parse_args()
    args.save = args.save + '_n_epoch{}'.format(args.n_epoch)
    args.save = args.save + '_n_hid{}'.format(args.n_hid)
    args.save = args.save + '_n_dim{}'.format(args.n_dim)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + '_max_project{}'.format(args.max_project)
    args.save = args.save + '_n_head{}'.format(args.n_head)
    args.save = args.save + 'clip{}_model.pt'.format(args.clip)
    return args


def main():
    args = parse_args()
    OpenKSModel.list_modules()
    model: ExpertRecModel = OpenKSModel.get_module("PyTorch", "HGTExpertRec")("openks/data/nsf_dblp_kg/nsfkg/", args)
    model.preprocess_data()
    model.load_data_and_model()
    logger.info('Training HGT with #param: %d' % model.get_n_params())
    model.train_expert()
    model.evaluate()
    # model.inference_expert()

    # model.train_team()
    # model.inference_team()


if __name__ == "__main__":
    main()
