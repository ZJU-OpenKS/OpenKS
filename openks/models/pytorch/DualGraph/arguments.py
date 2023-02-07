import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DualGraph Arguments.')

    parser.add_argument('--DS', dest='DS', help='Dataset', default='PROTEINS') 

    parser.add_argument("--selector_model",
                        type=str,
                        default="pointwise",
                        choices=["pointwise", "pairwise", "none"],
                        help="Method for selector. 'none' indicates using self-training model")
    parser.add_argument("--integrate_method",
                        type=str,
                        default="intersection",
                        choices=["intersection", "p_only", "s_only"],
                        help="Method to combine results from prediction and retrieval module.")
    parser.add_argument("--selector_upperbound", type=float, default=2, help="# of samples / k taken before intersection.")
    parser.add_argument("--num_iters", type=int, default=-1, help="# of iterations. -1 indicates it's determined by data_ratio.")
    parser.add_argument("--alpha", type=float, default=0.5, help="confidence hyperparameter for predictor.")
    parser.add_argument("--beta", type=float, default=2, help="confidence hyperparameter for selector")

    parser.add_argument("--p_dir", type=str, default="preditor", help="Directory of the predictor.")
    parser.add_argument("--s_dir", type=str, default="selector", help="Directory of the selector.")

    # ratio of instances to promote each round
    parser.add_argument("--sample_ratio", type=float, default=0.2)

    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument("--hidden_dim", type=int, default=128, help="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--p_dropout", type=float, default=0.5, help="Input and RNN dropout rate.")
    parser.add_argument("--s_dropout", type=float, default=0.5, help="Input and RNN dropout rate for selector.")

    parser.add_argument("--lr", type=float, default=0.01, help="Applies to SGD and Adam.")
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='')
    parser.add_argument("--optim", type=str, default="adam", help="sgd, adagrad, rmsprop, adam or adamax.")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--aug", default='subgraph', type=str, help='dnodes, pedges, subgraph, mask_nodes, random2, random3, random4')
    parser.add_argument("--aug_ratio", default=0.5, type=str, help='dnodes, pedges, subgraph, mask_nodes, random2, random3, random4')
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping.")
    parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")
    parser.add_argument("--log", type=str, default="logs.txt", help="Write training log to file.")
    parser.add_argument("--save_epoch", type=int, default=100, help="Save model checkpoints every k epochs.")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Root dir for saving models.")
    # parser.add_argument("--id", type=str, default="00", help="Model ID under which to save models.")
    # parser.add_argument("--info", type=str, default="", help="Optional info for the experiment.")

    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--cuda", type=bool, default=True)
    return parser.parse_args()