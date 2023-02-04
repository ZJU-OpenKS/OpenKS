import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='KGNN Arguments')

    parser.add_argument("--DS", dest="DS", help="Dataset", default="REDDIT-MULTI-5K") 
    # "PROTEINS", "DD", "Mutagenicity", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB

    # "PTC_MR", "MUTAG", "NCI1", "ENZYMES", "FRANKENSTEIN"

    parser.add_argument("--base_kernel",
                        type=str,
                        default="subtree_wl",
                        choices=["subtree_wl", "shortest_path", "graphlet"],
                        help="Method for graph kernel")
    parser.add_argument("--integrate_method",
                        type=str,
                        default="intersection",
                        choices=["intersection", "p_only", "q_only", "none"],
                        help="Method to combine results from gnn and gk module.")
    parser.add_argument("--gk_upperbound", type=float, default=2, help="# of samples taken before intersection.")
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--p_dir", type=str, default="gnn", help="Directory of the gnn.")
    parser.add_argument("--q_dir", type=str, default="gk", help="Directory of the gk.")

    parser.add_argument("--wl_iter", type=int, default=0, help="")
    # parser.add_argument("--use_Nystroem", type=bool, default=True) # True False
    # parser.add_argument("--nystroem_dim", type=int, default=32, help="")
    parser.add_argument("--gnn_num_epoch", type=int, default=20, help="")
    parser.add_argument("--memnn_num_epoch", type=int, default=20, help="")
    parser.add_argument("--hidden_dim", type=int, default=64, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=0.01, help="Applies to SGD or Adam.")
    parser.add_argument("--optim", type=str, default="adam", help="sgd, adagrad, rmsprop, adam or adamax.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="")
    parser.add_argument("--p_dropout", type=float, default=0.5, help="gnn dropout rate.")
    parser.add_argument("--q_dropout", type=float, default=0.5, help="gk dropout rate.")
    parser.add_argument('--num_gc_layers', type=int, default=3, help='Number of graph convolution layers before each pooling')
    parser.add_argument("--hops", type=int, default=3, help="hops of memory network")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping.")

    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--log_step", type=int, default=50, help="Print log every k steps.")
    parser.add_argument("--save_epoch", type=int, default=100, help="Save model checkpoints every k epochs.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    return parser.parse_args()