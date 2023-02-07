import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset', default='PROTEINS') 
    # "NCI1", "MUTAG", "PROTEINS", "DD", "IMDB-BINARY", "IMDB-MULTI", 
    # "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "PTC_MR", "ENZYMES"
    parser.add_argument("--aug", default='subgraph', type=str, help='dnodes, pedges, subgraph, mask_nodes, random2, random3, random4')
    parser.add_argument("--aug_ratio", default=0.2, type=str, help='dnodes, pedges, subgraph, mask_nodes, random2, random3, random4')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64, help='')
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int, default=3, help='Number of graph convolution layers before each pooling')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of training epochs')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.01, help='')
    parser.add_argument('--tau', dest='tau', type=float, default=1, help='')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--hidden_graphs', type=int, default=16, metavar='N', help='Number of hidden graphs')
    parser.add_argument('--size_hidden_graphs', type=int, default=5, metavar='N', help='Number of nodes of each hidden graph')
    parser.add_argument('--penultimate_dim', type=int, default=32, metavar='N', help='Size of penultimate layer of NN')
    parser.add_argument('--max_step', type=int, default=3, metavar='N', help='Max length of walks')
    parser.add_argument('--normalize', action='store_true', default=False, help='Whether to normalize the kernel values')
    return parser.parse_args()
