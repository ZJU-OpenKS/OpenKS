import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='CLERA Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset', default='REDDIT-MULTI-5K') 
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64, help='')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='')
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--aug1", default='pedges', type=str, help='diffusion, dnodes, pedges, subgraph, mask_nodes, random4')
    parser.add_argument("--aug2", default='mask_nodes', type=str, help='diffusion, dnodes, pedges, subgraph, mask_nodes, random4')
    parser.add_argument("--aug_ratio", default=0.2, type=str, help='dnodes, pedges, subgraph, mask_nodes, random4')
    parser.add_argument("--attention", default=False, help='whether to use attention mechanism to compute loss')
    return parser.parse_args()

