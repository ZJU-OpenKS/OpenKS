import sys, traceback, pdb
import tensorflow as tf
from models.gnn_linkmodel import GnnEventModel
from docopt import docopt

def get_embedding():
    return

def eval_test():
    args = docopt(__doc__)
    try:
        model = GnnEventModel(args)
        model.eval_test()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    return

if __name__ == 'main':
    get_embedding()
    eval_test()