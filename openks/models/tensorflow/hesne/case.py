"""
Usage:
    train.py [options]

Options:
    -h --help                Show this screen.
    --config_file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --save_dir DIR           Save dir name.
    --init_from DIR          Init from name
    --freeze-graph-model     Freeze weights of graph model components.
"""

import sys, traceback, pdb
from models.gnn_eventsuccess_withattention_his import GnnEventModel_withattention_his
from models.gnn_eventsuccess_withattention_update import GnnEventModel_withattention_update
from models.gnn_eventsuccess_withattention_his_reco import GnnEventModel_withattention_his_reco
from models.gnn_eventsuccess_withattention_update_reco import GnnEventModel_withattention_update_reco
from docopt import docopt


def main():
    args = docopt(__doc__)
    try:
        # model = GnnEventModel_withattention_his(args)
        model = GnnEventModel_withattention_update(args)
        # model = GnnEventModel_withattention_his_reco(args)
        # model = GnnEventModel_withattention_update_reco(args)
        model.train()
        # model.test()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == '__main__':
    main()
