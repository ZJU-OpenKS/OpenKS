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
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_ind import GnnEventModel_withattention_ind
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_rev import GnnEventModel_withattention_rev
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_old import GnnEventModel_withattention_old
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_new import GnnEventModel_withattention_new
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_noeventupdate import GnnEventModel_withattention_noeventupdate
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_nohisupdate import GnnEventModel_withattention_nohisupdate
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withoutupdate import GnnEventModel_withoutupdate
from openks.models.tensorflow.hesne.models.gnn_eventsuccess_withattention_reco import GnnEventModel_withattention_reco
from docopt import docopt


def main():
    args = docopt(__doc__)
    try:
        # model = GnnEventModel_withattention_ind(args)
        model = GnnEventModel_withattention_rev(args)
        # model = GnnEventModel_withattention_old(args)
        # model = GnnEventModel_withattention_new(args)
        # model = GnnEventModel_withattention_noeventupdate(args)
        # model = GnnEventModel_withattention_nohisupdate(args)


        # model = GnnEventModel_withattention_reco(args)
        # model = GnnEventModel_withoutupdate(args)
        # model = GnnEventModel_withattention_sum(args)
        # model = GnnEventModel_withattention(args)
        # model = GnnEventModel_timestamp(args)
        # model = GnnEventModel_changed_new_eventgate(args)
        # model = GnnEventModel_changed_new(args)
        # model = GnnEventModel_changed(args)
        # model = GnnEventModel_changed_gate(args)
        # model = GnnEventModel_changed_ws(args)
        # model = GnnEventModel_learnsuc(args)
        # model = GnnEventModel(args)
        model.train()
        # model.test()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

if __name__ == '__main__':
    main()
