'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
Changed: Lichen Zhao (zlc1114@buaa.edu.cn)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.config_vqa import CONF
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.loss_helper.loss_vqa import get_loss
# from .eval_helper_v2 import get_eval  # training: too slow!
from .eval_helper import get_eval
from openks.models.pytorch.mmd_modules.ThreeDJCG.utils.eta import decode_eta
from openks.models.pytorch.mmd_modules.ThreeDJCG.lib.pointnet2.pytorch_utils import BNMomentumScheduler


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_related_object_loss: {train_related_object_loss}
[loss] train_vqa_loss: {train_vqa_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_lang_acc: {train_lang_acc}
[sco.] train_vqa_acc: {train_vqa_acc}
[sco.] train_number_acc: {train_number_acc}
[sco.] train_color_acc: {train_color_acc}
[sco.] train_yes_no_acc: {train_yes_no_acc}
[sco.] train_other_acc: {train_other_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou50_mAP: {train_iou50_mAP}, train_iou50_AR: {train_iou50_AR}
[sco.] train_iou_max_rate_0.25: {train_iou_max_rate_25}, train_iou_max_rate_0.5: {train_iou_max_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] now_fetch_time: {now_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] mean_real_time: {mean_real_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_related_object_loss: {train_related_object_loss}
[train] train_vqa_loss: {train_vqa_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_box_loss: {train_box_loss}
[train] train_lang_acc: {train_lang_acc}
[train] train_vqa_acc: {train_vqa_acc}
[train] train_number_acc: {train_number_acc}
[train] train_color_acc: {train_color_acc}
[train] train_yes_no_acc: {train_yes_no_acc}
[train] train_other_acc: {train_other_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou50_mAP: {train_iou50_mAP}, train_iou50_AR: {train_iou50_AR}
[train] train_max_iou25_AR: {train_max_iou25_AR}, train_max_iou50_AR: {train_max_iou50_AR}
[val]   val_loss: {val_loss}
[val]   val_related_object_loss: {val_related_object_loss}
[val]   val_vqa_loss: {val_vqa_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_lang_acc: {val_lang_acc}
[val]   val_vqa_acc: {val_vqa_acc}
[val]   val_number_acc: {val_number_acc}
[val]   val_color_acc: {val_color_acc}
[val]   val_yes_no_acc: {val_yes_no_acc}
[val]   val_other_acc: {val_other_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou50_mAP: {val_iou50_mAP}, val_iou50_AR: {val_iou50_AR}
[val]   val_max_iou25_AR: {val_max_iou25_AR}, val_max_iou50_AR: {val_max_iou50_AR}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] related_object_loss: {related_object_loss}
[loss] vqa_loss: {vqa_loss}
[loss] lang_loss: {lang_loss}
[loss] objectness_loss: {objectness_loss}
[loss] vote_loss: {vote_loss}
[loss] box_loss: {box_loss}
[loss] lang_acc: {lang_acc}
[sco.] vqa_acc: {vqa_acc}
[sco.] number_acc: {number_acc}
[sco.] color_acc: {color_acc}
[sco.] yes_no_acc: {yes_no_acc}
[sco.] other_acc: {other_acc}
[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou50_mAP: {iou50_mAP}, iou50_AR: {iou50_AR}
"""

class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10,
    detection=True, reference=True, use_lang_classifier=True, lr_args=None, bn_args=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__

        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier

        self.lr_args = lr_args
        self.bn_args = bn_args

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "related_object_loss": float("inf"),
            "vqa_loss": float("inf"),
            "lang_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "lang_acc": -float("inf"),
            "vqa_acc": -float("inf"),
            "number_acc": -float("inf"),
            "color_acc": -float("inf"),
            "yes_no_acc": -float("inf"),
            "other_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou50_mAP": -float("inf"),
            "iou50_AR": -float("inf"),
            "max_iou25_AR": -float("inf"),
            "max_iou50_AR": -float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        eval_path = os.path.join(CONF.PATH.OUTPUT, stamp, "eval.txt")
        self.eval_fout = open(eval_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_args:
            config = lr_args
            config['optimizer'] = optimizer
            lr_type = config['type']
            config.pop('type')
            if lr_type == 'cosine':
                self.lr_scheduler = CosineAnnealingLR(**config)
            elif lr_type == 'step':
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                raise NotImplementedError('lr step')
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_args is not None:
            bn_decay_step = bn_args['step']
            bn_decay_rate = bn_args['rate']
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None


    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step
        # base_lr = self.lr_scheduler.get_lr()[0]
        # base_group_lr = [param['lr'] for param in self.optimizer.param_groups]
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                if self.lr_scheduler:
                    # self.lr_scheduler.step()
                    print("learning rate --> {}\n".format(self.lr_scheduler.get_lr()), flush=True)
                    # now_lr = self.lr_scheduler.get_lr()[0]
                    for (idx, param_group) in enumerate(self.optimizer.param_groups):
                        # print(param_group.keys(), '<< param key shape')
                        print('[LR Param Group]', param_group['Param_Name'], param_group['lr'], '<< should', flush=True)
                        # param_group['lr'] = base_group_lr[idx] / base_lr * now_lr

                # feed
                self.dataloader['train'].dataset.shuffle_data()
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()

                if epoch_id % 10 == 0:
                    self._finish(epoch_id+1)
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str, flush=True)

    def _log_eval(self, info_str):
        self.eval_fout.write(info_str + "\n")
        self.eval_fout.flush()
        print(info_str, flush=True)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            "real_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "related_object_loss": [],
            "vqa_loss": [],
            "lang_loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "lang_acc": [],
            "vqa_acc": [],
            "number_acc": [],
            "color_acc": [],
            "yes_no_acc": [],
            "other_acc": [],
            "obj_acc": [],
            "pos_ratio": [],
            "neg_ratio": [],
            "iou50_mAP": [],
            "iou50_AR": [],
            "max_iou25_AR": [],
            "max_iou50_AR": []
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_loss(
            data_dict=data_dict,
            config=self.config,
            detection=self.detection,
            qa=self.reference, 
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        self._running_log["related_object_loss"] = data_dict["related_object_loss"]
        self._running_log["vqa_loss"] = data_dict["vqa_loss"]
        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
            reference=self.reference,
            use_lang_classifier=self.use_lang_classifier
        )

        # dump
        self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        self._running_log["vqa_acc"] = np.mean(data_dict["vqa_acc"])
        self._running_log["number_acc"] = np.mean(data_dict["number_acc"])
        self._running_log["color_acc"] = np.mean(data_dict["color_acc"])
        self._running_log["yes_no_acc"] = np.mean(data_dict["yes_no_acc"])
        self._running_log["other_acc"] = np.mean(data_dict["other_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
        self._running_log["iou50_mAP"] = np.mean(data_dict["mAP_0.5"])
        self._running_log["iou50_AR"] = np.mean(data_dict["AR_0.5"])
        self._running_log["max_iou25_AR"] = np.mean(data_dict["related_object_max_iou25_AR"])
        self._running_log["max_iou50_AR"] = np.mean(data_dict["related_object_max_iou50_AR"])

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)
        """
        total_num = 1000
        self._set_phase("val")
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()
            start = time.time()
            with torch.no_grad():
                # forward
                data_dict["epoch"] = epoch_id
                torch.cuda.synchronize()
                for _ in range(total_num):
                    data_dict = self._forward(data_dict)
                    # self._compute_loss(data_dict)
                end = time.time()
                avg = (end - start) / total_num
                print("##############################")
                print("time", end - start)
                print("avg", avg)
                print("##############################")
        """
        start_solver = time.time()
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda()
                # else:  # lang_feat_raw (the raw text description)
                #     print('not tensor', key)

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "vqa_loss": 0,
                "lang_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                # acc
                "lang_acc": 0,
                "vqa_acc": 0,
                "number_acc": 0,
                "color_acc": 0,
                "yes_no_acc": 0,
                "other_acc": 0,
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "iou50_mAP": 0,
                "iou50_AR": 0,
                "max_iou25_AR": 0,
                "max_iou50_AR": 0
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            # with torch.autograd.set_detect_anomaly(True):
            # forward
            data_dict["epoch_id"] = epoch_id
            start = time.time()
            data_dict = self._forward(data_dict)
            self._compute_loss(data_dict)
            self.log[phase]["forward"].append(time.time() - start)

            # backward
            if phase == "train":
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)

            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["related_object_loss"].append(self._running_log["related_object_loss"].item())
            self.log[phase]["vqa_loss"].append(self._running_log["vqa_loss"].item())
            self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
            self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
            self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
            self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())

            self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
            self.log[phase]["vqa_acc"].append(self._running_log["vqa_acc"])
            self.log[phase]["number_acc"].append(self._running_log["number_acc"])
            self.log[phase]["color_acc"].append(self._running_log["color_acc"])
            self.log[phase]["yes_no_acc"].append(self._running_log["yes_no_acc"])
            self.log[phase]["other_acc"].append(self._running_log["other_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
            self.log[phase]["iou50_mAP"].append(self._running_log["iou50_mAP"])
            self.log[phase]["iou50_AR"].append(self._running_log["iou50_AR"])
            self.log[phase]["max_iou25_AR"].append(self._running_log["max_iou25_AR"])
            self.log[phase]["max_iou50_AR"].append(self._running_log["max_iou50_AR"])

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                real_time = time.time() - start_solver
                self.log[phase]["real_time"].append(real_time)
                start_solver = time.time()
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0: # and self._global_iter_id != 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                if self._global_iter_id % 50 == 0:
                    self._dump_log("train")
                self._global_iter_id += 1


        # check best
        if phase == "val":
            cur_criterion = "iou50_AR"
            cur_criterion_25 = "iou50_mAP"
            cur_best = np.mean(self.log[phase][cur_criterion])
            cur_best_25 = np.mean(self.log[phase][cur_criterion_25])
            if cur_best + cur_best_25 > self.best[cur_criterion] + self.best[cur_criterion_25]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("best {} achieved: {}".format(cur_criterion_25, cur_best_25))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["vqa_loss"] = np.mean(self.log[phase]["vqa_loss"])
                self.best["lang_loss"] = np.mean(self.log[phase]["lang_loss"])
                self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
                self.best["vote_loss"] = np.mean(self.log[phase]["vote_loss"])
                self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
                self.best["lang_acc"] = np.mean(self.log[phase]["lang_acc"])
                self.best["vqa_acc"] = np.mean(self.log[phase]["vqa_acc"])
                self.best["number_acc"] = np.mean(self.log[phase]["number_acc"])
                self.best["color_acc"] = np.mean(self.log[phase]["color_acc"])
                self.best["yes_no_acc"] = np.mean(self.log[phase]["yes_no_acc"])
                self.best["other_acc"] = np.mean(self.log[phase]["other_acc"])
                self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
                self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
                self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
                self.best["iou50_mAP"] = np.mean(self.log[phase]["iou50_mAP"])
                self.best["iou50_AR"] = np.mean(self.log[phase]["iou50_AR"])

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

            det_cur_criterion = "max_iou50_AR"
            det_cur_best = np.mean(self.log[phase][det_cur_criterion])
            if det_cur_best > self.best[det_cur_criterion]:
                self.best["max_iou25_AR"] = np.mean(self.log[phase]["max_iou25_AR"])
                self.best["max_iou50_AR"] = np.mean(self.log[phase]["max_iou50_AR"])
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                #torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "related_object_loss", "vqa_loss", "lang_loss", "objectness_loss", "vote_loss", "box_loss"],
            "score": ["lang_acc", "vqa_acc", "number_acc", "color_acc", "yes_no_acc", "other_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou50_mAP", "iou50_AR", "max_iou25_AR", "max_iou50_AR"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]
        real_time = self.log["train"]["real_time"]

        mean_train_time = np.mean(iter_time)
        mean_train_time = np.mean(real_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_related_object_loss=round(np.mean([v for v in self.log["train"]["related_object_loss"]]), 5),
            train_vqa_loss=round(np.mean([v for v in self.log["train"]["vqa_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_vqa_acc=round(np.mean([v for v in self.log["train"]["vqa_acc"]]), 5),
            train_number_acc=round(np.mean([v for v in self.log["train"]["number_acc"]]), 5),
            train_color_acc=round(np.mean([v for v in self.log["train"]["color_acc"]]), 5),
            train_yes_no_acc=round(np.mean([v for v in self.log["train"]["yes_no_acc"]]), 5),
            train_other_acc=round(np.mean([v for v in self.log["train"]["other_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou50_mAP=round(np.mean([v for v in self.log["train"]["iou50_mAP"]]), 5),
            train_iou50_AR=round(np.mean([v for v in self.log["train"]["iou50_AR"]]), 5),
            train_iou_max_rate_25=round(np.mean([v for v in self.log["train"]["max_iou25_AR"]]), 5),
            train_iou_max_rate_5=round(np.mean([v for v in self.log["train"]["max_iou50_AR"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            now_fetch_time=round(fetch_time[-1], 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            mean_real_time=round(np.mean(real_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        self._log_eval("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_related_object_loss=round(np.mean([v for v in self.log["train"]["related_object_loss"]]), 5),
            train_vqa_loss=round(np.mean([v for v in self.log["train"]["vqa_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_vqa_acc=round(np.mean([v for v in self.log["train"]["vqa_acc"]]), 5),
            train_number_acc=round(np.mean([v for v in self.log["train"]["number_acc"]]), 5),
            train_color_acc=round(np.mean([v for v in self.log["train"]["color_acc"]]), 5),
            train_yes_no_acc=round(np.mean([v for v in self.log["train"]["yes_no_acc"]]), 5),
            train_other_acc=round(np.mean([v for v in self.log["train"]["other_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou50_mAP=round(np.mean([v for v in self.log["train"]["iou50_mAP"]]), 5),
            train_iou50_AR=round(np.mean([v for v in self.log["train"]["iou50_AR"]]), 5),
            train_max_iou25_AR=round(np.mean([v for v in self.log["train"]["max_iou25_AR"]]), 5),
            train_max_iou50_AR=round(np.mean([v for v in self.log["train"]["max_iou50_AR"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_related_object_loss=round(np.mean([v for v in self.log["val"]["related_object_loss"]]), 5),
            val_vqa_loss=round(np.mean([v for v in self.log["val"]["vqa_loss"]]), 5),
            val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_vote_loss=round(np.mean([v for v in self.log["val"]["vote_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
            val_vqa_acc=round(np.mean([v for v in self.log["val"]["vqa_acc"]]), 5),
            val_number_acc=round(np.mean([v for v in self.log["val"]["number_acc"]]), 5),
            val_color_acc=round(np.mean([v for v in self.log["val"]["color_acc"]]), 5),
            val_yes_no_acc=round(np.mean([v for v in self.log["val"]["yes_no_acc"]]), 5),
            val_other_acc=round(np.mean([v for v in self.log["val"]["other_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            val_iou50_mAP=round(np.mean([v for v in self.log["val"]["iou50_mAP"]]), 5),
            val_iou50_AR=round(np.mean([v for v in self.log["val"]["iou50_AR"]]), 5),
            val_max_iou25_AR=round(np.mean([v for v in self.log["val"]["max_iou25_AR"]]), 5),
            val_max_iou50_AR=round(np.mean([v for v in self.log["val"]["max_iou50_AR"]]), 5),
        )
        self._log(epoch_report)
        self._log_eval(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            related_object_loss=round(self.best["related_object_loss"], 5),
            vqa_loss=round(self.best["vqa_loss"], 5),
            lang_loss=round(self.best["lang_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            vote_loss=round(self.best["vote_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            lang_acc=round(self.best["lang_acc"], 5),
            vqa_acc=round(self.best["vqa_acc"], 5),
            number_acc=round(self.best["number_acc"], 5),
            color_acc=round(self.best["color_acc"], 5),
            yes_no_acc=round(self.best["yes_no_acc"], 5),
            other_acc=round(self.best["other_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou50_mAP=round(self.best["iou50_mAP"], 5),
            iou50_AR=round(self.best["iou50_AR"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
