
import logging
import os
import time
import torch
import torch.nn as nn
from ..utils.meter import AverageMeter
from ..utils.metrics import R1_mAP_eval, R1_mAP_eval_with_time
from ..utils.ums import ums_dist
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
NUM_ENLARGE_SIZE = 100
"""
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
"""
"""
def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 if_print=False):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute(if_print=if_print)
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
"""

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval_with_time(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, imgpath, timestamp) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            #evaluator.update((feat, vid, camid))
                            evaluator.update((feat, pid, camid, imgpath, timestamp))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, pid, camid, camids, target_view, imgpath, timestamp, x, y) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        # x = x.to(device)
                        # y = y.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, pid, camid, imgpath, timestamp, x, y))
                cmc, mAP, _, _, _, _, _, locus_data = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 if_print=False):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval_with_time(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    timestamp_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, timestamp, x, y) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            #x = x.to(device)
            #y = y.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid, imgpath,timestamp, x, y))
            img_path_list.extend(imgpath)
            timestamp_list.extend(timestamp)

    cmc, mAP, _, _, _, _, _, locus_data = evaluator.compute(if_print=if_print)
    locus_pid = torch.tensor(locus_data["pid"], dtype=torch.int64).to(device)
    locus_g_pids = torch.tensor(locus_data["g_pids"], dtype=torch.int64).to(device)
    locus_cid = torch.tensor(locus_data["cid"], dtype=torch.int64).to(device)
    locus_time = torch.tensor(locus_data["time"], dtype=torch.int64).to(device)
    locus_x = torch.tensor(locus_data["x"], dtype=torch.float32).to(device)
    locus_y = torch.tensor(locus_data["y"], dtype=torch.float32).to(device)

    print("##################data_organize##################")
    x_min = torch.min(locus_x.view(-1))
    y_min = torch.min(locus_y.view(-1))
    t_min = torch.min(locus_time.view(-1))
    t_max = torch.max(locus_time.view(-1))
    num = len(locus_pid)
    T_all = []
    locus_time = (locus_time.float() - t_min) / (t_max - t_min)# * NUM_ENLARGE_SIZE
    locus_x = (locus_x - x_min) * NUM_ENLARGE_SIZE
    locus_y = (locus_y - y_min)* NUM_ENLARGE_SIZE
    for i in range(num):
        T = data_organize(locus_pid[i], locus_g_pids[i], locus_cid[i], locus_time[i], locus_x[i], locus_y[i])
        T_all.append(T)
    print("##################compute_score##################")
    scores = torch.zeros(num)
    query_num = 0
    for i in range(num):
        if i > 100000:
            scores[i] = 0
        else:
            scores[i] = ums_dist(T_all[query_num], T_all[i])
        if i != query_num and T_all[query_num][0]["pid"] == T_all[i][0]["pid"]:
            print("scores[i]", scores[i])
            scores[i] = 0

    print("###################print_rank###################")
    locus_time = locus_time * NUM_ENLARGE_SIZE
    sorted_logist, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    _, ind2 = torch.sort(locus_time[query_num], descending=False, dim=-1)
    # x,y放大，为了看得更清楚
    #locus_x = (locus_x - 118) * 1000
    #locus_y = (locus_y - 29) * 1000
    #locus_x = locus_x - 118 * 1000
    #locus_y = locus_y - 29 * 1000
    print("base_pid", locus_pid[query_num])
    print("base_g_pid", locus_g_pids[query_num][ind2])
    print("base_time", locus_time[query_num][ind2])
    print("base_x", locus_x[query_num][ind2])
    print("base_y", locus_y[query_num][ind2])
    for i in range(3):
        ind = sorted_indices[i+1]
        print("##################################################")
        print("match:", i+1)
        print("score:", scores[ind])
        _, ind2 = torch.sort(locus_time[ind], descending=False, dim=-1)
        print("locus_pid", locus_pid[ind])
        print("locus_g_pid", locus_g_pids[ind][ind2])
        print("locus_time", locus_time[ind][ind2])
        print("locus_x", locus_x[ind][ind2])
        print("locus_y", locus_y[ind][ind2])


    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

def data_organize(pid, g_pids, cid, time, x, y):
    T = []
    sorted_logist, sorted_indices = torch.sort(time, descending=False, dim=-1)
    g_pids = g_pids[sorted_indices]
    cid = cid[sorted_indices]
    time = time[sorted_indices]
    x = x[sorted_indices]
    y = y[sorted_indices]
    for i in range(len(g_pids)):
        t = {}
        t["pid"] = pid
        t["g_pid"] = g_pids[i]
        t["cid"] = cid[i]
        t["time"] = time[i]
        t["x"] = x[i]
        t["y"] = y[i]
        T.append(t)
    return T