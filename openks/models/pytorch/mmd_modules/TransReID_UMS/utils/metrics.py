
import torch
import torch.nn as nn
import numpy as np
import os
from ..utils.reranking import re_ranking

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_imgpaths, g_imgpaths,
              q_timestamps=None, g_timestamps=None, q_x=None, q_y=None, g_x=None, g_y=None, max_rank=50, if_print=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("matches", matches.shape)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    ln = nn.LayerNorm(normalized_shape=[20])
    locus_data={}
    locus_data["pid"] = []
    locus_data["g_pids"] = []
    locus_data["cid"] = []
    locus_data["time"] = []
    locus_data["x"] = []
    locus_data["y"] = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_imgpath = q_imgpaths[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        #print("#########################")
        #print ("distmat", distmat[q_idx][order][:20])
        #print("q_pid", q_pid)
        #print("g_pids[order]", g_pids[order][:20])
        distmat_tensor = torch.from_numpy(distmat[q_idx][order][:20])
        #distmat_tensor = torch.pow(distmat_tensor, 2)
        distmat_max = torch.max(distmat_tensor).item()
        distmat_tensor = distmat_max - distmat_tensor
        #distmat_tensor = torch.pow(distmat_tensor, 2)
        #distmat_tensor = ln(distmat_tensor)
        # print("distmat_tensor", distmat_tensor)
        score = torch.softmax(distmat_tensor * 2 + 1e-8, dim=0)
        #score = 1 - distmat_tensor.argmax(dim=0)
        #print("score", score)
        #print("score2", torch.softmax(distmat_tensor * 8 + 1e-8, dim=0))
        #print("timestamps", g_timestamps[order][:20])
        #print("g_camids", g_camids[order][:20])
        locus_data["pid"].append(q_pid)
        locus_data["g_pids"].append(g_pids[order][:20])
        locus_data["cid"].append(g_camids[order][:20])
        locus_data["time"].append(g_timestamps[order][:20])
        locus_data["x"].append(g_x[order][:20])
        locus_data["y"].append(g_y[order][:20])
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        if if_print == True:
            print("#########################")
            #print("order", order.shape, order)
            print("q_pid", q_pid)
            print("q_imgpath", q_imgpath)
            print("g_pids[order]", g_pids[order][:50])
            print("g_imgpaths[order]", g_imgpaths[order][:10])

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]#[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()

        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    #print("all_cmc", all_cmc.shape, num_valid_q)
    all_cmc = all_cmc.sum(0) / num_valid_q
    #print("all_cmc", all_cmc[0])
    mAP = np.mean(all_AP)

    return all_cmc, mAP, locus_data


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpaths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, imgpath = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        #self.imgpaths.append(imgpath)
        self.imgpaths.extend(np.asarray(imgpath))

    def compute(self, if_print=False):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        #q_imgpaths = self.imgpaths[:self.num_query]
        q_imgpaths = np.asarray(self.imgpaths[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        #g_imgpaths = self.imgpaths[self.num_query:]
        g_imgpaths = np.asarray(self.imgpaths[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_imgpaths, g_imgpaths, if_print=if_print)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

class R1_mAP_eval_with_time():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval_with_time, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpaths = []
        self.timestamps = []
        self.x = []
        self.y = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, imgpath, timestamp, x, y = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        #self.imgpaths.append(imgpath)
        self.imgpaths.extend(np.asarray(imgpath))
        self.timestamps.extend(np.asarray(timestamp))
        self.x.extend(np.asarray(x))
        self.y.extend(np.asarray(y))

    def compute(self, if_print=False):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_imgpaths = np.asarray(self.imgpaths[:self.num_query])
        q_timestamps = np.asarray(self.timestamps[:self.num_query])
        q_x = np.asarray(self.x[:self.num_query])
        q_y = np.asarray(self.y[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_imgpaths = np.asarray(self.imgpaths[self.num_query:])
        g_timestamps = np.asarray(self.timestamps[self.num_query:])
        g_x = np.asarray(self.x[self.num_query:])
        g_y = np.asarray(self.y[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP, locus_data = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_imgpaths, g_imgpaths,
                    q_timestamps, g_timestamps, q_x, q_y, g_x, g_y, if_print=if_print)


        return cmc, mAP, distmat, self.pids, self.camids, qf, gf, locus_data

