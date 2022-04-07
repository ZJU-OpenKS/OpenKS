import argparse
import os
import numpy as np
import json
from collections import defaultdict


def match_hoi(pred_det, gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i, gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break

    return is_match, remaining_gt_dets


def compute_ap(precision, recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall >= t]
        if selected_p.size == 0:
            p = 0
        else:
            p = np.max(selected_p)
        ap += p / 11.

    return ap


def compute_pr(y_true, y_score, npos):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)

    if len(tp) == 0:
        return 0, 0, False

    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall, True


def load_gt_dets():
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = '/home/xian/Documents/code/GGNet/Dataset/hico_det/annotations/anno_list.json'
    anno_list = json.load(open(anno_list_json, "r"))

    gt_dets = {}
    for anno in anno_list:
        if "test" not in anno['global_id']:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


class HICOEvaluator():
    def __init__(self, preds, gts, subject_category_id, rare_triplets, non_rare_triplets, correct_mat):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        self.preds = []
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })



        self.gts = []
        for img_gts in gts:
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id'}
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (self.gts[-1]['annotations'][hoi['subject_id']]['category_id'],
                           self.gts[-1]['annotations'][hoi['object_id']]['category_id'],
                           hoi['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = [pred_bboxes[pred_hoi['subject_id']]['category_id'],
                               pred_bboxes[pred_hoi['object_id']]['category_id'], pred_hoi['category_id']]
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'],
                           pred_hoi['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0


class hico_val():
    def __init__(self, model_path, model_id):
        self.out_dir = os.path.join(model_path, 'predictions_model_' + str(model_id))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.annotations = load_gt_dets()
        print(len(self.annotations))
        self.hoi_list = json.load(open('//home/xian/Documents/code/GGNet/Dataset/hico_det/annotations/hoi_list_new.json', 'r'))
        self.file_name_to_obj_cat = json.load(
            open('/home/xian/Documents/code/GGNet/Dataset/hico_det/annotations/file_name_to_obj_cat.json', "r"))

        self.global_ids = self.annotations.keys()
        print(len(self.global_ids))
        self.hoi_id_to_num = json.load(open('/home/xian/Documents/code/GGNet/Dataset/hico_det/annotations/hoi_id_to_num.json', "r"))
        self.rare_id_json = [key for key, item in self.hoi_id_to_num.items() if item['rare']]
        print(len(self.rare_id_json))
        self.pred_anno = {}

        self.preds = []
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in
                      zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for
                        subject_id, object_id, category_id, score in
                        zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

    def evaluation_default(self, predict_annot):
        if self.pred_anno == {}:
            pred_anno = {}
            for pre_anno in predict_annot:
                global_id = pre_anno['file_name'].split('.')[0]
                pred_anno[global_id] = {}
                bbox = pre_anno['predictions']
                hois = pre_anno['hoi_prediction']
                for hoi in hois:
                    obj_id = bbox[hoi['object_id']]['category_id']
                    obj_bbox = bbox[hoi['object_id']]['bbox']
                    sub_bbox = bbox[hoi['subject_id']]['bbox']
                    score = hoi['score']
                    verb_id = hoi['category_id']

                    hoi_id = '0'
                    for item in self.hoi_list:
                        if item['object_cat'] == obj_id and item['verb_id'] == verb_id:
                            hoi_id = item['id']
                    assert int(hoi_id) > 0

                    data = np.array([sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                                     obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3],
                                     score]).reshape(1, 9)
                    if hoi_id not in pred_anno[global_id]:
                        pred_anno[global_id][hoi_id] = np.empty([0, 9])

                    pred_anno[global_id][hoi_id] = np.concatenate((pred_anno[global_id][hoi_id], data), axis=0)
            self.pred_anno = pred_anno

        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations, self.pred_anno, self.out_dir)
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        print(mAP['mAP'])
        mAP['invalid'] = len(outputs) - count
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            'mAP_default.json')
        io.dump_json_object(mAP, mAP_json)

        print(f'APs have been saved to {self.out_dir}')

    def evaluation_ko(self, predict_annot):

        if self.pred_anno == {}:
            pred_anno = {}
            for pre_anno in predict_annot:
                global_id = pre_anno['file_name'].split('.')[0]
                pred_anno[global_id] = {}
                bbox = pre_anno['predictions']
                hois = pre_anno['hoi_prediction']
                for hoi in hois:
                    obj_id = bbox[hoi['object_id']]['category_id']
                    obj_bbox = bbox[hoi['object_id']]['bbox']
                    sub_bbox = bbox[hoi['subject_id']]['bbox']
                    score = hoi['score']
                    verb_id = hoi['category_id']

                    hoi_id = '0'
                    for item in self.hoi_list:
                        if item['object_cat'] == obj_id and item['verb_id'] == verb_id:
                            hoi_id = item['id']
                    assert int(hoi_id) > 0

                    data = np.array([sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                                     obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3],
                                     score]).reshape(1, 9)
                    if hoi_id not in pred_anno[global_id]:
                        pred_anno[global_id][hoi_id] = np.empty([0, 9])

                    pred_anno[global_id][hoi_id] = np.concatenate((pred_anno[global_id][hoi_id], data), axis=0)
            self.pred_anno = pred_anno

        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations,
                              self.pred_anno, mode="ko",
                              obj_cate=hoi['object_cat'])
            outputs.append(o)

        mAP = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            mAP['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        mAP['mAP'] = map_ / count
        mAP['invalid'] = len(outputs) - count
        print(count_rare, count_non_rare)
        mAP['mAP_rare'] = map_rare / count_rare
        mAP['mAP_non_rare'] = map_non_rare / count_non_rare

        mAP_json = os.path.join(
            self.out_dir,
            'mAP_ko.json')
        io.dump_json_object(mAP, mAP_json)

        print(f'APs have been saved to {self.out_dir}')

    def eval_hoi(self, hoi_id, global_ids, gt_dets, pred_anno,
                 mode='default', obj_cate=None):
        print(f'Evaluating hoi_id: {hoi_id} ...')
        y_true = []
        y_score = []
        det_id = []
        npos = 0
        for global_id in global_ids:
            if mode == 'ko':
                if global_id + ".jpg" not in self.file_name_to_obj_cat:
                    continue
                obj_cats = self.file_name_to_obj_cat[global_id + ".jpg"]
                if int(obj_cate) not in obj_cats:
                    continue

            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []

            npos += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 9])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]

            sorted_idx = [idx for idx, _ in sorted(
                zip(range(num_dets), hoi_dets[:, 8].tolist()),
                key=lambda x: x[1],
                reverse=True)]
            for i in sorted_idx:
                pred_det = {
                    'human_box': hoi_dets[i, :4],
                    'object_box': hoi_dets[i, 4:8],
                    'score': hoi_dets[i, 8]
                }
                print(hoi_dets[i, 8])
                is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
                y_true.append(is_match)
                y_score.append(pred_det['score'])
                det_id.append((global_id, i))

        # Compute PR
        precision, recall, mark = compute_pr(y_true, y_score, npos)
        if not mark:
            ap = 0
        else:
            ap = compute_ap(precision, recall)
        # Compute AP
        print(f'AP:{ap}')
        return (ap, hoi_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        default="hoidet_hico_ggnet",
        type=str)
    parser.add_argument(
        '--start_epoch',
        default=100,
        type=int)
    parser.add_argument(
        '--end_epoch',
        default=120,
        type=int)
    args = parser.parse_args()
    dir = args.exp
    begin = args.start_epoch
    end = args.end_epoch

    for i in range(begin, end+1):
        model_num = i
        model_dir = f"/home/xian/Documents/code/GGNet/exp/hoidet/{dir}/"
        hoi_eval = hico_eval(f"{model_dir}/", f"{model_num}")
        file = json.load(open(f"{model_dir}/predictions_model_{model_num}.json", "r"))
        hoi_eval.evaluation_default(file)
        hoi_eval.evaluation_ko(file)