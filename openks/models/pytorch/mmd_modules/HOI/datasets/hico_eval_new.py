import argparse
import os
import numpy as np
# import sys
#
# sys.path.insert(0, "/home/xian/Documents/code/GGNet/src/lib")
# from utils.bbox import compute_iou
import json
import util.io as io


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


def compute_area(bbox, invalid=None):
    x1, y1, x2, y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


def compute_iou(bbox1, bbox2, verbose=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)

    area1 = compute_area(bbox1, invalid=0.0)
    area2 = compute_area(bbox2, invalid=0.0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou


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
    anno_list_json = '/data1/weiyunfei/project/qpic/data/hico_20160224_det/hico-det/anno_list.json'
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


class HICOEval():
    def __init__(self, model_path, model_id):
        self.out_dir = os.path.join(model_path, 'predictions_model_' + str(model_id))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.annotations = load_gt_dets()
        print(len(self.annotations))
        self.hoi_list = json.load(open('/data1/weiyunfei/project/qpic/data/hico_20160224_det/hico-det/hoi_list_new.json', 'r'))
        self.file_name_to_obj_cat = json.load(
            open('/data1/weiyunfei/project/qpic/data/hico_20160224_det/hico-det/file_name_to_obj_cat.json', "r"))

        self.global_ids = self.annotations.keys()
        print(len(self.global_ids))
        self.hoi_id_to_num = json.load(open('/data1/weiyunfei/project/qpic/data/hico_20160224_det/hico-det/hoi_id_to_num.json', "r"))
        self.rare_id_json = [key for key, item in self.hoi_id_to_num.items() if item['rare']]
        print(len(self.rare_id_json))
        self.pred_anno = {}

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
        model_dir = f"/data1/weiyunfei/project/qpic/"
        hoi_eval = HICOEvaluator(f"{model_dir}/", f"{model_num}")
        file = json.load(open(f"{model_dir}/predictions_model_{model_num}.json", "r"))
        hoi_eval.evaluation_default(file)
        hoi_eval.evaluation_ko(file)