import torch
import os.path as op
import argparse
import json
from tqdm import tqdm
from typing import Callable, Optional, Tuple
from easydict import EasyDict

from .mmd_modules.det_sgg.relation_predictor.relation_predictor import RelationPredictor
from .mmd_modules.det_sgg.relation_predictor.AttrRCNN import AttrRCNN
from .mmd_modules.det_sgg.maskrcnn_benchmark.data.transforms import build_transforms
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from .mmd_modules.det_sgg.maskrcnn_benchmark.config import cfg
from .mmd_modules.det_sgg.relation_predictor.config import sg_cfg
from .mmd_modules.det_sgg.maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from .mmd_modules.det_sgg.maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from .mmd_modules.det_sgg.maskrcnn_benchmark.utils.miscellaneous import mkdir

from openks.mm.graph import MMGraph
from openks.mm.graph import ImageViewEntity, HasEntity, Interact

class SGG:
    """
    A wrapper class for adjusting graph input
    """
    def __init__(self, args={}):
        parser = argparse.ArgumentParser(description="SGG for Graph")
        args_ = {
            'opts': ['MODEL.DEVICE', 'cuda', 
                     'DATASETS.LABELMAP_FILE', 'visualgenome/VG-SGG-dicts-danfeiX-clipped.json'],
            'mode': 'relation',
            'ckpt': "",
        }
        
        args_.update(args)
        args = EasyDict(args_)
        if args.mode == 'entity':
            args.config_file = 'openks/models/pytorch/mmd_modules/det_sgg/sgg_configs/vgattr/vinvl_x152c4.yaml'
        elif args.mode  == 'relation':
            args.config_file = 'openks/models/pytorch/mmd_modules/det_sgg/sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml'
        else:
            raise NotImplementedError(args)
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        output_dir = cfg.OUTPUT_DIR
        mkdir(output_dir)

        self.device = cfg.MODEL.DEVICE

        if cfg.MODEL.META_ARCHITECTURE == "RelationPredictor":
            self.model = RelationPredictor(cfg)
        elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)
        self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()
        
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)

        # dataset labelmap is used to convert the prediction to class labels
        dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                    cfg.DATASETS.LABELMAP_FILE)
        assert dataset_labelmap_file
        self.dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
        self.dataset_labelmap = {int(val): key
                            for key, val in self.dataset_allmap['label_to_idx'].items()}
        # # visual_labelmap is used to select classes for visualization
        # try:
        #     self.visual_labelmap = load_labelmap_file(args_.labelmap_file)
        # except:
        #     self.visual_labelmap = None

        # if cfg.MODEL.ATTRIBUTE_ON:
        #     self.dataset_attr_labelmap = {
        #             int(val): key for key, val in
        #             self.dataset_allmap['attribute_to_idx'].items()}
        
        if cfg.MODEL.RELATION_ON:
            self.dataset_relation_labelmap = {
                    int(val): key for key, val in
                    self.dataset_allmap['predicate_to_idx'].items()}

        self.transforms = build_transforms(cfg, is_train=False)

    def preprocess(self, image):
        img_input, _ = self.transforms(image, target=None)
        img_input = img_input.to(self.device)
        return img_input

    def det_single_image(self, entity):
        # img_file = entity.file_name.name
        # image = self.read_img(img_file)
        image = self.preprocess(entity.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
            prediction = prediction[0].to(torch.device("cpu"))

        img_height, img_width = entity.image.size

        prediction_pred = prediction.prediction_pairs
        relations = prediction_pred.get_field("idx_pairs").tolist()
        relation_scores = prediction_pred.get_field("scores").tolist()
        predicates = prediction_pred.get_field("labels").tolist()
        prediction = prediction.predictions

        prediction = prediction.resize((img_width, img_height))
        boxes = prediction.bbox.tolist()
        classes = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()

        rt_box_list = []
        if 'attr_scores' in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist()}
                for box, cls, score, attr, attr_conf in
                zip(boxes, classes, scores, attr_labels, attr_scores)
            ]
        else:
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score}
                for box, cls, score in
                zip(boxes, classes, scores)
            ]
        rt_relation_list = [{"subj_id": relation[0], "obj_id":relation[1], "class": predicate+1, "conf": score}
                for relation, predicate, score in
                zip(relations, predicates, relation_scores)]
        return {'objects': rt_box_list, 'relations':rt_relation_list}

    def object2entity(self, obj_id, objects, graph, entity, added) -> ImageViewEntity:
        obj_entity = None
        if not added[obj_id]:
            obj = objects[obj_id]
            x0, x1, y0, y1 = obj["rect"]
            label = self.dataset_labelmap[obj["class"]]
            score = obj["conf"]
            obj_entity = ImageViewEntity(entity, x0, x1, y0, y1, label, score)
            hasRel = HasEntity(entity, obj_entity)
            graph.add_relation(hasRel)
            added[obj_id] = obj_entity.id
        else:
            obj_entity = graph.get_entity_by_id(added[obj_id])
        return obj_entity

    def singleobject2entity(self, obj_id, obj, graph, entity, added) -> ImageViewEntity:
        obj_entity = None
        if not added[obj_id]:
            x0, y0, x1, y1 = obj["rect"]
            label = self.dataset_labelmap[obj["class"]]
            score = obj["conf"]
            obj_entity = ImageViewEntity(entity, x0, y0, x1, y1, label, score)
            hasRel = HasEntity(entity, obj_entity)
            graph.add_relation(hasRel)
            added[obj_id] = obj_entity.id
        else:
            obj_entity = graph.get_entity_by_id(added[obj_id])
        return obj_entity

    # TODO: support large scale dataset
    def __call__(self, graph: Optional[MMGraph] = None) -> MMGraph:
        total_objects = []
        total_relations = []
        for entity in tqdm(graph.get_entities_by_concept("image")):
            dets = self.det_single_image(entity)
            objects = dets['objects']
            relations = dets['relations']
            total_objects.append(objects)
            total_relations.append(relations)

        for objects, relations in zip(total_objects, total_relations):
            added = [None] * len(objects)
            for rel in relations:
                    subj_id = rel["subj_id"]
                    obj_id = rel["obj_id"]
                    predicate = self.dataset_relation_labelmap[rel["class"]]
                    rel_conf = rel["conf"]

                    sub_entity = self.object2entity(subj_id, objects, graph, entity, added)
                    obj_entity = self.object2entity(obj_id, objects, graph, entity, added)
                    
                    rel_pred = Interact(sub_entity, obj_entity, predicate=predicate, score=rel_conf)
                    graph.add_relation(rel_pred)
                    
            # whether there is no relation
            for i, obj in enumerate(objects):
                obj_entity = self.singleobject2entity(i, obj, graph, entity, added)
                

        return graph
