from vsrl_eval import VCOCOeval


if __name__ == '__main__':
    vcoco = VCOCOeval(vsrl_annot_file="annotations/vcoco_test.json", coco_annot_file="annotations/instances_vcoco_all_2014.json", split_file="data/splits/vcoco_test.ids")
    vcoco._do_eval("vcoco.pickle")
