# Verbs in COCO (V-COCO) Dataset

This repository hosts the Verbs in COCO (V-COCO) dataset and associated code to evaluate models for the Visual Semantic Role Labeling (VSRL) task as ddescribed in <a href=http://arxiv.org/abs/1505.04474>this technical report</a>. 

### Citing
If you find this dataset or code base useful in your research, please consider citing the following papers:

    @article{gupta2015visual,
      title={Visual Semantic Role Labeling},
      author={Gupta, Saurabh and Malik, Jitendra},
      journal={arXiv preprint arXiv:1505.04474},
      year={2015}
    }
    
    @incollection{lin2014microsoft,
      title={Microsoft COCO: Common objects in context},
      author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
      booktitle={Computer Vision--ECCV 2014},
      pages={740--755},
      year={2014},
      publisher={Springer}
    }
    
### Installation
1. Clone repository (recursively, so as to include COCO API).
    ```Shell
    git clone --recursive https://github.com/s-gupta/v-coco.git
    ```

2. This dataset builds off <a href=http://mscoco.org/>MS COCO</a>, please download MS-COCO images and annotations. 

3. Current V-COCO release only uses a subset of MS-COCO images (Image IDs listed in ```data/splits/vcoco_all.ids```). Use the following script to pick out annotations from the COCO annotations to allow faster loading in V-COCO.  
    ```Shell
    # Assume you cloned the repository to `VCOCO_DIR'
    cd $VCOCO_DIR
    # If you downloaded coco annotations to coco-data/annotations
    python script_pick_annotations.py coco-data/annotations
    ```
    
4. Build ```coco/PythonAPI/pycocotools/_mask.so```, ```cython_bbox.so```. 
    ```Shell
    # Assume you cloned the repository to `VCOCO_DIR'
    cd $VCOCO_DIR/coco/PythonAPI/ && make
    cd $VCOCO_DIR && make
    ```

### Using the dataset
1. An IPython notebook, illustrating how to use the annotations in the dataset is available in ```V-COCO.ipynb```
2. The current release of the dataset includes annotations as indicated in Table 1 in the paper. We are collecting role annotations for the 6 categories (that are missing) and will make them public shortly.


### Evaluation
We provide evaluation code that computes ```agent AP``` and ```role AP```, as explained in the paper. 

In order to use the evaluation code, store your predictions as a ```pickle file (.pkl)``` in the following format:
  ```Shell
  [ {'image_id':        # the coco image id,
     'person_box':      #[x1, y1, x2, y2] the box prediction for the person,
     '[action]_agent':  # the score for action corresponding to the person prediction,
     '[action]_[role]': # [x1, y1, x2, y2, s], the predicted box for role and 
                        # associated score for the action-role pair.
     } ]
  ```

Assuming your detections are stored in ```det_file=/path/to/detections/detections.pkl```, do


  ```Shell
  from vsrl_eval import VCOCOeval
  vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    # e.g. vsrl_annot_file: data/vcoco/vcoco_val.json
    #      coco_file:       data/instances_vcoco_all_2014.json
    #      split_file:      data/splits/vcoco_val.ids
  vcocoeval._do_eval(det_file, ovr_thresh=0.5)
  ```

  We introduce two scenarios for ```role AP``` evaluation. 
    
  1. [Scenario 1] In this scenario, for the test cases with missing role annotations an agent role prediction is correct if the action is correct & the overlap between the person boxes is >0.5 & the corresponding role is empty e.g. ```[0,0,0,0]``` or ```[NaN,NaN,NaN,NaN]```. This scenario is fit for missing roles due to occlusion.

  2. [Scenario 2] In this scenario, for the test cases with missing role annotations an agent role prediction is correct if the action is correct & the overlap between the person boxes is >0.5 (the corresponding role is ignored). This scenario is fit for the cases with roles outside the COCO categories.


