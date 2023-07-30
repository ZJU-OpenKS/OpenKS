import logging
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from ..model import VisualConstructionModel
from .visual_entity_modules import clip
from .visual_entity_modules.datasets import loaddata
from .visual_entity_modules.newbert_model import TransformerBiaffine as Model
from PIL import Image
import argparse
from pathlib import Path
import util.misc as utils
import numpy as np
import random
import time
import datetime
from .visual_entity_modules.models import build_model
from .visual_entity_modules.datasets1 import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
#from .visual_entity_modules.engine import evaluate, train_one_epoch
from .visual_entity_modules.engine import evaluate, train_one_epoch
import json
import torchvision.transforms as T
@VisualConstructionModel.register("RELTRExtract", "PyTorch")
class RELTRTorch(VisualConstructionModel):
    def __init__(self, name: str, dataset=None, args=None):
        # super().__init__(name=name, dataset=dataset, args=args)
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--lr_backbone', default=1e-5, type=float)
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--epochs', default=150, type=int)
        parser.add_argument('--lr_drop', default=100, type=int)
        parser.add_argument('--clip_max_norm', default=0.1, type=float,
                            help='gradient clipping max norm')

        # Model parameters
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")

        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_entities', default=100, type=int,
                            help="Number of query slots")
        parser.add_argument('--num_triplets', default=200, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        # Loss
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        # * Matcher
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                            help="giou box coefficient in the matching cost")

        # * Loss coefficients
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        parser.add_argument('--rel_loss_coef', default=1, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # dataset parameters
        parser.add_argument('--dataset', default='vg')
        parser.add_argument('--ann_path', default='./data/vg/', type=str)
        parser.add_argument('--img_folder', default='/home/cong/Dokumente/tmp/data/visualgenome/images/', type=str)

        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--num_workers', default=2, type=int)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

        parser.add_argument('--return_interm_layers', action='store_true',
                            help="Return the fpn if there is the tag")

        self.parser = parser

    def parse_args(self,args):
        return args

    # def data_reader(self, *args):

    #    return super().data_reader(*args)

    def evaluate(self, *args):
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # for output bounding box post-processing
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)

        def rescale_bboxes(out_bbox, size):
            img_w, img_h = size
            b = box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b

        # VG classes
        CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench',
                   'bike',
                   'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                   'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                   'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                   'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                   'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                   'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                   'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                   'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                   'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                   'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                   'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                   'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                   'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

        REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                       'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                       'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                       'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                       'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                       'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

        model, _, _ = build_model(args)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model'])
        model.eval()

        img_path = args.img_path
        im = Image.open(img_path)

        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.+ confidence
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                probas_obj.max(-1).values > 0.3))

        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = 10
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(
            -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[
                0])[:topk]
        keep_queries = keep_queries[indices]

        # use lists to store the outputs via up-values
        conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                lambda self, input, output: dec_attn_weights_sub.append(output[1])
            ),
            model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                lambda self, input, output: dec_attn_weights_obj.append(output[1])
            )
        ]
        with torch.no_grad():
            # propagate through the model
            outputs = model(img)

            for hook in hooks:
                hook.remove()

            # don't need the list anymore
            conv_features = conv_features[0]
            dec_attn_weights_sub = dec_attn_weights_sub[0]
            dec_attn_weights_obj = dec_attn_weights_obj[0]

            # get the feature map shape
            h, w = conv_features['0'].tensors.shape[-2:]
            im_w, im_h = im.size

            fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
            for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                    zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                ax = ax_i[0]
                ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
                ax.axis('off')
                ax.set_title(f'query id: {idx.item()}')
                ax = ax_i[1]
                ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
                ax.axis('off')
                ax = ax_i[2]
                ax.imshow(im)
                ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                           fill=False, color='blue', linewidth=2.5))
                ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                           fill=False, color='orange', linewidth=2.5))

                ax.axis('off')
                ax.set_title(
                    CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[
                        probas_obj[idx].argmax()], fontsize=10)

            fig.tight_layout()
            plt.show()

    def train(self, *args):
        parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[self.parser])
        args = parser.parse_args()
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        if args.frozen_weights is not None:
            assert args.masks, "Frozen training is meant for segmentation only"
        print(args)

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion, postprocessors = build_model(args)
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(args.output_dir)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            # del checkpoint['optimizer']
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            print('It is the {}th checkpoint'.format(checkpoint['epoch']))
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device,
                                                  args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                          args.clip_max_norm)
            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']  # anti-crash
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device,
                                                  args)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def run(self, mode="train"):
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.evaluate()
        elif mode == "single":
            raise ValueError("UnImplemented mode!")