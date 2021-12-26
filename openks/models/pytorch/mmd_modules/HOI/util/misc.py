"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class RecorderHOI:
    def __init__(self):
        self.epoch = None
        self.best_metrics = None

    def set_best_metrics(self, metrics, epoch):
        self.epoch = epoch
        self.best_metrics = metrics

    def is_best(self, metrics, epoch):
        if self.best_metrics is None:
            self.epoch = epoch
            self.best_metrics = metrics
            return True

        if metrics['mAP'] > self.best_metrics['mAP']:
            self.epoch = epoch
            self.best_metrics = metrics
            return True

        return False

    def get_best_metrics(self):
        return self.epoch, self.best_metrics

    def print_best_metrics(self):
        line = f'best metrics (epoch {self.epoch}):\n' \
               f'mAP: {self.best_metrics["mAP"]} ' \
               f'mAP rare: {self.best_metrics["mAP rare"]} ' \
               f'mAP non-rare: {self.best_metrics["mAP non-rare"]} ' \
               f'mean max recall: {self.best_metrics["mean max recall"]}'

        print('--------------------')
        print(line)
        print('--------------------')


def save_checkpoints(args, output_dir, recorder, epoch, test_stats,
                     model_without_ddp, optimizer, lr_scheduler,
                     hoi_evaluator=None):
    if args.output_dir:
        checkpoint_paths = [output_dir / 'checkpoint.pth']

        # save best metrics
        if recorder is not None and recorder.is_best(test_stats, epoch):
            checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            if hoi_evaluator is not None:
                hoi_evaluator.save_best_prediction(output_dir)
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

        for checkpoint_path in checkpoint_paths:
            save_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if recorder is not None:
                best_epoch, best_metrics = recorder.get_best_metrics()
                save_dict.update({
                    'best_metrics': (best_epoch, best_metrics)
                })
            save_on_master(save_dict, checkpoint_path)


def save_logs(args, train_stats, test_stats, epoch, n_parameters,
              output_dir, coco_evaluator=None):
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    if args.output_dir and is_main_process():
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


def load_pretrained_encoder(args, checkpoint):
    print('Loading encoder...')
    extra_weights = {}
    if args.enc_layers + args.hoi_enc_layers == 6:
        replace_list = [
            'self_attn{}.in_proj_weight', 'self_attn{}.in_proj_bias',
            'self_attn{}.out_proj.weight', 'self_attn{}.out_proj.bias',
            'linear1{}.weight', 'linear1{}.bias',
            'linear2{}.weight', 'linear2{}.bias',
            'norm1.weight', 'norm1.bias',
            'norm2.weight', 'norm2.bias',
        ]
        for i in range(args.enc_layers, 6):
            prefix1 = f'transformer.encoder.layers.{i}.'
            prefix2 = f'transformer.encoder.hoi_layers.{i - args.enc_layers}.'
            for k in replace_list:
                if k.startswith('self'):
                    extra_weights[f'{prefix2}{k.format("")}'] = checkpoint[f'{prefix1}{k.format("")}']
                elif k.startswith('linear'):
                    extra_weights[f'{prefix2}{k.format("")}'] = checkpoint[f'{prefix1}{k.format("")}']
                else:
                    extra_weights[f'{prefix2}{k}'] = checkpoint[f'{prefix1}{k}']

    return extra_weights


def load_pretrained_decoder(args, checkpoint):
    print('Loading decoder...')
    extra_weights = {}
    if args.dec_layers + args.hoi_dec_layers == 6:
        replace_dict = {
            'cross_attn{}.in_proj_weight': 'multihead_attn.in_proj_weight',
            'cross_attn{}.in_proj_bias': 'multihead_attn.in_proj_bias',
            'cross_attn{}.out_proj.weight': 'multihead_attn.out_proj.weight',
            'cross_attn{}.out_proj.bias': 'multihead_attn.out_proj.bias',
        }

        if args.vanilla_dec_type == 'vanilla_bottleneck' and args.load_bottleneck_dec_ca_weights:
            for i in range(0, args.dec_layers):
                prefix = f'transformer.decoder.layers.{i}.'
                for k, v in replace_dict.items():
                    for tgt in ['_sub', '_obj', '_verb']:
                        name = prefix + k.format(tgt)
                        extra_weights[name] = checkpoint[f'{prefix}{v}']

        if args.hoi_dec_type == 'vanilla_bottleneck':
            if args.load_bottleneck_dec_ca_weights:
                for i in range(args.dec_layers, 6):
                    prefix1 = f'transformer.decoder.layers.{i}.'
                    prefix2 = f'transformer.decoder.hoi_layers.{i - args.dec_layers}.'
                    for k, v in replace_dict.items():
                        for tgt in ['_sub', '_obj', '_verb']:
                            name = prefix2 + k.format(tgt)
                            extra_weights[name] = checkpoint[f'{prefix1}{v}']
        else:
            replace_list = [
                'self_attn.in_proj_weight', 'self_attn.in_proj_bias',
                'self_attn.out_proj.weight', 'self_attn.out_proj.bias',
                'multihead_attn.in_proj_weight', 'multihead_attn.in_proj_bias',
                'multihead_attn.out_proj.weight', 'multihead_attn.out_proj.bias',
                'norm1.weight', 'norm1.bias',
                'norm2.weight', 'norm2.bias',
                'norm3.weight', 'norm3.bias'
            ]
            if args.dim_feedforward_hoi == 2048:
                replace_list.extend([
                    'linear1.weight', 'linear1.bias',
                    'linear2.weight', 'linear2.bias',
                ])

            dec_layer = 0
            for i in range(args.dec_layers, 6):
                prefix1 = f'transformer.decoder.layers.{i}.'
                prefix2 = f'transformer.decoder.hoi_layers.'
                for _ in range(3):
                    for k in replace_list:
                        name = prefix2 + f'{dec_layer}.' + k
                        extra_weights[name] = checkpoint[f'{prefix1}{k}']
                    dec_layer += 1

    return extra_weights


def load_split_query(checkpoint):
    extra_weights = {'h_query_embed.weight': checkpoint['query_embed.weight'],
                     'o_query_embed.weight': checkpoint['query_embed.weight'],
                     'v_query_embed.weight': checkpoint['query_embed.weight']}
    return extra_weights


def load_pretrained_detr(args, model_without_ddp):
    print('Loading weights of DETR pre-trained on ms-coco...')
    checkpoint = torch.load(args.pretrained, 'cpu')['model']

    extra_weights = {}
    extra_weights.update(load_pretrained_encoder(args, checkpoint))
    extra_weights.update(load_pretrained_decoder(args, checkpoint))
    if args.split_query:
        extra_weights.update(load_split_query(checkpoint))
    print('All Loaded')

    checkpoint.update(extra_weights)

    print('\nexclude weights:')
    for n, p in model_without_ddp.named_parameters():
        if n not in checkpoint:
            print(n)

    model_without_ddp.load_state_dict(checkpoint, strict=False)
    del checkpoint, extra_weights


def load_model_weights(args, model_without_ddp, optimizer=None, lr_scheduler=None, recorder=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval:
            if 'optimizer' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
            if 'best_metrics' in checkpoint and recorder is not None:
                best_epoch, best_metrics = checkpoint['best_metrics']
                recorder.set_best_metrics(best_metrics, best_epoch)
    elif args.pretrained:
        load_pretrained_detr(args, model_without_ddp)
