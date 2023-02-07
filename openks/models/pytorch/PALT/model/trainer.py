import logging
import math
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer, AdamW, get_linear_schedule_with_warmup
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import EvalPrediction, PredictionOutput, PREFIX_CHECKPOINT_DIR, TrainOutput
from typing import Tuple, Dict, Union, Any
import torch.nn as nn
import os
from packaging import version


logger = logging.getLogger(__name__)


class GroupRandomSampler(Sampler):

    def __init__(self, data_source, group):
        self.data_source = data_source
        self.group = group
        assert len(data_source) % group == 0

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source) // self.group
        perm = torch.randperm(n)
        perm = (perm[:, None] * self.group + torch.arange(self.group)).view(-1)
        return iter(perm.tolist())

    def __len__(self):
        return self.num_samples


class GroupDistributredSampler(DistributedSampler):

    def __init__(self, dataset, group_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        self.group_size = group_size
        assert len(
            dataset) % group_size == 0, f'length of dataset should be a multiple of group size, but get length {len(dataset)} and group size {group_size}'
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_groups = math.ceil(len(self.dataset) / self.group_size / self.num_replicas)
        self.num_samples = self.num_groups * self.group_size
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset) // self.group_size
            perm = torch.randperm(n, generator=g)
            perm = (perm[:, None] * self.group_size + torch.arange(self.group_size)).view(-1)
            indices = perm.tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        sub_indices = []
        for group_id in range(self.num_groups):
            sub_indices += indices[(group_id * self.num_replicas + self.rank) * self.group_size
                                   :(group_id * self.num_replicas + self.rank + 1) * self.group_size]
        assert len(sub_indices) == self.num_samples
        return iter(sub_indices)


class KGCTrainer(Trainer):

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
        if self.optimizers is not None:
            assert False, "\033[31mThis should not happen.\033[0m"
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay) and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (any(nd in n for nd in no_decay) and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9,0.98), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def use_group_shuffle(self, num_neg):
        self.group_shuffle = True
        self.num_neg = num_neg

    def stop_group_shuffle(self):
        self.group_shuffle = False

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        if hasattr(self, 'group_shuffle') and self.group_shuffle:
            train_sampler = self._get_group_sampler()
        else:
            train_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError('Trainer: evaluation requires an eval_dataset.')
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if hasattr(self, 'group_shuffle') and self.group_shuffle:
            if self.args.local_rank != -1:
                sampler = GroupDistributredSampler(eval_dataset, group_size=self.num_neg * 3 + 1, shuffle=False)
            else:
                sampler = SequentialSampler(eval_dataset)
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
        return data_loader

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
        return data_loader
    
    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            assert False, "debug"

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        assert False, "debug"
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)


    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            assert False, "debug"
        else:
            loss.backward()

        return loss.item()

    def _get_group_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                GroupRandomSampler(self.train_dataset, self.num_neg * 3 + 1)
                if self.args.local_rank == -1
                else GroupDistributredSampler(self.train_dataset, group_size=self.num_neg * 3 + 1)
            )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        batch_size = dataloader.batch_size
        logger.info('***** Running %s *****', description)
        logger.info('  Num examples = %d', self.num_examples(dataloader))
        logger.info('  Batch size = %d', batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()
        if self.args.past_index >= 0:
            past = None
        for inputs in tqdm(dataloader, desc=description, disable=not self.is_local_master()):
            has_labels = any((inputs.get(k) is not None for k in ['labels', 'lm_labels', 'masked_lm_labels']))
            for (k, v) in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs['mems'] = past
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    (step_eval_loss, logits) = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get('labels') is not None:
                    if label_ids is None:
                        label_ids = inputs['labels'].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs['labels'].detach()), dim=0)
        if self.args.local_rank != -1:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        if self.compute_metrics is not None and preds is not None and (label_ids is not None):
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics['eval_loss'] = np.mean(eval_losses)
        for key in list(metrics.keys()):
            if not key.startswith('eval_'):
                metrics[f'eval_{key}'] = metrics.pop(key)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
