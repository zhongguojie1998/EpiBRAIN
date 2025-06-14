import copy
import glob
import json
import logging
import os

import numpy as np
import torch
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD
import torch.optim as optim
import torchmetrics as tm
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from data.dataset import DumySampler, GenomeIntervalDataset, StrictDistributedSampler
from model.pytorch_borzoi_model import Borzoi
from model.pytorch_borzoi_utils import safe_state_dict_loader
from utils.logging import LOGGER_PREFIX, TrainingLogger, timer
from utils.loss import LOSS_DICT
from utils.multi_gpu import blocking_sync_wait, cleanup, deepspeed_setup, global_aggregate, setup, torchrun_setup


def aggregate_test_res(trainer, prefix="Test"):
    # aggregate the results and clear per rank file
    pattern = f"{trainer.logging_config.res_dir}/{prefix}_preds_rank_*_epoch_{trainer.current_epoch}.pt"

    file_list = glob.glob(pattern)
    if not file_list:
        trainer.logger.warning(f"{prefix} res aggregation failed (cannot find any pred file), skip")
    else:
        all_labels = []
        all_preds = []
        for file_path in file_list:
            data = torch.load(file_path, map_location="cpu")
            all_labels.append(data["label"])
            all_preds.append(data["pred"])

        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        torch.save(
            {"label": all_labels, "pred": all_preds},
            f"{trainer.logging_config.res_dir}/{prefix}_preds_epoch_{trainer.current_epoch}.pt",
        )

        for file_path in file_list:
            os.remove(file_path)


def get_metric_collection(prefix: str = "Valid/", num_outputs=1):
    collection = tm.MetricCollection(
        {
            "MSE": tm.MeanSquaredError(num_outputs=num_outputs),
            "MAE": tm.MeanAbsoluteError(num_outputs=num_outputs),
            "PearsonR": tm.PearsonCorrCoef(num_outputs=num_outputs),
        },
        prefix=prefix,
    )
    return collection


def construct_logging_metric_dict(trainer):
    metric_dict = {}
    for k, v in trainer.metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
            # check if it's the metrics log
            if "MSE" in k or "MAE" in k or "PearsonR" in k:
                for dim, value in enumerate(v):
                    trial_name = trainer.label_meta.loc[dim, "trial"]
                    metric_dict[f"{k}/{trial_name}"] = value
            else:
                metric_dict[k] = v.nanmean().item()
        else:
            metric_dict[k] = v
    return metric_dict


def count_grad_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_param_grad(model, trainable_params=[]):

    for name, param in model.named_parameters():
        param.requires_grad = False
        if name in trainable_params:
            param.requires_grad = True

    return model


class DNASeqModelTrainer:

    def __init__(self, config, rank, world_size, logger, local_rank=None):

        # set up the configuration
        self.config = config
        self.dataset_config = self.config.data.dataset
        self.training_config = self.config.training
        self.model_config = self.config.model
        self.logging_config = self.config.logging
        # get the logger
        self.logger = logger
        # get the hardware setting
        self.rank = rank
        ## with local rank, we can distribute on different machines
        if local_rank is not None:
            self.local_rank = local_rank
        else:
            self.local_rank = rank
        self.world_size = world_size
        self.should_log = (self.world_size > 1 and self.rank == 0) or self.world_size == 1

        # get the training settings
        self.current_epoch = 0
        self.current_step = 0  # based on update step
        self.current_lr = self.training_config.lr
        self.best_valid_loss = torch.inf
        self.metrics = {}

        self.trial_num = self.model_config.output_heads[self.model_config.use_head]

        # set up data
        self.data_split = ["train", "valid", "test"]
        self.data_func = {k: {"dataset": None, "data_sampler": None, "data_loader": None} for k in self.data_split}
        ## get the dataset
        self.get_dataset()
        ## get the dataloader
        self.get_dataloader()

        # set up model
        ## get the checkpoint if necessary
        if self.training_config.load_checkpoint is not None:
            with timer(f"Loading checkpoint", self.logger, self.rank, self.world_size):
                self.load_checkpoint()
        ## get the model
        self.get_model()
        ## get the optimizer
        self.get_optimizer()
        ## get the loss function
        self.get_loss()

        # some helper information
        if self.training_config.load_checkpoint is not None:
            if self.training_config.test_only:
                self.logger.info("Only Testing")
            else:
                self.logger.info("Continue Training")
                if self.current_epoch >= self.training_config.total_epoch:
                    self.logger.warning("The loaded checkpoint has exceeded the total epoch number. Dry run.")
        else:
            if not self.training_config.finetune:
                self.logger.info("Start Training")
            else:
                self.logger.info("Start Fine-tuning")

    def get_dataset(self):

        self.logger.info("Loading datasets...")

        for split in self.data_split:
            try:
                config = copy.deepcopy(self.dataset_config)
                if split != "train":
                    # for valid and test, we disable the data augumentation
                    config.update({"shift_augs": None, "rc_aug": False, "return_augs": False})
                self.data_func[split]["dataset"] = GenomeIntervalDataset(split, **config)

                assert self.trial_num == self.data_func[split]["dataset"].label_meta.shape[0]
                self.label_meta = self.data_func[split]["dataset"].label_meta

            except Exception as e:
                if split == "train" and not self.training_config.test_only:
                    self.logger.error("Failed to load training dataset. Please check the preprocess setting.")
                    self.logger.exception(e)
                    exit(1)
                elif split == "test" and self.training_config.test_only:
                    self.logger.error(
                        "Failed to load testing dataset in `test_only` mode. Please check the preprocess setting."
                    )
                    self.logger.exception(e)
                    exit(1)
                else:
                    self.logger.warning(f"No {split} dataset found.")

        self.logger.info(
            f"{'/'.join([k for k,v in self.data_func.items() if v["dataset"] is not None])} datasets loaded successfully."
        )

    def get_dataloader(self):

        if self.world_size > 1:
            sampler_rank = self.rank
        else:
            # if single card, rank=0 ensures the data is sampled from index 0
            sampler_rank = 0

        for split in self.data_split:

            dataset = self.data_func[split]["dataset"]

            # get data sampler for each card
            if self.training_config.test_only:
                # in testing mode, we force strict, though it may cause dead lock
                sampler_cls = StrictDistributedSampler if dataset is not None else DumySampler
            else:
                # in training mode, we allow resampling
                sampler_cls = DistributedSampler if dataset is not None else DumySampler

            sampler = sampler_cls(
                dataset=dataset,
                num_replicas=self.world_size,
                rank=sampler_rank,
                shuffle=False if split != "train" else True,
            )
            self.data_func[split]["data_sampler"] = sampler

            # get the dataloader
            self.data_func[split]["data_loader"] = (
                DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    shuffle=False,  # set shuffle in sampler, dataloader should be set as False
                    **self.training_config.dataloader_params,
                )
                if dataset is not None
                else None
            )

    def get_model(self):

        self.logger.info("Loading model...")

        # get our baseline model
        if "borzoi" in self.model_config.model_name:
            model_cls = Borzoi
        else:
            self.logger.error(f"Model {self.model_config.model_name} is not implemented yet.")
            exit(1)

        self.model = model_cls.from_hparams(**self.model_config)

        # if finetune, load the pretrained model
        if self.training_config.finetune:
            # load the pretrained state dict
            pretrained_model = model_cls.from_pretrained(self.model_config.model_name)
            org_model_state_dict = self.model.state_dict()
            updated_model_state_dict = safe_state_dict_loader(
                org_model_state_dict=org_model_state_dict,
                load_model_state_dict=pretrained_model.state_dict(),
                logger=self.logger,
            )
            org_model_state_dict.update(updated_model_state_dict)
            self.model.load_state_dict(org_model_state_dict)

            # initialize the finetune model
            if self.model_config.finetune_method == "lora":
                self.logger.info("LORA Finetune")
                finetune_config = LoraConfig(**self.model_config.finetune_param)
                self.model = get_peft_model(self.model, finetune_config)
            elif self.model_config.finetune_method == "finetune_layers":
                self.logger.info("Finetune the given layers")
                self.model = set_param_grad(self.model, **self.model_config.finetune_param)
            else:
                self.logger.error(f"Finetune method {self.model_config.finetune_method} is not implemented yet.")
                exit(1)

        # get all the trainable parameters
        self.trainable_params = [n for n, m in self.model.named_parameters() if m.requires_grad]
        for n in self.trainable_params:
            self.logger.debug(f"Trainable module name: {n}")

        # if necessary, load the checkpoint
        if self.training_config.load_checkpoint is not None:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        else:
            self.initialize(self.trainable_params)

        # set the gradient again (in case we load from the checkpoint)
        self.model = set_param_grad(self.model, self.trainable_params)
        trainable_para_num = count_grad_parameters(self.model)
        total_para_num = count_all_parameters(self.model)
        self.logger.info(
            f"Trainable params: {trainable_para_num} || All params: {total_para_num} || Trainable%: {trainable_para_num / total_para_num:.4f}"
        )

        # send the model to training device
        self.model = self.model.to(self.local_rank, non_blocking=True)
        if self.world_size > 1:
            # Add DDP wrapper
            self.model = DDP(
                self.model, device_ids=[self.local_rank], static_graph=True, find_unused_parameters=True
            )

            # If applicable, add gradient compression hook
            if self.training_config.use_grad_compression:
                state = PowerSGD.PowerSGDState(
                    process_group=None,  # we do compression on all ranks
                    **self.training_config.get("powerSGD_params", {}),
                )
                self.model.register_comm_hook(state, PowerSGD.powerSGD_hook)

        self.logger.info(f"Model {self.model_config.model_name} loaded successfully.")

    def get_optimizer(self):
        optim_class = eval(f"optim.{self.training_config.optimizer}")
        self.optimizer = optim_class(self.model.parameters(), **self.training_config.optimizer_params)

        scheduler_class = eval(f"optim.lr_scheduler.{self.training_config.scheduler}")
        self.scheduler = scheduler_class(self.optimizer, **self.training_config.scheduler_params)

        if self.training_config.scheduler in ["ReduceLROnPlateau"]:
            self.scheduler_update_freq = "epoch"

            # we should further check if valid set is available
            if not self.training_config.test_only and self.data_func["valid"]["dataset"] is None:
                self.logger.error("The given scheduler requires the valid dataset")
                exit(1)
        else:
            self.scheduler_update_freq = "batch"

        # if necessary, load the checkpoint
        if self.training_config.load_checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])

    def get_loss(self):
        self.criterion = LOSS_DICT.get(self.training_config.loss)

        if self.criterion is None:
            self.logger.error(f"Loss {self.training_config.loss} is not implemented yet.")
            exit(1)

    def initialize(self, trainable_params):
        # TODO: initialize the model parameter if we are going to use special initialization method
        # only re-initialize trainable parameters
        pass

    def load_checkpoint(self):

        self.logger.info(f"Loading checkpoint from {self.training_config.load_checkpoint}")
        self.checkpoint = torch.load(
            self.training_config.load_checkpoint, map_location=torch.device(self.local_rank)
        )

        # since this is first called in the model initialization pipeline, the model and optimizers loads the checkpoint in their own functions instead of here
        self.current_epoch = self.checkpoint["epoch"]
        self.current_step = self.checkpoint["step"]
        self.current_lr = self.checkpoint["lr"]
        self.best_valid_loss = self.checkpoint["best_valid_loss"]

        self.logger.info("Checkpoint loaded successfully.")

    def save_checkpoint(self, save_name=None):

        if save_name is None:
            save_name = self.current_epoch

        # if DDP, then the model is wrapped in module
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        self.logger.info(f"Saving checkpoint for epoch {self.current_epoch}...")
        if self.should_log:
            checkpoint = {
                "epoch": self.current_epoch,
                "step": self.current_step,
                "lr": self.current_lr,
                "model_state_dict": {k: v.cpu() for k, v in model_to_save.state_dict().items()},
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_valid_loss": self.best_valid_loss,
            }
            torch.save(checkpoint, f"{self.logging_config.checkpoint_dir}/chk_epoch_{save_name}.pt")
        self.logger.info("Checkpoint saved successfully.")
        blocking_sync_wait(self.world_size)

    def lr_warmup(self):
        # lr warmup, gradually change from 0 to the given lr
        if self.current_step < self.training_config.lr_warmup_step:
            lr_scale = min(
                1.0,
                float(self.current_step + 1) / float(self.training_config.lr_warmup_step),
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * float(self.training_config.lr)

    def train_step(self):
        self.model.train()

        # for loss log
        batch_loss = 0
        batch_count = 0
        # for other metric log
        # tm_metrics = get_metric_collection(prefix="Train/", num_outputs=self.trial_num).to(self.local_rank)
        # tm_metrics.reset()

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label, ind) in enumerate(dataloader):

            # seq_embedding shape [batch, L, 4]
            # label shape [batch, num_central_bin (896), num_trail (93)]
            seq_embedding, label = seq_embedding.to(self.local_rank, non_blocking=True), label.to(
                self.local_rank, non_blocking=True
            )
            # pred_embedding shape [batch, num_trail (93), num_central_bin (896)]
            pred = self.model(
                seq_embedding.permute(0, 2, 1),
                self.model_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            ).permute(0, 2, 1)

            # loss
            loss = self.criterion(pred, label) / self.training_config.accum_step
            batch_loss += loss.detach().cpu().item() * self.training_config.accum_step
            batch_count += 1

            # metrics
            # tm_metrics.update(pred.reshape(-1, self.trial_num), label.reshape(-1, self.trial_num))

            # whether to do the loss aggregation
            should_update = ((i + 1) % self.training_config.accum_step == 0) or (i + 1 == len(dataloader))
            if self.world_size > 1 and not should_update:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if should_update:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
                self.lr_warmup()
                self.optimizer.step()
                self.current_lr = self.optimizer.param_groups[0]["lr"]
                if self.scheduler_update_freq == "batch":
                    self.scheduler.step()
                self.current_step += 1

            # log training status
            ## output the sample index
            self.logger.debug(
                f"batch id for step {self.current_step} is {','.join([str(int(i)) for i in ind])}",
                diagnose=self.logging_config.diagnose,
            )
            ## in the training loop, we only look at the local loss
            if self.current_step % self.logging_config.report_every == 0:
                # report_loss = batch_loss / (i + 1)
                report_loss = batch_loss / batch_count

                if self.logging_config.use_tensorboard:
                    self.logger.metric(
                        f"Train/rank[{self.rank}]_loss",
                        report_loss,
                        step=self.current_step,
                        log_also=False,
                        diagnose=self.logging_config.diagnose,
                    )
                    self.logger.metric("Train/lr", self.current_lr, step=self.current_step, log_also=False)

                    # we can only log these info in tensorboard
                    if self.logging_config.log_more:
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace(".", "/")

                            # add weight histogram
                            weight = value.data.detach().cpu().numpy()
                            if not np.isnan(weight).any():
                                self.logger.metric(
                                    "weights/" + tag,
                                    weight,
                                    self.current_step,
                                    log_also=False,
                                    write_hist=True,
                                )
                            else:
                                self.logger.warning(
                                    f"failed to add weight histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                                )

                            # only add gradients if they are not None
                            if value.grad is not None:
                                grad = value.grad.data.detach().cpu().numpy()
                                if not np.isnan(grad).any():
                                    self.logger.metric(
                                        "grads/" + tag,
                                        grad,
                                        self.current_step,
                                        log_also=False,
                                        write_hist=True,
                                    )
                                else:
                                    self.logger.warning(
                                        f"failed to add grad histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                                    )
                else:
                    self.logger.info(
                        f"[Train] [Epoch {self.current_epoch}] Step {self.current_step} | Loss: {report_loss:.6f} | lr: {self.current_lr}"
                    )

                batch_loss = 0
                batch_count = 0

            # here we reset the gradient in the final stage as we may need to log the gradient value
            if should_update:
                self.optimizer.zero_grad()
            
            if self.logging_config.diagnose:
                if self.current_step % self.logging_config.get("diagnose_save_step", 1) == 0:
                    self.save_checkpoint(f"diag_step_{self.current_step}")

        self.current_epoch += 1

        # get and write the metric logs
        # self.metrics.update(tm_metrics.compute())

    def infer_step(self, log_loss=False, save_pred=False, log_prefix="Valid"):
        self.model.eval()

        if log_loss:
            running_loss_local = torch.tensor(0.0, device=self.local_rank)
        if save_pred:
            preds = []
            labels = []

        tm_metrics = get_metric_collection(prefix=f"{log_prefix}/", num_outputs=self.trial_num).to(self.local_rank)
        tm_metrics.reset()

        dataloader = self.data_func[log_prefix.lower()]["data_loader"]

        with torch.no_grad():
            for i, (seq_embedding, label, ind) in enumerate(dataloader):
                # seq_embedding shape [batch, L, 4]
                # label shape [batch, num_central_bin (896), num_trail (93)]
                seq_embedding, label = seq_embedding.to(self.local_rank, non_blocking=True), label.to(
                    self.local_rank, non_blocking=True
                )
                # pred_embedding shape [batch, num_trail (93), num_central_bin (896)]
                pred = self.model(
                    seq_embedding.permute(0, 2, 1),
                    self.model_config.use_head,
                    data_parallel_training=True if self.world_size > 1 else False,
                ).permute(0, 2, 1)

                # loss
                if log_loss:
                    loss = self.criterion(pred, label)
                    running_loss_local += loss.detach()
                # pred
                if save_pred:
                    preds.append(pred.detach().cpu())
                    labels.append(label.detach().cpu())
                # metrics
                tm_metrics.update(pred.reshape(-1, self.trial_num), label.reshape(-1, self.trial_num))

        # get and write the metric logs
        self.metrics.update(tm_metrics.compute())
        # get global loss
        if log_loss:
            total_loss = running_loss_local.clone()
            # aggregate the loss value from all devices
            global_aggregate(total_loss, aggregate="sum", world_size=self.world_size)

            global_avg_loss = total_loss / (len(dataloader) * self.world_size)
            self.metrics.update({f"{log_prefix}/loss": global_avg_loss.cpu().item()})

        if save_pred:
            # we have to pre save all the res and later aggregate them
            torch.save(
                {"label": torch.cat(labels, dim=0), "pred": torch.cat(preds, dim=0)},
                f"{self.logging_config.res_dir}/{log_prefix}_preds_rank_{self.rank}_epoch_{self.current_epoch}.pt",
            )


class DeepspeedTrainer(DNASeqModelTrainer):

    def __init__(self, config, rank, world_size, logger, local_rank=None):

        # set up the configuration
        self.config = config
        self.dataset_config = self.config.data.dataset
        self.training_config = self.config.training
        self.model_config = self.config.model
        self.logging_config = self.config.logging
        # get the logger
        self.logger = logger
        # get the hardware setting
        self.rank = rank
        ## with local rank, we can distribute on different machines
        if local_rank is not None:
            self.local_rank = local_rank
        else:
            self.local_rank = rank
        self.world_size = world_size
        self.should_log = (self.world_size > 1 and self.rank == 0) or self.world_size == 1

        # get the training settings
        self.current_epoch = 0
        self.current_step = 0  # based on update step
        self.current_lr = self.training_config.lr
        self.best_valid_loss = torch.inf
        self.metrics = {}

        self.trial_num = self.model_config.output_heads[self.model_config.use_head]

        # set up data
        self.data_split = ["train", "valid", "test"]
        self.data_func = {k: {"dataset": None, "data_sampler": None, "data_loader": None} for k in self.data_split}
        ## get the dataset
        self.get_dataset()
        ## get the dataloader
        self.get_dataloader()

        # set up model
        ## get the model, optimizer and scheduler from deepspeed
        self.get_deepspeed()
        ## if necessary, laod the checkpoint
        if self.training_config.load_checkpoint is not None:
            self.load_checkpoint()

        ## get the loss function
        self.get_loss()

        # some helper information
        if self.training_config.load_checkpoint is not None:
            if self.training_config.test_only:
                self.logger.info("Only Testing")
            else:
                self.logger.info("Continue Training")
                if self.current_epoch >= self.training_config.total_epoch:
                    self.logger.warning("The loaded checkpoint has exceeded the total epoch number. Dry run.")
        else:
            if not self.training_config.finetune:
                self.logger.info("Start Training")
            else:
                self.logger.info("Start Fine-tuning")

    def get_deepspeed(self):
        import deepspeed


        self.logger.info("Loading model...")

        # get our baseline model
        if "borzoi" in self.model_config.model_name:
            model_cls = Borzoi
        else:
            self.logger.error(f"Model {self.model_config.model_name} is not implemented yet.")
            exit(1)

        self.model = model_cls.from_hparams(**self.model_config)

        # if finetune, load the pretrained model
        if self.training_config.finetune:
            # load the pretrained state dict
            pretrained_model = model_cls.from_pretrained(self.model_config.model_name)
            org_model_state_dict = self.model.state_dict()
            updated_model_state_dict = safe_state_dict_loader(
                org_model_state_dict=org_model_state_dict,
                load_model_state_dict=pretrained_model.state_dict(),
                logger=self.logger,
            )
            org_model_state_dict.update(updated_model_state_dict)
            self.model.load_state_dict(org_model_state_dict)

            # initialize the finetune model
            if self.model_config.finetune_method == "lora":
                self.logger.info("LORA Finetune")
                finetune_config = LoraConfig(**self.model_config.finetune_param)
                self.model = get_peft_model(self.model, finetune_config)
            elif self.model_config.finetune_method == "finetune_layers":
                self.logger.info("Finetune the given layers")
                self.model = set_param_grad(self.model, **self.model_config.finetune_param)
            else:
                self.logger.error(f"Finetune method {self.model_config.finetune_method} is not implemented yet.")
                exit(1)

        # get all the trainable parameters
        # initialize the trainable model parameters if we didn't load from checkpoint
        self.trainable_params = [n for n, m in self.model.named_parameters() if m.requires_grad]

        if self.training_config.load_checkpoint is None:
            self.initialize(self.trainable_params)

        for n in self.trainable_params:
            self.logger.debug(f"Trainable module name: {n}")
        trainable_para_num = count_grad_parameters(self.model)
        total_para_num = count_all_parameters(self.model)
        self.logger.info(
            f"Trainable params: {trainable_para_num} || All params: {total_para_num} || Trainable%: {trainable_para_num / total_para_num:.4f}"
        )

        # set up deepspeed
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=[m for _, m in self.model.named_parameters() if m.requires_grad],
            config=self.training_config.deepspeed_config,
            dist_init_required=self.world_size > 1,  # we are using DDP
        )

        # set up scheduler step logic
        if self.training_config.scheduler in ["ReduceLROnPlateau"]:
            self.scheduler_update_freq = "epoch"

            # we should further check if valid set is available
            if not self.training_config.test_only and self.data_func["valid"]["dataset"] is None:
                self.logger.error("The given scheduler requires the valid dataset")
                exit(1)
        else:
            self.scheduler_update_freq = "batch"

        self.logger.info(f"Model {self.model_config.model_name} loaded successfully.")

    def load_checkpoint(self):

        self.logger.info(f"Loading checkpoint from {self.training_config.load_checkpoint}")

        # if necessary, load the checkpoint
        _, client_state = self.model_engine.load_checkpoint(
            self.training_config.checkpoint_dir, self.training_config.load_checkpoint
        )

        self.current_epoch = client_state["epoch"]
        self.current_step = client_state["step"]
        self.current_lr = client_state["lr"]
        self.best_valid_loss = client_state["best_valid_loss"]

        self.logger.info("Checkpoint loaded successfully.")

    def save_checkpoint(self, save_name=None):

        if save_name is None:
            save_name = self.current_epoch

        self.logger.info(f"Saving checkpoint for epoch {self.current_epoch}...")

        # in deepspeed, we can spread the model on different ranks, so just let every rank save their results
        self.model_engine.save_checkpoint(
            self.logging_config.checkpoint_dir,
            save_name,
            client_state={
                "epoch": self.current_epoch,
                "step": self.current_step,
                "lr": self.current_lr,
                "best_valid_loss": self.best_valid_loss,
            },
        )

        self.logger.info("Checkpoint saved successfully.")

    def train_step(self):
        self.model_engine.train()

        # for loss log
        batch_loss = 0
        batch_count = 0

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label, ind) in enumerate(dataloader):

            # seq_embedding shape [batch, L, 4]
            # label shape [batch, num_central_bin (896), num_trail (93)]
            seq_embedding, label = seq_embedding.to(self.local_rank, non_blocking=True), label.to(
                self.local_rank, non_blocking=True
            )
            # pred org shape [batch, num_trail (93), num_central_bin (896)]
            pred = self.model_engine(
                seq_embedding.permute(0, 2, 1),
                self.model_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            ).permute(0, 2, 1)

            # loss
            loss = self.criterion(pred, label) / self.training_config.accum_step
            batch_loss += loss.detach().cpu().item() * self.training_config.accum_step
            batch_count += 1

            self.model_engine.backward(loss)

            # log training status
            ## output the sample index
            self.logger.debug(
                f"batch id for step {self.current_step} is {','.join([str(int(i)) for i in ind])}",
                diagnose=self.logging_config.diagnose,
            )
            ## in the training loop, we only look at the local loss
            self.current_step = self.model_engine.global_steps
            self.current_lr = self.optimizer.param_groups[0]["lr"]
            if self.current_step % self.logging_config.report_every == 0:
                # report_loss = batch_loss / (i + 1)
                report_loss = batch_loss / batch_count

                if self.logging_config.use_tensorboard:
                    self.logger.metric(
                        f"Train/rank[{self.rank}]_loss",
                        report_loss,
                        step=self.current_step,
                        log_also=False,
                        diagnose=self.logging_config.diagnose,
                    )
                    self.logger.metric("Train/lr", self.current_lr, step=self.current_step, log_also=False)

                    # we can only log these info in tensorboard
                    if self.logging_config.log_more:
                        for tag, value in self.model_engine.module.named_parameters():
                            tag = tag.replace(".", "/")

                            # add weight histogram
                            weight = value.data.detach().cpu().numpy()
                            if not np.isnan(weight).any():
                                self.logger.metric(
                                    "weights/" + tag,
                                    weight,
                                    self.current_step,
                                    log_also=False,
                                    write_hist=True,
                                )
                            else:
                                self.logger.warning(
                                    f"failed to add weight histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                                )

                            # only add gradients if they are not None
                            if value.grad is not None:
                                grad = value.grad.data.detach().cpu().numpy()
                                if not np.isnan(grad).any():
                                    self.logger.metric(
                                        "grads/" + tag,
                                        grad,
                                        self.current_step,
                                        log_also=False,
                                        write_hist=True,
                                    )
                                else:
                                    self.logger.warning(
                                        f"failed to add grad histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                                    )
                else:
                    self.logger.info(
                        f"[Train] [Epoch {self.current_epoch}] Step {self.current_step} | Loss: {report_loss:.6f} | lr: {self.current_lr}"
                    )

                batch_loss = 0
                batch_count = 0

            self.model_engine.step()

            if self.logging_config.diagnose:
                if self.current_step % self.logging_config.get("diagnose_save_step", 1) == 0:
                    self.save_checkpoint(f"diag_step_{self.current_step}")

        self.current_epoch += 1

    def infer_step(self, log_loss=False, save_pred=False, log_prefix="Valid"):

        ## only change here is change from model to model_engine
        self.model_engine.eval()

        if log_loss:
            running_loss_local = torch.tensor(0.0, device=self.local_rank)
        if save_pred:
            preds = []
            labels = []

        tm_metrics = get_metric_collection(prefix=f"{log_prefix}/", num_outputs=self.trial_num).to(self.local_rank)
        tm_metrics.reset()

        dataloader = self.data_func[log_prefix.lower()]["data_loader"]

        with torch.no_grad():
            for i, (seq_embedding, label, ind) in enumerate(dataloader):
                # seq_embedding shape [batch, L, 4]
                # label shape [batch, num_central_bin (896), num_trail (93)]
                seq_embedding, label = seq_embedding.to(self.local_rank, non_blocking=True), label.to(
                    self.local_rank, non_blocking=True
                )
                # pred_embedding shape [batch, num_trail (93), num_central_bin (896)]
                pred = self.model_engine(
                    seq_embedding.permute(0, 2, 1),
                    self.model_config.use_head,
                    data_parallel_training=True if self.world_size > 1 else False,
                ).permute(0, 2, 1)

                # loss
                if log_loss:
                    loss = self.criterion(pred, label)
                    running_loss_local += loss.detach()
                # pred
                if save_pred:
                    preds.append(pred.detach().cpu())
                    labels.append(label.detach().cpu())
                # metrics
                tm_metrics.update(pred.reshape(-1, self.trial_num), label.reshape(-1, self.trial_num))

        # get and write the metric logs
        self.metrics.update(tm_metrics.compute())
        # get global loss
        if log_loss:
            total_loss = running_loss_local.clone()
            # aggregate the loss value from all devices
            global_aggregate(total_loss, aggregate="sum", world_size=self.world_size)

            global_avg_loss = total_loss / (len(dataloader) * self.world_size)
            self.metrics.update({f"{log_prefix}/loss": global_avg_loss.cpu().item()})

        if save_pred:
            # we have to pre save all the res and later aggregate them
            torch.save(
                {"label": torch.cat(labels, dim=0), "pred": torch.cat(preds, dim=0)},
                f"{self.logging_config.res_dir}/{log_prefix}_preds_rank_{self.rank}_epoch_{self.current_epoch}.pt",
            )


# this function also support single GPU training
def mp_main(rank, world_size, myconfig, local_rank=None):
    # special train loggers which can also log to tensorboard
    logger = TrainingLogger(
        name=f"{LOGGER_PREFIX}-Trainer",
        level=myconfig.logging.log_level,
        log_dir=myconfig.logging.log_dir,
        redirect=myconfig.logging.write_log_to_file,
        overwrite=False,
        rank=rank,
        world_size=world_size,
        use_tensorboard=myconfig.logging.use_tensorboard,
    )

    try:
        # if we have multiple GPUs, we need to set up DDP
        trainer_cls = DNASeqModelTrainer
        if myconfig.training.torchrun:
            torchrun_setup()
        elif myconfig.training.deepspeed:
            deepspeed_setup()
            trainer_cls = DeepspeedTrainer
        else:
            setup(rank, world_size, myconfig.training.MASTER_ADDR, myconfig.training.MASTER_PORT)

        # Set up the trainer
        trainer = trainer_cls(
            config=myconfig,
            rank=rank,
            world_size=world_size,
            logger=logger,
            local_rank=local_rank,
        )
        blocking_sync_wait(world_size)

        if not myconfig.training.test_only:
            for epoch in range(trainer.current_epoch, myconfig.training.total_epoch):
                trainer.logger.info(f"Current Epoch: {trainer.current_epoch + 1}")

                for split in trainer.data_split:
                    trainer.data_func[split]["data_sampler"].set_epoch(epoch)

                if trainer.data_func["train"]["data_loader"] is not None:
                    with timer(f"[Train] [Epoch {trainer.current_epoch + 1}]", logger, rank, world_size):
                        trainer.train_step()

                    blocking_sync_wait(world_size)

                if trainer.data_func["valid"]["data_loader"] is not None:
                    with timer(f"[Valid] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                        trainer.infer_step(log_loss=True, log_prefix="Valid", save_pred=False)

                    blocking_sync_wait(world_size)

                    valid_loss = trainer.metrics["Valid/loss"]
                    if valid_loss < trainer.best_valid_loss:
                        trainer.best_valid_loss = valid_loss
                        logger.info(f"New best validation loss: {valid_loss:.6f}")
                        with timer(f"Saving new best model", logger, rank, world_size):
                            trainer.save_checkpoint(save_name="best_valid_loss")

                    if trainer.scheduler_update_freq == "epoch":
                        trainer.scheduler.step(valid_loss)

                # write the model
                if trainer.current_epoch % myconfig.logging.save_every == 0:
                    with timer(f"Regular saving checkpoint", logger, rank, world_size):
                        trainer.save_checkpoint()
                    # save_test_res = True

                # # testing the model
                # if trainer.data_func["test"]["data_loader"] is not None:
                #     with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                #         trainer.infer_step(log_loss=True, log_prefix="Test", save_pred=save_test_res)

                #     blocking_sync_wait(world_size)

                # write the metrics
                ## if we use tensorboard, the metrics will be written into tb (initialized when creating the training logger)
                if trainer.should_log:
                    metric_dict = construct_logging_metric_dict(trainer)
                    if myconfig.logging.use_tensorboard:
                        for k, v in metric_dict.items():
                            trainer.logger.metric(k, v, step=trainer.current_step, log_also=False)
                    else:
                        for k, v in metric_dict.items():
                            trainer.logger.metric(k, v, step=trainer.current_step)
                        with open(
                            f"{myconfig.logging.log_dir}/metrics/epoch_{trainer.current_epoch}.json", "w"
                        ) as f:
                            json.dump(metric_dict, f, indent=4)
                blocking_sync_wait(world_size)

            # for later continue training if needed
            logger.info("Training end.")
            with timer(f"Save final checkpoint", logger, rank, world_size):
                trainer.save_checkpoint(save_name="last_epoch")

        else:
            if trainer.data_func["train"]["data_loader"] is not None:
                with timer(f"[Train/Infer] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                    trainer.infer_step(log_loss=False, log_prefix="Train", save_pred=True)
                blocking_sync_wait(world_size)

            if trainer.data_func["valid"]["data_loader"] is not None:
                with timer(f"[Valid/Infer] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                    trainer.infer_step(log_loss=False, log_prefix="Valid", save_pred=True)
                blocking_sync_wait(world_size)

            with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                trainer.infer_step(log_loss=False, log_prefix="Test", save_pred=True)
            blocking_sync_wait(world_size)

            # save the raw preds
            if trainer.should_log:
                aggregate_test_res(trainer, "Train")
                aggregate_test_res(trainer, "Valid")
                aggregate_test_res(trainer, "Test")

        blocking_sync_wait(world_size)

        if not myconfig.training.deepspeed:
            cleanup(world_size)

    except Exception as e:
        logger.exception(e)
    finally:
        # in case the program doesn't stop as expected
        cleanup(world_size)
