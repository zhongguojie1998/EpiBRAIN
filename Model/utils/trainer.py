import copy
import glob
import json
import os

import torch
import torch.optim as optim
import torchmetrics as tm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from data.dataset import DumySampler, GenomeIntervalDataset, StrictDistributedSampler
from model.pytorch_borzoi_model import Borzoi
from utils.logging import LOGGER_PREFIX, TrainingLogger, timer
from utils.loss import LOSS_DICT
from utils.multi_gpu import blocking_sync_wait, cleanup, global_aggregate, setup, torchrun_setup


def aggregate_test_res(trainer):
    # aggregate the results and clear per rank file
    pattern = f"{trainer.logging_config.res_dir}/preds_rank_*_epoch_{trainer.current_epoch}.pt"

    file_list = glob.glob(pattern)
    if not file_list:
        trainer.logger.warning("Test res aggregation failed (cannot find any pred file), skip")
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
            f"{trainer.logging_config.res_dir}/preds_epoch_{trainer.current_epoch}.pt",
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
        # get the model
        self.get_model()
        # get the optimizer
        self.get_optimizer()
        # get the loss function
        self.get_loss()
        # load the parameter if necessary
        if self.training_config.load_checkpoint is not None:
            with timer(f"Loading checkpoint", self.logger, self.rank, self.world_size):
                self.load_checkpoint()
            if self.training_config.test_only:
                self.logger.info("Only Testing")
            else:
                self.logger.info("Continue Training")
                if self.current_epoch >= self.training_config.total_epoch:
                    self.logger.warning("The loaded checkpoint has exceeded the total epoch number. Dry run.")
        else:
            self.initialize()
            self.logger.info("Start Training")

    def get_dataset(self):

        self.logger.info("Loading datasets...")

        if self.training_config.test_only:
            try:
                config = copy.deepcopy(self.dataset_config)
                config.update({"shift_augs": None, "rc_aug": False, "return_augs": False})
                self.data_func["test"]["dataset"] = GenomeIntervalDataset("test", **config)
            except Exception as e:
                self.logger.error(
                    "Failed to load testing dataset in `test_only` mode. Please check the preprocess setting."
                )
                self.logger.exception(e)
                exit(1)
        else:
            for split in self.data_split:
                try:
                    config = copy.deepcopy(self.dataset_config)
                    if split != "train":
                        # for valid and test, we disable the data augumentation
                        config.update({"shift_augs": None, "rc_aug": False, "return_augs": False})
                    self.data_func[split]["dataset"] = GenomeIntervalDataset(split, **config)
                except Exception as e:
                    if split == "train":
                        self.logger.error("Failed to load training dataset. Please check the preprocess setting.")
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

        if self.model_config.model_name == "borzoi":
            self.model = Borzoi.from_hparams(**self.model_config)
        else:
            self.logger.error(f"Model {self.model_config.model_name} is not implemented yet.")
            exit(1)

        self.model = self.model.to(self.local_rank, non_blocking=True)
        if self.world_size > 1:
            # Add DDP wrapper
            self.model = DDP(
                self.model, device_ids=[self.local_rank], static_graph=True, find_unused_parameters=True
            )

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

    def get_loss(self):
        self.criterion = LOSS_DICT.get(self.training_config.loss)

        if self.criterion is None:
            self.logger.error(f"Loss {self.training_config.loss} is not implemented yet.")
            exit(1)

    def initialize(self):
        # TODO: initialize the model parameter if we are going to use special initialization method
        pass

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.training_config.load_checkpoint}")
        checkpoint = torch.load(self.training_config.load_checkpoint, map_location=torch.device(self.local_rank))

        # if DDP, then the model is wrapped in module
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.current_step = checkpoint["step"]
        self.current_lr = checkpoint["lr"]
        self.best_valid_loss = checkpoint["best_valid_loss"]

        self.logger.info("Model parameters loaded successfully.")

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
                "model_state_dict": {
                    k: v.cpu() for k, v in model_to_save.state_dict().items() if isinstance(v, torch.Tensor)
                },
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
            self.current_lr = lr_scale * float(self.training_config.lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr

    def train_step(self):
        self.model.train()

        running_loss = torch.tensor(0.0, device=self.local_rank)
        tm_metrics = get_metric_collection(prefix="Train/", num_outputs=self.trial_num).to(self.local_rank)
        tm_metrics.reset()

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label) in enumerate(dataloader):
            # seq_embedding shape [batch, L, 4]
            # label shape [batch, num_central_bin (896), num_trail (93)]
            seq_embedding, label = seq_embedding.to(self.local_rank), label.to(self.local_rank)
            # pred_embedding shape [batch, num_trail (93), num_central_bin (896)]
            pred = self.model(
                seq_embedding.permute(0, 2, 1),
                self.model_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            ).permute(0, 2, 1)

            # loss
            loss = self.criterion(pred, label) / self.training_config.accum_step
            running_loss += loss.detach()

            # metrics
            tm_metrics.update(pred.reshape(-1, self.trial_num), label.reshape(-1, self.trial_num))

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
                if self.scheduler_update_freq == "batch":
                    self.scheduler.step()
                self.current_step += 1

            # log training status
            if self.current_step % self.logging_config.report_every == 0:
                report_loss = running_loss.item() * self.training_config.accum_step / (i + 1)

                if self.logging_config.use_tensorboard:
                    self.logger.metric("Train/loss", report_loss, step=self.current_step, log_also=False)
                    self.logger.metric("Train/lr", self.current_lr, step=self.current_step, log_also=False)

                    # we can only log these info in tensorboard
                    if self.logging_config.log_more:
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace(".", "/")
                            self.logger.metric(
                                "weights/" + tag,
                                value.data.cpu().numpy(),
                                self.current_step,
                                log_also=False,
                                write_hist=True,
                            )
                            try:
                                # only add gradients if they are not None
                                if value.grad is not None:
                                    self.logger.metric(
                                        "grads/" + tag,
                                        value.data.cpu().numpy(),
                                        self.current_step,
                                        log_also=False,
                                        write_hist=True,
                                    )
                            except:
                                self.logger.warning(
                                    f"failed to add grad histogram for '{tag}' in counter: {self.current_step}"
                                )
                else:
                    self.logger.info(
                        f"[Train] [Epoch {self.current_epoch}] Step {self.current_step} | Loss: {report_loss:.6f} | lr: {self.current_lr}"
                    )

            # here we reset the gradient in the final stage as we may need to log the gradient value
            if should_update:
                self.optimizer.zero_grad()

        self.current_epoch += 1

        # get and write the metric logs
        ## in the training loop, we only look at the local loss (but for other metrics, we still look at overall performance)
        self.metrics.update(tm_metrics.compute())

        if self.should_log:
            self.metrics.update({"Train/loss": running_loss * self.training_config.accum_step / len(dataloader)})

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
            for i, (seq_embedding, label) in enumerate(dataloader):
                # seq_embedding shape [batch, L, 4]
                # label shape [batch, num_central_bin (896), num_trail (93)]
                seq_embedding, label = seq_embedding.to(self.local_rank), label.to(self.local_rank)
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
            self.metrics.update({f"{log_prefix}/loss": global_avg_loss})

        if save_pred:
            # we have to pre save all the res and later aggregate them
            torch.save(
                {"label": torch.cat(labels, dim=0), "pred": torch.cat(preds, dim=0)},
                f"{self.logging_config.res_dir}/preds_rank_{self.rank}_epoch_{self.current_epoch}.pt",
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
        if myconfig.training.torch_run:
            torchrun_setup()
        else:
            setup(rank, world_size, myconfig.training.MASTER_ADDR, myconfig.training.MASTER_PORT)

        # Set up the trainer
        trainer = DNASeqModelTrainer(
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
                for k, v in trainer.metrics.items():
                    trainer.logger.metric(
                        k,
                        v.nanmean().cpu().item() if isinstance(v, torch.Tensor) else v,
                        step=trainer.current_step,
                    )
                if not myconfig.logging.use_tensorboard and trainer.should_log:
                    with open(f"{myconfig.logging.log_dir}/metrics/epoch_{trainer.current_epoch}.json", "w") as f:
                        json.dump(
                            {
                                k: v.nanmean().cpu().item() if isinstance(v, torch.Tensor) else v
                                for k, v in trainer.metrics.items()
                            },
                            f,
                            indent=4,
                        )
                blocking_sync_wait(world_size)

            # for later continue training if needed
            logger.info("Training end.")
            with timer(f"Save final checkpoint", logger, rank, world_size):
                trainer.save_checkpoint(save_name="last_epoch")

        else:
            with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                trainer.infer_step(log_loss=False, log_prefix="Test", save_pred=True)

            blocking_sync_wait(world_size)
            # save the raw testing preds
            if trainer.should_log:
                aggregate_test_res(trainer)

        blocking_sync_wait(world_size)
        cleanup(world_size)

    except Exception as e:
        logger.exception(e)
    finally:
        # in case the program doesn't stop as expected
        cleanup(world_size)
