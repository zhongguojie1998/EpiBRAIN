import glob
import json
import os

import torch
import torch.optim as optim
import torchmetrics as tm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from data.dataset import DumySampler, GenomeIntervalDataset, StrictDistributedSampler
from model.pytorch_borzoi_model import Borzoi
from utils.logging import LOGGER_PREFIX, LazyLogger, timer
from utils.loss import LOSS_DICT
from utils.multi_gpu import blocking_sync_wait, cleanup, global_aggregate, setup


def get_metric_collection(prefix: str = "Validation/"):
    collection = tm.MetricCollection(
        {
            "MSE": tm.MeanSquaredError(),
            "MAE": tm.MeanAbsoluteError(),
            "PearsonR": tm.PearsonCorrCoef(),
            "SpearmanR": tm.SpearmanCorrCoef(),
        },
        prefix=prefix,
    )
    return collection


class DNASeqModelTrainer:
    def __init__(self, config, rank, world_size, logger):

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
        self.world_size = world_size
        self.should_log = (self.world_size > 1 and self.rank == 0) or self.world_size == 1

        # get the training settings
        self.current_epoch = 0
        self.best_valid_loss = torch.inf
        self.metrics = {}

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
            self.load_checkpoint()
            if self.training_config.test_only:
                self.logger.info("Only Testing")
            else:
                self.logger.info("Continue Training")
                if self.current_epoch >= self.training_config.total_epoch:
                    self.logger.warning("The loaded checkpoint has exceeded the total epoch number. Dry run.")
        else:
            self.logger.info("Start Training")

    def get_dataset(self):

        self.logger.info("Loading datasets...")

        if self.training_config.test_only:
            try:
                self.data_func["test"]["dataset"] = GenomeIntervalDataset("test", **self.dataset_config)
            except:
                self.logger.error(
                    "Failed to load testing dataset in `test_only` mode. Please check the preprocess setting."
                )
                exit(1)
        else:
            for split in self.data_split:
                try:
                    self.data_func[split]["dataset"] = GenomeIntervalDataset(split, **self.dataset_config)
                except:
                    if split == "train":
                        self.logger.error("Failed to load training dataset. Please check the preprocess setting.")
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
            sampler_cls = StrictDistributedSampler if dataset is not None else DumySampler

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

        self.model = self.model.to(self.rank, non_blocking=True)
        if self.world_size > 1:
            # Add DDP wrapper
            self.model = DDP(self.model, device_ids=[self.rank], static_graph=True, find_unused_parameters=True)

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

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.training_config.load_checkpoint}")
        checkpoint = torch.load(self.training_config.load_checkpoint, map_location=torch.device(self.rank))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
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
                "model_state_dict": {
                    k: v.cpu() for k, v in model_to_save.state_dict().items() if isinstance(v, torch.Tensor)
                },
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_valid_loss": self.best_valid_loss,
            }
            torch.save(checkpoint, f"{self.logging_config.checkpoint_dir}/chk_epoch_{save_name}.pt")
        self.logger.info("Checkpoint saved successfully.")

    def train_step(self):
        self.model.train()

        running_loss = torch.tensor(0.0, device=self.rank)
        tm_metrics = get_metric_collection(prefix="Train/").to(self.rank)
        tm_metrics.reset()

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label) in enumerate(dataloader):
            # seq_embedding shape [batch, L, 4]
            # label shape [batch, num_central_bin (896), num_trail (93)]
            seq_embedding, label = seq_embedding.to(self.rank), label.to(self.rank)
            # pred_embedding shape [batch, num_trail (93), num_central_bin (896)]
            pred = self.model(
                seq_embedding.permute(0, 2, 1),
                self.model_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            ).permute(0, 2, 1)

            # loss
            loss = self.criterion(pred, label) / self.training_config.accum_step
            running_loss += loss.detach()
            self.logger.debug(f"Batch {i}, loss {loss.detach().item():.3f}")

            # metrics
            tm_metrics.update(pred, label)

            # whether to do the loss aggregation
            is_accum_step = ((i + 1) % self.training_config.accum_step == 0) or (i + 1 == len(dataloader))
            if self.world_size > 1 and not is_accum_step:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler_update_freq == "batch":
                    self.scheduler.step()

        self.current_epoch += 1

        # get and write the metric logs
        ## in the training loop, we only look at the local loss (but for other metrics, we still look at overall performance)
        self.metrics.update(tm_metrics.compute(sync_dist=self.world_size > 1))
        self.metrics.update({"Train/loss": running_loss * self.training_config.accum_step / len(dataloader)})

    def infer_step(self, log_loss=False, save_pred=False, log_prefix="Valid"):
        self.model.eval()

        if log_loss:
            running_loss_local = torch.tensor(0.0, device=self.rank)
        if save_pred:
            preds = []
            labels = []

        tm_metrics = get_metric_collection(prefix=f"{log_prefix}/").to(self.rank)
        tm_metrics.reset()

        dataloader = self.data_func[log_prefix.lower()]["data_loader"]

        with torch.no_grad():
            for i, (seq_embedding, label) in enumerate(dataloader):
                # seq_embedding shape [batch, L, 4]
                # label shape [batch, num_central_bin (896), num_trail (93)]
                seq_embedding, label = seq_embedding.to(self.rank), label.to(self.rank)
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
                tm_metrics.update(pred, label)

        # get and write the metric logs
        self.metrics.update(tm_metrics.compute(sync_dist=self.world_size > 1))
        # get global loss
        if log_loss:
            total_loss = running_loss_local.clone()
            global_aggregate(total_loss, aggregate="sum", world_size=self.world_size)
            if self.should_log:
                global_avg_loss = total_loss / (len(dataloader) * self.world_size)
                self.metrics.update({f"{log_prefix}/loss": global_avg_loss})
        if save_pred:
            # we have to pre save all the res and later aggregate them
            torch.save(
                {"label": torch.cat(labels, dim=0), "pred": torch.cat(preds, dim=0)},
                f"{self.logging_config.res_dir}/preds_rank{self.rank}_epoch_{self.current_epoch}.pt",
            )


# this function also support single GPU training
def mp_main(rank, world_size, myconfig):
    # special train loggers which can also log to tensorboard
    logger = LazyLogger(f"{LOGGER_PREFIX}-Trainer:rank_{rank}")

    # if we have multiple GPUs, we need to set up DDP
    setup(rank, world_size, myconfig.training.MASTER_ADDR, myconfig.training.MASTER_PORT)

    # Set up the trainer
    trainer = DNASeqModelTrainer(myconfig, rank, world_size, logger)
    blocking_sync_wait(world_size)

    if not myconfig.training.test_only:
        for epoch in range(trainer.current_epoch, myconfig.training.total_epoch):
            save_test_res = False

            for split in trainer.data_split:
                trainer.data_func[split]["data_sampler"].set_epoch(epoch)

            if trainer.data_func["train"]["data_loader"] is not None:
                with timer(f"[Train] [Epoch {epoch}]", logger, rank, world_size):
                    trainer.train_step()

                blocking_sync_wait(world_size)

            trainer.logger.info(f"Current Epoch: {trainer.current_epoch}")

            if trainer.data_func["valid"]["data_loader"] is not None:
                with timer(f"[Valid] [Epoch {epoch}]", logger, rank, world_size):
                    trainer.infer_step(log_loss=True, log_prefix="Valid", save_pred=False)

                blocking_sync_wait(world_size)

                valid_loss = trainer.metrics["Valid/loss"]
                if valid_loss < trainer.best_valid_loss:
                    trainer.best_valid_loss = valid_loss
                    logger.info(f"New best validation loss: {valid_loss:.6f}")
                    logger.info("Saving new best model")
                    trainer.save_checkpoint(save_name="best_valid_loss")
                    save_test_res = True

                if trainer.scheduler_update_freq == "epoch":
                    trainer.scheduler.step(valid_loss)

                blocking_sync_wait(world_size)

            # write the model
            if trainer.current_epoch % myconfig.logging.save_every == 0:
                trainer.save_checkpoint()
                # save_test_res = True

            if trainer.data_func["test"]["data_loader"] is not None:
                with timer(f"[Test] [Epoch {epoch}]", logger, rank, world_size):
                    trainer.infer_step(log_loss=True, log_prefix="Test", save_pred=save_test_res)

                blocking_sync_wait(world_size)

            # write the metrics
            ## if we use tensorboard, the metrics will be written into tb (initialized when creating the training logger)
            for k, v in trainer.metrics.items():
                trainer.logger.metric(k, v.item(), step=trainer.current_epoch)
            if not myconfig.logging.use_tensorboard and trainer.should_log:
                with open(f"{myconfig.logging.log_dir}/metrics/epoch_{trainer.current_epoch}.json", "w") as f:
                    json.dump(
                        {
                            k: v.cpu().item() if isinstance(v, torch.tensor) else v
                            for k, v in trainer.metrics.items()
                        },
                        f,
                        indent=4,
                    )
            if save_test_res and trainer.should_log:
                # aggregate the results and clear per rank file
                pattern = f"{trainer.logging_config.res_dir}/preds_rank*_epoch_{trainer.current_epoch}.pt"

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

        # for later continue training if needed
        trainer.save_checkpoint(save_name="last_epoch")

    else:
        with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
            trainer.infer_step(log_loss=False, log_prefix="Test", save_pred=True)

    blocking_sync_wait(world_size)
    cleanup(world_size)
