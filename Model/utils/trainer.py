import copy
import glob
import itertools
import json
import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD
import torch.optim as optim
import torchmetrics as tm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from data.dataset import DumySampler, GenomeIntervalDataset, StrictDistributedSampler, collate_fn
from model.model_utils import setup_model, std_pred_head_config
from utils.logging import LOGGER_PREFIX, TrainingLogger, timer
from utils.loss import LOSS_DICT
from utils.multi_gpu import blocking_sync_wait, cleanup, deepspeed_setup, global_aggregate, setup, torchrun_setup
from utils.scheduler import SCHEDULER_DICT


def create_optimizer_grouped_parameters(model, use_groups=True):
    """
    Create optimizer parameter groups with differential weight decay:
    - overall_decay_params: 1.0e-6 weight decay for most parameters
    - transformer_decay_params: 2.0e-8 weight decay for transformer layers
    - no_decay_params: 0.0 weight decay for biases and 1D parameters

    Args:
        model: The model to extract parameters from
        use_groups: Whether to use grouped parameters or return all parameters

    Returns:
        list: Parameter groups suitable for optimizer initialization
    """
    if not use_groups:
        return [param for param in model.parameters() if param.requires_grad]

    overall_decay_params = []
    transformer_decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check for parameters to exclude from weight decay
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            if "transformer" in name:
                # remove LoRA parameters from transformer decay group
                if "lora" in name:
                    no_decay_params.append(param)
                else:
                    transformer_decay_params.append(param)
            else:
                overall_decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": overall_decay_params, "weight_decay": 4.0e-8},
        {"params": transformer_decay_params, "weight_decay": 2.0e-8},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return optimizer_grouped_parameters


def aggregate_test_res(trainer, prefix="Test", remove_raw=True):
    # aggregate the results and clear per rank file
    pattern = f"{trainer.logging_config.res_dir}/{prefix}_preds_rank_*_epoch_{trainer.current_epoch}*.pt"

    file_list = glob.glob(pattern)
    if not file_list:
        trainer.logger.warning(f"{prefix} res aggregation failed (cannot find any pred file), skip")
    else:
        all_labels = {}
        all_preds = {}
        all_inds = []
        for file_path in file_list:
            data = torch.load(file_path, map_location="cpu")

            all_inds.append(data["index"])
            for k, v in data["label"].items():
                if k not in all_labels and k != "transcripts_mask":
                    all_labels[k] = []
                if k != "transcripts_mask":
                    all_labels[k].append(v)
            for k, v in data["pred"].items():
                if k not in all_preds:
                    all_preds[k] = []
                all_preds[k].append(v)

        all_labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}
        all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        all_inds = torch.cat(all_inds, dim=0)

        torch.save(
            {"label": all_labels, "pred": all_preds, "index": all_inds},
            f"{trainer.logging_config.res_dir}/{prefix}_preds_epoch_{trainer.current_epoch}.pt",
        )
        if remove_raw:
            for file_path in file_list:
                os.remove(file_path)


class CrossColumnPearsonR(tm.Metric):
    """
    Accumulates all batches into large MxN array, then calculates 
    Pearson correlation across columns, gets N values, then averages them.
    """
    def __init__(self):
        super().__init__()
        self.add_state("corr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds, target: MxN tensors where M is samples, N is features/columns
        Calculate correlations for current batch and accumulate
        """
        assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"

        # Calculate correlation for each row in current batch
        batch_corrs = []
        for i in range(preds.shape[0]):
            pred_row = preds[i, :]
            target_row = target[i, :]

            # Calculate Pearson correlation efficiently for two vectors
            pred_centered = pred_row - pred_row.mean()
            target_centered = target_row - target_row.mean()

            # Compute correlation using dot product formula
            numerator = torch.sum(pred_centered * target_centered)
            pred_norm = torch.norm(pred_centered)
            target_norm = torch.norm(target_centered)

            # Avoid division by zero
            if pred_norm > 1e-8 and target_norm > 1e-8:
                corr = numerator / (pred_norm * target_norm)
                batch_corrs.append(corr)

        if batch_corrs:
            # Accumulate sum of correlations and count of valid samples
            batch_corr_sum = torch.sum(torch.stack(batch_corrs))
            self.corr_sum += batch_corr_sum
            self.total_samples += len(batch_corrs)

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0)

        # Return mean of all accumulated correlations
        return self.corr_sum / self.total_samples


def get_metric_collection(prefix: str = "Valid/", config={}):

    metric_factories = {
        "regression": {
            "MSE": lambda **kw: tm.MeanSquaredError(**kw),
            "MAE": lambda **kw: tm.MeanAbsoluteError(**kw),
            "PearsonR": lambda **kw: tm.PearsonCorrCoef(**kw),
            "TranscriptsPearsonR": lambda **kw: tm.PearsonCorrCoef(**kw),
        },
        "classification": {
            "Accuracy": lambda **kw: tm.Accuracy(task="multiclass", **kw),
            "F1Score": lambda **kw: tm.F1Score(task="multiclass", average="macro", **kw),
            "Precision": lambda **kw: tm.Precision(task="multiclass", average="macro", **kw),
            "Recall": lambda **kw: tm.Recall(task="multiclass", average="macro", **kw),
        },
    }

    metrics_dict = {}
    for head_name, head_cfg in config.items():
        task = head_cfg.get("task_type")
        trials = head_cfg["label_meta"]["trial"].to_list()
        cls_num = head_cfg.get("class_num")

        if task not in metric_factories:
            raise ValueError(f"Unknown task_type '{task}' for head '{head_name}'")

        init_kwargs = {}
        if task == "classification":
            if cls_num is None:
                raise ValueError(f"class_num is required for classification head '{head_name}'")
            init_kwargs["num_classes"] = cls_num

        for trial, (metric_name, factory) in itertools.product(trials, metric_factories[task].items()):
            key = f"{trial}/{head_name}/{metric_name}"
            # only RNA has "TranscriptsPearsonR"
            if metric_name == "TranscriptsPearsonR" and "RNA" not in trial:
                continue
            metrics_dict[key] = factory(**init_kwargs)
    
        # add cross-cell metrics
        for modality, _ in head_cfg["label_meta"].groupby("modality"):
            for metric_name, factory in metric_factories[task].items():
                if metric_name == "TranscriptsPearsonR" and "RNA" not in modality:
                    continue
                # cross cell metric only applies to PearsonR
                if "PearsonR" in metric_name and task == "regression":
                    key = f"{modality}/{head_name}/cross_cell/{metric_name}"
                    metrics_dict[key] = CrossColumnPearsonR()
                else:
                    # average metric
                    key = f"{modality}/{head_name}/average/{metric_name}"
                    metrics_dict[key] = factory(**init_kwargs)

    return tm.MetricCollection(metrics_dict, prefix=prefix), {k: v.keys() for k, v in metric_factories.items()}


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
        # Set device: if rank is "cpu" or "cuda:X", use it directly; otherwise treat as device id
        if isinstance(self.local_rank, str):
            self.device = self.local_rank
        elif torch.cuda.is_available():
            self.device = self.local_rank
        else:
            self.device = "cpu"

        # set up model and data, make sure the logic aligned
        self.model_data_align()

        # get the training settings
        self.current_epoch = 0
        self.current_step = 0  # based on update step
        self.best_valid_loss = torch.inf
        self.metrics = {}

        # set up data
        self.data_split = self.config.data.used_dataset
        self.data_func = {
            k: {"dataset": None, "data_sampler": None, "data_loader": None} for k in ["train", "valid", "test"]
        }
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

    def model_data_align(self):

        # get the model setting
        heads_config, use_cell_encoder, _ = std_pred_head_config(self.model_config.output_heads)

        # get the data setting
        raw_label_meta = pd.read_csv(f"{self.dataset_config.storage_path}/raw_label_meta.csv")

        # assign data config to each head
        self.head_data_setting = {}
        for head_name, head_config in heads_config.items():
            if head_name not in self.training_config.use_head:
                continue

            task = head_config["task"]
            assert task in raw_label_meta["task"].to_list(), f"Task {task} not found in label meta"

            task_label_meta = raw_label_meta[raw_label_meta["task"] == task].reset_index(drop=True).copy()

            if not use_cell_encoder:
                assert (
                    len(task_label_meta) == head_config["track_num"]
                ), f"Model setting track number of head {head_name}, task {task} ({head_config['track_num']}) mismatch with label meta ({len(task_label_meta)})"
            else:
                unique_cell_types = sorted(
                    raw_label_meta["cell_type"].unique()
                )  # here we use all cell types from all tasks because cell type embedding is shared across all tasks
                assert (
                    len(unique_cell_types) == head_config["celltype_num"]
                ), f"Model setting cell type number ({head_config['celltype_num']}) mismatch all the cell types in label meta ({unique_cell_types})"
                unique_modalities = sorted(task_label_meta["modality"].unique())
                assert (
                    len(unique_modalities) == head_config["modality_num"]
                ), f"Model setting modality number, task {task} ({head_config['modality_num']}) mismatch modalities in label meta ({len(unique_modalities)})"

                ## in model, we generate all modalities for each cell type, we save the dims with label info in the label_meta index
                full_info = pd.DataFrame()
                for p, (i, j) in enumerate(itertools.product(unique_cell_types, unique_modalities)):
                    full_info.loc[p, ["cell_type", "modality"]] = [i, j]
                task_label_meta = full_info.merge(
                    task_label_meta, on=["cell_type", "modality"], how="left"
                ).dropna()

            task_label_meta["label_dim"] = range(len(task_label_meta))
            task_label_meta.index.name = "dim"
            self.head_data_setting[head_name] = {
                "label_meta": task_label_meta,
                "task_type": task,
                "class_num": head_config["class_num"],
            }
            task_label_meta.to_csv(f"{self.logging_config.log_dir}/{task}_label_meta.csv", index=True)

    def get_dataset(self):

        self.logger.info("Loading datasets...")

        self.data_rand_seed = mp.Value("i", 0)

        for split in self.data_split:
            try:
                config = copy.deepcopy(self.dataset_config)
                if split != "train":
                    # for valid and test, we disable the data augumentation
                    config.update({"shift_augs": None, "rc_aug": False, "return_augs": False})
                # if transcripts are in self.training_config.loss.keys(), we need to return the transcripts mask
                load_task = set([v["task_type"] for v in self.head_data_setting.values()])
                if any(["transcripts" in i for i in self.training_config.loss.keys()]):
                    load_task.add("transcripts")
                self.data_func[split]["dataset"] = GenomeIntervalDataset(
                    split,
                    **config,
                    external_rand_seed=self.data_rand_seed,
                    load_task=load_task,
                )

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
            f"{'/'.join([k for k,v in self.data_func.items() if v['dataset'] is not None])} datasets loaded successfully."
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
                shuffle=split == "train",
            )
            self.data_func[split]["data_sampler"] = sampler

            # get the dataloader
            self.data_func[split]["data_loader"] = (
                DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    shuffle=False,  # set shuffle in sampler, dataloader should be set as False
                    drop_last=split == "train",
                    collate_fn=collate_fn,
                    **self.training_config.dataloader_params,
                )
                if dataset is not None
                else None
            )

    def get_model(self):

        self.logger.info("Loading model...")

        self.model = setup_model(self.config, self.logger)

        # since the batchsize is small, we need to sync batchnorm statistics
        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # if necessary, load the checkpoint
        ## load the full model before we wrap the model into DDP
        if self.training_config.load_checkpoint is not None:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])

        # send the model to training device
        self.model = self.model.to(self.device, non_blocking=True)
        if self.world_size > 1:
            # Add DDP wrapper
            if torch.cuda.is_available():
                self.model = DDP(
                    self.model, device_ids=[self.local_rank], static_graph=True, find_unused_parameters=True
                )
            else:
                self.model = DDP(
                    self.model, static_graph=True, find_unused_parameters=True
                )

            # If applicable, add gradient compression hook
            if self.training_config.use_grad_compression:
                self.hook_state = PowerSGD.PowerSGDState(
                    process_group=None,  # we do compression on all ranks
                    **self.training_config.get("powerSGD_params", {}),
                )
                if self.training_config.load_checkpoint is not None and self.checkpoint.get("hook_state", {}):
                    self.hook_state.__setstate__(self.checkpoint["hook_state"])

                self.model.register_comm_hook(self.hook_state, PowerSGD.powerSGD_hook)

        self.logger.info(f"Model {self.model_config.model_name} loaded successfully.")

    def get_optimizer(self):
        # get optim
        optim_class = eval(f"optim.{self.training_config.optimizer}")
        # Apply differential weight decay using the shared function
        optimizer_grouped_parameters = create_optimizer_grouped_parameters(
            self.model, self.training_config.get("add_opt_group", False)
        )
        self.optimizer = optim_class(optimizer_grouped_parameters, **self.training_config.optimizer_params)

        # get scheduler
        scheduler_class = SCHEDULER_DICT.get(self.training_config.scheduler)
        if scheduler_class is None:
            try:
                scheduler_class = eval(f"optim.lr_scheduler.{self.training_config.scheduler}")
            except:
                self.logger.error(f"Scheduler {self.training_config.scheduler} is not implemented yet.")
                exit(1)
        self.scheduler = scheduler_class(self.optimizer, **self.training_config.scheduler_params)

        # get initial lr
        self.current_lr = self.scheduler.get_last_lr()[0]

        if "ReduceLROnPlateau" in self.training_config.scheduler:
            self.scheduler_need_monitor = True
            # we should further check if valid set is available
            if not self.training_config.test_only and self.data_func["valid"]["dataset"] is None:
                self.logger.error("The given scheduler requires the valid dataset")
                exit(1)
        else:
            self.scheduler_need_monitor = False

        # if necessary, load the checkpoint
        if self.training_config.load_checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            self.current_lr = self.checkpoint["lr"]

    def get_loss(self):
        # Loss configuration format: {"loss_name": {"loss_weight": float, "loss_head": str, "params": dict}}
        loss_config = self.training_config.loss

        self.loss_config = {}
        loss_heads = set()

        for loss_name, loss_info in loss_config.items():
            loss_class = LOSS_DICT.get(loss_name)
            if loss_class is None:
                self.logger.error(f"Loss {loss_name} is not implemented yet.")
                exit(1)

            loss_head = loss_info["loss_head"]
            loss_heads.add(loss_head)

            self.loss_config[loss_name] = {
                "criterion": loss_class(**loss_info.get("params", {})),
                "weight": loss_info["loss_weight"],
                "head": loss_head,
            }

        # Check if loss_head matches use_head
        if loss_heads != set(self.training_config.use_head):
            raise ValueError(f"Loss heads {loss_heads} do not match use_head {set(self.training_config.use_head)}")

    def compute_transcripts(self, pred_subset, label_subset, transcripts_mask, cell_data):
        # take pred and label for RNA predictions, aggregate by transcripts_mask
        ## transform label_subset back to original scale
        ### scale para
        scale_tensor = (
            torch.tensor(cell_data["scale"].values, device=label_subset.device, dtype=pred_subset.dtype).unsqueeze(0).unsqueeze(0)
        )  # bsz x n_window x cell_types
        label_subset = label_subset / scale_tensor
        pred_subset = pred_subset / scale_tensor
        ### soft clip para
        if "clip_soft" not in cell_data.columns:
            soft_clip_tensor = torch.full((cell_data.shape[0],), torch.inf, device=label_subset.device, dtype=pred_subset.dtype)
        else:
            clip_arr = cell_data["clip_soft"].fillna(torch.inf).values
            soft_clip_tensor = torch.tensor(clip_arr, device=label_subset.device, dtype=pred_subset.dtype)

        soft_clip_tensor = soft_clip_tensor.unsqueeze(0).unsqueeze(0)  # -> (1,1,cell_types)

        #### get soft clip back to original scale
        label_subset = torch.where(
            label_subset > soft_clip_tensor,
            (label_subset - soft_clip_tensor + 1) ** 2 + soft_clip_tensor - 1,
            label_subset,
        )
        ### sum stat para
        sum_stat_tensor = (
            torch.tensor(
                np.where(cell_data["sum_stat"] == "sum_three_quarter", 4 / 3, 1.0),
                device=label_subset.device,
                dtype=pred_subset.dtype
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # -> shape (1,1,cell_types)
        label_subset = label_subset**sum_stat_tensor
        ## transform pred_subset back to original scale
        pred_subset = pred_subset**sum_stat_tensor

        # aggregate by label_mask
        pred_transcripts = torch.einsum(
            "tbn,bnc->tc", transcripts_mask, pred_subset
        )  # total_transcripts x cell_types
        label_transcripts = torch.einsum(
            "tbn,bnc->tc", transcripts_mask, label_subset
        )
        return pred_transcripts, label_transcripts

    def compute_loss(self, pred, label):
        loss_dict = {}
        total_loss = 0.0

        for loss_name, loss_info in self.loss_config.items():
            head_name = loss_info["head"]
            criterion = loss_info["criterion"]
            weight = loss_info["weight"]

            # Get head data setting to find label indices and task type
            head_data = self.head_data_setting[head_name]
            task_type = head_data["task_type"]
            label_meta = head_data["label_meta"]

            # Extract prediction subset for this head
            # pred[head_name] shape: [batch, seq_len, total_tracks]
            task_pred = pred[head_name]
            # Get corresponding labels
            task_label = label[task_type].to(self.device, non_blocking=True)
            # Compute loss
            if "transcripts" in loss_name:
                # get transcripts masks
                if "transcripts_mask" in label:
                    transcripts_mask = label["transcripts_mask"].to(
                        self.device, non_blocking=True, dtype=task_pred.dtype
                    )  # total_transcripts x bsz x n_window
                else:
                    raise ValueError(f"Transcripts mask not found in label for loss {loss_name}")
                # TODO: split the transcript info to minus strand and plus strand, and handle them separately
                cell_data = label_meta[label_meta["modality"].str.contains("RNA")]
                pred_subset = task_pred[:, :, cell_data.index.tolist(), ...]  # bsz x n_window x cell_types
                label_subset = task_label[:, :, cell_data["label_dim"].tolist()]  # bsz x n_window x cell_types

                pred_transcripts, label_transcripts = self.compute_transcripts(
                    pred_subset, label_subset, transcripts_mask, cell_data
                )
                # unsqueeze to make it 1 x total_transcripts x cell_types, looks like batch size 1
                loss_value = criterion(pred_transcripts.unsqueeze(0), label_transcripts.unsqueeze(0))
            if "cross_cell" in loss_name:
                loss_value = 0
                for _, cell_data in label_meta.groupby("modality"):
                    pred_subset = task_pred[:, :, cell_data.index.tolist(), ...]
                    label_subset = task_label[:, :, cell_data["label_dim"].tolist()]
                    loss_value += criterion(pred_subset, label_subset)
            else:
                # We need to slice the tracks dimension based on label_meta
                pred_subset = task_pred[:, :, label_meta.index.tolist(), ...]
                loss_value = criterion(pred_subset, task_label)

            loss_dict[loss_name] = loss_value.detach().cpu().item()

            # Add to total loss with weight
            total_loss += weight * loss_value

        # Add total loss to dict for unified handling
        loss_dict["total_loss"] = total_loss.detach().cpu().item()

        return total_loss, loss_dict

    @property
    def inference_model(self):
        """Property to access the model for inference, allowing subclasses to override."""
        return self.model

    @property
    def training_model(self):
        """Property to access the model for training logging, allowing subclasses to override."""
        # Unwrap DDP and torch.compile wrappers for parameter access
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _log_training_metrics(self, report_loss, should_exit_on_nan=False):
        """Shared training metrics logging logic with NaN detection and exit capability."""
        nan_detected = False

        if self.logging_config.use_tensorboard:
            for k, v in report_loss.items():
                self.logger.metric(
                    f"Train/rank[{self.rank}]_{k}",
                    v,
                    step=self.current_step,
                    log_also=False,
                    diagnose=self.logging_config.diagnose,
                )
            self.logger.metric("Train/lr", self.current_lr, step=self.current_step, log_also=False)

            # we can only log these info in tensorboard
            if self.logging_config.log_more:
                # track sqrtsuml2 of gradients and weights
                weights = []
                grads = []
                for tag, value in self.training_model.named_parameters():
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
                        self.logger.metric(
                            "weights_norm/" + tag,
                            np.sqrt(np.linalg.norm(weight) ** 2),
                            self.current_step,
                            log_also=False,
                        )
                        weights.append(weight)
                    else:
                        self.logger.warning(
                            f"failed to add weight histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                        )
                        nan_detected = True

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
                            self.logger.metric(
                                "grads_norm/" + tag,
                                np.sqrt(np.linalg.norm(grad) ** 2),
                                self.current_step,
                                log_also=False,
                            )
                            grads.append(grad)
                        else:
                            self.logger.warning(
                                f"failed to add grad histogram for '{tag}' in counter: {self.current_step}, Nan occur!"
                            )
                            nan_detected = True
                # log total weights and grads
                weights_norm = np.sqrt(sum(np.linalg.norm(w) ** 2 for w in weights))
                grads_norm = np.sqrt(sum(np.linalg.norm(g) ** 2 for g in grads))
                self.logger.metric("grads_norm/total", grads_norm, self.current_step, log_also=False)
                self.logger.metric("weights_norm/total", weights_norm, self.current_step, log_also=False)
        else:
            for k, v in report_loss.items():
                self.logger.info(f"[Train] [Epoch {self.current_epoch}] Step {self.current_step} | {k}: {v:.6f}")

        if should_exit_on_nan and nan_detected:
            self.logger.error("NaN detected in model weight/gradients. Exiting after this step.")
            return True
        else:
            return False

    def _diagnose_extra_log(self, ind):
        ## output the sample index
        self.logger.debug(
            f"batch id for step {self.current_step} is {','.join([str(int(i)) for i in ind])}",
            diagnose=self.logging_config.diagnose,
        )
        if self.current_step % self.logging_config.get("diagnose_save_step", 1) == 0:
            self.save_checkpoint(f"diag_step_{self.current_step}")

    def load_checkpoint(self):

        self.logger.info(f"Loading checkpoint from {self.training_config.load_checkpoint}")
        # if load_checkpoint is best_valid_loss, we need to find the actual file
        if self.training_config.load_checkpoint == "best_valid_loss":
            load_file = f"{self.logging_config.checkpoint_dir}/chk_epoch_best_valid_loss.pt"
        # else is a epoch number, which doesn't end with .pt
        elif isinstance(self.training_config.load_checkpoint, int) or not self.training_config.load_checkpoint.endswith(".pt"):
            load_file = f"{self.logging_config.checkpoint_dir}/chk_epoch_{self.training_config.load_checkpoint}.pt"
        else:
            load_file = self.training_config.load_checkpoint
        self.checkpoint = torch.load(
            load_file, map_location=torch.device(self.device)
        )

        # since this is first called in the model initialization pipeline, the model and optimizers loads the checkpoint in their own functions instead of here
        self.current_epoch = self.checkpoint["epoch"]
        self.current_step = self.checkpoint["step"]
        self.best_valid_loss = self.checkpoint["best_valid_loss"]

        self.logger.info("Checkpoint loaded successfully.")

    def save_checkpoint(self, save_name=None):

        if save_name is None:
            save_name = self.current_epoch

        # Unwrap model from DDP and torch.compile wrappers
        model_to_save = self.model
        # First unwrap DDP if present
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        # Then unwrap torch.compile if present
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod

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
            if self.training_config.use_grad_compression:
                # assume `state` is in scope or saved as self.powerSGD_state
                checkpoint["hook_state"] = self.hook_state.__getstate__()

            torch.save(checkpoint, f"{self.logging_config.checkpoint_dir}/chk_epoch_{save_name}.pt")
        self.logger.info("Checkpoint saved successfully.")
        blocking_sync_wait(self.world_size)

    def train_step(self):
        self.model.train()

        # for loss log
        nan_termination = False
        # for loss tracking
        batch_loss_dict = {}
        batch_count = 0
        epoch_loss_list = []

        # for other metric log
        # tm_metrics = get_metric_collection(prefix="Train/", num_outputs=self.trial_num).to(self.local_rank)
        # tm_metrics.reset()

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label, ind) in enumerate(dataloader):

            # seq_embedding shape [batch, L, 4]
            # label is now a dictionary {task_type: tensor}
            seq_embedding = seq_embedding.to(self.device, non_blocking=True)
            # pred is now a dictionary {head_name: tensor}
            pred = self.model(
                seq_embedding.permute(0, 2, 1),
                self.training_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            )

            # loss - use new compute_loss function
            total_loss, loss_dict = self.compute_loss(pred, label)
            loss = total_loss / self.training_config.accum_step

            # log loss
            for k, v in loss_dict.items():
                if k not in batch_loss_dict:
                    batch_loss_dict[k] = 0.0
                batch_loss_dict[k] += v
            batch_count += 1
            epoch_loss_list.append(loss.detach().cpu().item() * self.training_config.accum_step)

            # metrics
            # tm_metrics.update(pred.reshape(-1, self.trial_num).double(), label.reshape(-1, self.trial_num).double())

            # whether to do the loss aggregation
            should_update = ((i + 1) % self.training_config.accum_step == 0) or (i + 1 == len(dataloader))
            if self.world_size > 1 and not should_update:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if should_update:
                # log training status
                ## in the training loop, we only look at the local loss
                tensorboard_log_every = self.logging_config.get("tensorboard_log_every") or self.logging_config.report_every
                if self.current_step % tensorboard_log_every == 0:
                    report_loss = {k: v / batch_count for k, v in batch_loss_dict.items()}

                    nan_termination = self._log_training_metrics(report_loss, should_exit_on_nan=True)

                    batch_loss_dict = {}
                    batch_count = 0

                # model step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.clip_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.logging_config.diagnose:
                self._diagnose_extra_log(ind)

            if should_update:
                self.current_lr = self.scheduler.get_last_lr()[0]
                self.current_step += 1

            if nan_termination:
                exit(1)

        self.current_epoch += 1
        self.metrics.update({f"Train/epoch_avg_loss": np.mean(epoch_loss_list)})
        # get and write the metric logs
        # self.metrics.update(tm_metrics.compute())

    def infer_step(self, log_loss=False, save_pred=False, save_method="merge", log_prefix="Valid"):
        self.inference_model.eval()

        if log_loss:
            running_loss_local = torch.tensor(0.0, device=self.device)
        if save_pred and save_method == "merge":
            preds = {}
            labels = {}
            inds = []

        tm_metrics, metric_name_dict = get_metric_collection(
            prefix=f"{log_prefix}/", config=self.head_data_setting
        )
        tm_metrics.to(self.device)
        tm_metrics.reset()

        dataloader = self.data_func[log_prefix.lower()]["data_loader"]

        with torch.no_grad():
            for i, (seq_embedding, label, ind) in enumerate(dataloader):
                # seq_embedding shape [batch, L, 4]
                # label is now a dictionary {task_type: tensor}
                seq_embedding = seq_embedding.to(self.device, non_blocking=True)
                # pred is now a dictionary {head_name: tensor}
                pred = self.inference_model(
                    seq_embedding.permute(0, 2, 1),
                    self.training_config.use_head,
                    data_parallel_training=True if self.world_size > 1 else False,
                )

                # loss - use new compute_loss function
                if log_loss:
                    total_loss, loss_dict = self.compute_loss(pred, label)
                    running_loss_local += total_loss.detach()
                # pred
                if save_pred:
                    if save_method == "merge":
                        for k, v in pred.items():
                            if k not in preds:
                                preds[k] = []
                            preds[k].append(v.detach().cpu())
                        for k, v in label.items():
                            if k not in labels:
                                labels[k] = []
                            labels[k].append(v.detach().cpu())
                        inds.append(ind)
                    else:
                        torch.save(
                            {
                                "label": {k: v.detach().cpu() for k, v in label.items()},
                                "pred": {k: v.detach().cpu() for k, v in pred.items()},
                                "index": ind,
                            },
                            f"{self.logging_config.res_dir}/{log_prefix}_preds_rank_{self.rank}_epoch_{self.current_epoch}_batch_{i}.pt",
                        )

                # metrics for each track, skip TranscriptsPearsonR first
                for head_name, head_data in self.head_data_setting.items():
                    task_type = head_data["task_type"]
                    label_meta = head_data["label_meta"].reset_index().set_index("trial")
                    for trial in label_meta.index:
                        pred_dim, label_dim = label_meta.loc[trial, ["dim", "label_dim"]]
                        label_subset = label[task_type][:, :, label_dim].to(self.device, non_blocking=True)
                        pred_subset = pred[head_name][:, :, pred_dim, ...]
                        B, L = pred_subset.shape[:2]
                        # Update metrics for this head
                        for metric_name in metric_name_dict[task_type]:
                            key = f"{trial}/{head_name}/{metric_name}"
                            # only update for no-transcripts metrics
                            if "Transcripts" not in metric_name:
                                tm_metrics[key].update(
                                    pred_subset.reshape(B * L, -1).double(),
                                    label_subset.reshape(B * L, -1).double(),
                                )
                    # special handling of TranscriptsPearsonR - we need to aggregate the transcripts by transcripts_mask
                    # get transcripts masks
                    if "transcripts_mask" in label:
                        transcripts_mask = label["transcripts_mask"].to(
                            self.device, non_blocking=True, dtype=pred_subset.dtype
                        )  # total_transcripts x bsz x n_window
                        # TODO: split the transcript info to minus strand and plus strand, and handle them separately
                        for mod in label_meta['modality'][label_meta["modality"].str.contains("RNA")].unique():
                            cell_data = label_meta[label_meta["modality"] == mod]
                            pred_subset = pred[head_name][:, :, cell_data["dim"].tolist(), ...]  # bsz x n_window x cell_types
                            label_subset = label[task_type][:, :, cell_data["label_dim"].tolist(), ...].to(self.device, non_blocking=True)  # bsz x n_window x cell_types
                            pred_transcripts, label_transcripts = self.compute_transcripts(
                                pred_subset, label_subset, transcripts_mask, cell_data
                            ) # shape of total_transcripts x cell_types_tracks
                            for i, trial in enumerate(cell_data.index):
                                # update per track transcripts pearsonr
                                key = f"{trial}/{head_name}/TranscriptsPearsonR"
                                tm_metrics[key].update(
                                    pred_transcripts[:, i].double(),
                                    label_transcripts[:, i].double(),
                                )
                            # update cross_cell transcripts pearsonr
                            key = f"{mod}/{head_name}/cross_cell/TranscriptsPearsonR"
                            tm_metrics[key].update(
                                pred_transcripts.double(),
                                label_transcripts.double(),
                            )
                    else:
                        logging.warning(f"Transcripts mask not found in label, skip calculation for TranscriptsPearsonR")
                    # metrics for each other modality (across cell types)
                    for mod in label_meta['modality'][~label_meta["modality"].str.contains("RNA")].unique():
                        cell_data = label_meta[label_meta["modality"] == mod]
                        pred_subset = pred[head_name][:, :, cell_data["dim"].tolist(), ...]  # bsz x n_window x cell_types
                        label_subset = label[task_type][:, :, cell_data["label_dim"].tolist(), ...].to(self.device, non_blocking=True)  # bsz x n_window x cell_types
                        key = f"{mod}/{head_name}/cross_cell/PearsonR"
                        tm_metrics[key].update(
                            pred_subset.reshape(B * L, -1).double(),
                            label_subset.reshape(B * L, -1).double(),
                        )

        # get and write the metric logs
        self.metrics.update(tm_metrics.compute())
        # get global loss
        if log_loss:
            total_loss = running_loss_local.clone()
            # aggregate the loss value from all devices
            global_aggregate(total_loss, aggregate="sum", world_size=self.world_size)

            global_avg_loss = total_loss / (len(dataloader) * self.world_size)
            self.metrics.update({f"{log_prefix}/loss": global_avg_loss.cpu().item()})

        if save_pred and save_method == "merge":
            # we have to pre save all the res and later aggregate them
            torch.save(
                {
                    "label": {k: torch.cat(v, dim=0) for k, v in labels.items()},
                    "pred": {k: torch.cat(v, dim=0) for k, v in preds.items()},
                    "index": torch.cat(inds, dim=0),
                },
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
        # Set device: if rank is "cpu" or "cuda:X", use it directly; otherwise treat as device id
        if isinstance(self.local_rank, str):
            self.device = self.local_rank
        elif torch.cuda.is_available():
            self.device = self.local_rank
        else:
            self.device = "cpu"

        # set up model and data, make sure the logic aligned
        self.model_data_align()

        # get the training settings
        self.current_epoch = 0
        self.current_step = 0  # based on update step
        self.best_valid_loss = torch.inf
        self.metrics = {}

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
        ## for deepspeed, we load the chk after we setup all the model, optim, schedu
        if self.training_config.load_checkpoint is not None:
            with timer(f"Loading checkpoint", self.logger, self.rank, self.world_size):
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

        self.model = setup_model(self.config, self.logger)

        # since the batchsize is small, we need to sync batchnorm statistics
        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # set up deepspeed with weight decay protocol
        optimizer_grouped_parameters = create_optimizer_grouped_parameters(
            self.model, self.training_config.get("add_opt_group", False)
        )
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=optimizer_grouped_parameters,
            config=self.training_config.deepspeed_config,
            dist_init_required=True,  # we are using DDP
        )
        self.current_lr = self.scheduler.get_last_lr()[0]

        # set up scheduler step logic
        if "ReduceLROnPlateau" in self.training_config.scheduler:
            self.scheduler_need_monitor = True
            # we should further check if valid set is available
            if not self.training_config.test_only and self.data_func["valid"]["dataset"] is None:
                self.logger.error("The given scheduler requires the valid dataset")
                exit(1)
        else:
            self.scheduler_need_monitor = False

        self.logger.info(f"Model {self.model_config.model_name} loaded successfully.")

    @property
    def inference_model(self):
        """Override to use model_engine for DeepSpeed inference."""
        return self.model_engine

    @property
    def training_model(self):
        """Override to use model_engine.module for DeepSpeed training logging."""
        # Unwrap torch.compile wrapper if present
        model = self.model_engine.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

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
        nan_termination = False
        # for loss tracking
        batch_loss_dict = {}
        batch_count = 0
        epoch_loss_list = []

        dataloader = self.data_func["train"]["data_loader"]

        for i, (seq_embedding, label, ind) in enumerate(dataloader):

            # seq_embedding shape [batch, L, 4]
            # label is now a dictionary {task_type: tensor}
            seq_embedding = seq_embedding.to(self.device, non_blocking=True)
            # pred is now a dictionary {head_name: tensor}
            pred = self.model_engine(
                seq_embedding.permute(0, 2, 1),
                self.training_config.use_head,
                data_parallel_training=True if self.world_size > 1 else False,
            )

            # loss - use new compute_loss function
            total_loss, loss_dict = self.compute_loss(pred, label)
            loss = total_loss / self.training_config.accum_step

            # log loss
            for k, v in loss_dict.items():
                if k not in batch_loss_dict:
                    batch_loss_dict[k] = 0.0
                batch_loss_dict[k] += v
            batch_count += 1
            epoch_loss_list.append(loss.detach().cpu().item() * self.training_config.accum_step)

            self.model_engine.backward(loss)

            # log training status
            ## in the training loop, we only look at the local loss
            should_update = ((i + 1) % self.training_config.accum_step == 0) or (i + 1 == len(dataloader))

            tensorboard_log_every = self.logging_config.get("tensorboard_log_every") or self.logging_config.report_every
            if self.current_step % tensorboard_log_every == 0 and should_update:
                report_loss = {k: v / batch_count for k, v in batch_loss_dict.items()}

                nan_termination = self._log_training_metrics(report_loss, should_exit_on_nan=True)

                batch_loss_dict = {}
                batch_count = 0

            # model optimize
            self.model_engine.step()

            self.current_step = self.model_engine.global_steps
            self.current_lr = self.scheduler.get_last_lr()[0]

            if self.logging_config.diagnose:
                self._diagnose_extra_log(ind)

            if nan_termination:
                exit(1)

        self.current_epoch += 1
        self.metrics.update({f"Train/epoch_avg_loss": np.mean(epoch_loss_list)})


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
        diagnose=myconfig.logging.diagnose,
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

                # set up random seed
                trainer.data_rand_seed.value = trainer.current_epoch
                for split in trainer.data_split:
                    trainer.data_func[split]["data_sampler"].set_epoch(trainer.current_epoch)

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

                    if trainer.scheduler_need_monitor:
                        trainer.scheduler.step(valid_loss)

                # testing the model and get the metrics during the run time
                if trainer.data_func["test"]["data_loader"] is not None:
                    with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                        trainer.infer_step(log_loss=False, log_prefix="Test", save_pred=False)

                    blocking_sync_wait(world_size)

                # write the model
                if trainer.current_epoch % myconfig.logging.save_every == 0:
                    with timer(f"Regular saving checkpoint", logger, rank, world_size):
                        trainer.save_checkpoint()

                # write the metrics
                ## if we use tensorboard, the metrics will be written into tb (initialized when creating the training logger)
                if trainer.should_log:
                    metric_dict = trainer.metrics
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
            trainer.data_rand_seed.value = trainer.current_epoch
            save_pred_res = myconfig.logging.get("save_pred_res", True)
            save_method = myconfig.logging.get("save_method", "split")

            if trainer.data_func["train"]["data_loader"] is not None:
                with timer(f"[Train/Infer] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                    trainer.infer_step(
                        log_loss=False, log_prefix="Train", save_pred=save_pred_res, save_method=save_method
                    )
                blocking_sync_wait(world_size)

            if trainer.data_func["valid"]["data_loader"] is not None:
                with timer(f"[Valid/Infer] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                    trainer.infer_step(
                        log_loss=False, log_prefix="Valid", save_pred=save_pred_res, save_method=save_method
                    )
                blocking_sync_wait(world_size)

            if trainer.data_func["test"]["data_loader"] is not None:
                with timer(f"[Test] [Epoch {trainer.current_epoch}]", logger, rank, world_size):
                    trainer.infer_step(
                        log_loss=False, log_prefix="Test", save_pred=save_pred_res, save_method=save_method
                    )
                blocking_sync_wait(world_size)

            # save the raw preds and write the metrics
            if trainer.should_log:
                aggregate_test_res(trainer, "Train", remove_raw=True)
                aggregate_test_res(trainer, "Valid", remove_raw=True)
                aggregate_test_res(trainer, "Test", remove_raw=True)
                metric_dict = trainer.metrics
                if myconfig.logging.use_tensorboard:
                    for k, v in metric_dict.items():
                        trainer.logger.metric(k, v, step=trainer.current_step, log_also=False)
                else:
                    for k, v in metric_dict.items():
                        trainer.logger.metric(k, v, step=trainer.current_step)
                    with open(f"{myconfig.logging.log_dir}/metrics/epoch_{trainer.current_epoch}.json", "w") as f:
                        json.dump(metric_dict, f, indent=4)

        blocking_sync_wait(world_size)

        if not myconfig.training.deepspeed:
            cleanup(world_size)

    except Exception as e:
        logger.exception(e)
    finally:
        # in case the program doesn't stop as expected
        cleanup(world_size)
