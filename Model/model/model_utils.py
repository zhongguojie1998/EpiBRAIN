import math

import numpy as np
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from model.model_building_block import Attention, FlashAttention

# model building utils
def compute_channel_sizes(
    filters_init: int,
    filters_end: int = None,
    filters_mult: float = None,
    divisible_by: int = 1,
    depth: int = 1,
) -> list[int]:

    def _round(x: float) -> int:
        return int(np.round(x / divisible_by) * divisible_by)

    if filters_mult is None:
        if filters_end is None:
            raise ValueError("Either filters_end or filters_mult must be provided.")
        if depth < 2:
            filters_mult = 1.0
        else:
            filters_mult = np.exp(np.log(filters_end / filters_init) / (depth - 1))

    sizes = []
    current = float(filters_init)
    for _ in range(depth):
        sizes.append(_round(current))
        current *= filters_mult

    return sizes


def std_pred_head_config(config):
    """
    Expected config format:
    {
        # Shared parameters (optional)
        'use_cell_encoder': bool,           # Whether to use shared celltype encoder (default: False)
        'celltype_hidden_dim': int,         # Hidden dim for celltype embedding (default: in_features//2)

        # Individual head configs
        'head_name': {
            'task': str,                    # Required: 'regression' or 'classification'

            # For use_cell_encoder=True:
            'celltype_num': int,            # Required: number of cell types (must be same across heads)
            'modality_num': int,            # Required: number of modalities (can differ between heads)
            'class_num': int,               # Required for classification: number of classes

            # For use_cell_encoder=False:
            'track_num': int,               # Required: total number of output tracks
            'class_num': int,               # Required for classification: number of classes
        }
    }
    """

    config = config.copy()

    # Step 1: Pop shared parameters
    use_cell_encoder = config.pop("use_cell_encoder", False)
    celltype_hidden_dim = config.pop("celltype_hidden_dim", None)

    if use_cell_encoder and celltype_hidden_dim is None:
        raise ValueError("celltype_hidden_dim not provided")

    # Step 2: Validate and store head configs
    head_configs = {}
    celltype_nums = []

    for head_name, head_config in config.items():
        # validate task
        task = head_config.get("task", None)
        if task not in ["regression", "classification"]:
            raise ValueError(
                f"Head {head_name}: unsupported task '{task}', must be 'regression' or 'classification'"
            )

        # get the required fields
        track_num = head_config.get("track_num")
        celltype_num = head_config.get("celltype_num")
        modality_num = head_config.get("modality_num")
        class_num = head_config.get("class_num")

        std_head_config = {"task": task}

        # Determine track configuration
        if use_cell_encoder:
            if (celltype_num is None) or (modality_num is None):
                raise ValueError(
                    f"Head {head_name}: celltype_num/modality_num required when use_cell_encoder=True"
                )
            std_head_config.update(
                {
                    "celltype_num": celltype_num,
                    "modality_num": modality_num,
                    "track_num": celltype_num * modality_num,
                }
            )
            celltype_nums.append(celltype_num)
        else:
            if track_num is not None:
                if celltype_num is not None or modality_num is not None:
                    print(
                        f"Head {head_name}: track_num provided along with celltype_num/modality_num. Using track_num only."
                    )
                std_head_config.update(
                    {
                        "track_num": track_num,
                        "celltype_num": None,
                        "modality_num": None,
                    }
                )
            else:
                raise ValueError(f"Head {head_name}: Must provide track_num when use_cell_encoder=False")

        std_head_config["class_num"] = class_num

        # Calculate output channels
        if task == "classification":
            if class_num is None:
                raise ValueError(f"Head {head_name}: class_num required for classification task")
            std_head_config["out_channels"] = std_head_config["track_num"] * class_num
        elif task == "regression":
            std_head_config["out_channels"] = std_head_config["track_num"]
        else:
            raise ValueError(
                f"Head {head_name}: unsupported task '{task}', must be 'regression' or 'classification'"
            )

        head_configs[head_name] = std_head_config

    # Step 3: Validate celltype_num consistency if using cell encoder
    if use_cell_encoder:
        if len(set(celltype_nums)) != 1:
            raise ValueError(
                f"All heads must have same celltype_num when use_cell_encoder=True. Got: {set(celltype_nums)}"
            )

    return head_configs, use_cell_encoder, celltype_hidden_dim


# model init utils
def lecun_normal_init(layer):
    if isinstance(layer, nn.Conv1d):
        fan_in = layer.in_channels * layer.kernel_size[0]
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(layer.weight, mean=0, std=std)
    elif isinstance(layer, nn.Linear):
        fan_in = layer.in_features
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(layer.weight, mean=0, std=std)
    # bias are set to 0 for all layers in TF
    if isinstance(layer, (nn.Linear, nn.Conv1d)) and layer.bias is not None:
        layer.bias.data.zero_()


def conditional_recursive_he_normal_init(layer):
    # --- Stop Condition ---
    # If the current module is one of the types we want to skip,
    # we stop the recursion for this entire branch.
    if isinstance(layer, (Attention, FlashAttention)):
        return
    # --- Initialization Logic for the current module ---
    # Apply the appropriate initialization based on the module type.
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if layer.bias is not None:
            layer.bias.data.zero_()
    # --- Recursive  ---
    for child in layer.children():
        conditional_recursive_he_normal_init(child)


def other_init(layer):
    # same as in TF
    if isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d)):
        layer.bias.data.zero_()
        layer.weight.data.fill_(1.0)


# model setting up utils
def count_grad_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_param_grad(model, trainable_params=[]):

    for name, param in model.named_parameters():
        param.requires_grad = False
        for k in trainable_params:
            if k in name:
                param.requires_grad = True
                break

    return model


def safe_state_dict_loader(org_model_state_dict, load_model_state_dict, partial_load, logger):
    filtered_dict = {}

    for k, v in load_model_state_dict.items():
        if k in org_model_state_dict and v.size() == org_model_state_dict[k].size():
            loaded = False
            for n in partial_load:
                if n in k:
                    filtered_dict[k] = v
                    logger.debug(f"Parameter loaded as specified: {k}.")
                    loaded = True
                    break
            if not loaded:
                logger.debug(f"Parameter not loaded as specified: {k}.")
        else:
            logger.warning(
                f"Parameter not loaded: {k}. This is expected if you have modified the model but not expected if you want to get the exact same model."
            )

    return filtered_dict


def setup_model(config, logger):
    from model.model import Borzoi  # Import moved here to avoid circular import

    model_config = config.model
    training_config = config.training

    # get our model skeleton
    if "borzoi" in model_config.model_name or "flashzoi" in model_config.model_name:
        model_cls = Borzoi
    else:
        logger.error(f"Model {model_config.model_name} is not implemented yet.")
        exit(1)

    model = model_cls.from_hparams(**model_config)

    # initialize the model
    if training_config.load_checkpoint is None:
        model.init_weights()

    # if finetune, load the pretrained model
    if training_config.finetune:
        # in case we need to load the pretrained model and only want to load part of them
        partial_load = model_config.get("partial_load", None)
        partial_load = list(model.state_dict().keys()) if partial_load is None else partial_load

        # load the pretrained state dict
        pretrained_model = model_cls.from_pretrained(model_config.model_name)
        org_model_state_dict = model.state_dict()

        updated_model_state_dict = safe_state_dict_loader(
            org_model_state_dict=org_model_state_dict,
            load_model_state_dict=pretrained_model.state_dict(),
            partial_load=partial_load,
            logger=logger,
        )
        org_model_state_dict.update(updated_model_state_dict)
        model.load_state_dict(org_model_state_dict)

        # initialize the finetune model
        if model_config.finetune_method == "lora":
            logger.info("LORA Finetune")
            finetune_config = LoraConfig(**model_config.finetune_param)
            model = get_peft_model(model, finetune_config)
        elif model_config.finetune_method == "finetune_layers":
            logger.info("Finetune the given layers")
            model = set_param_grad(model, **model_config.finetune_param)
        elif model_config.finetune_method == "cont_from_pretrain":
            logger.info("Load from pretrained model and continue train")
            pass
        else:
            logger.error(f"Finetune method {model_config.finetune_method} is not implemented yet.")
            exit(1)

    # get all the trainable parameters
    trainable_params = [n for n, m in model.named_parameters() if m.requires_grad]
    for n in trainable_params:
        logger.debug(f"Trainable module name: {n}")

    trainable_para_num = count_grad_parameters(model)
    total_para_num = count_all_parameters(model)
    logger.info(
        f"Trainable params: {trainable_para_num} || All params: {total_para_num} || Trainable%: {trainable_para_num / total_para_num:.4f}"
    )

    logger.debug(model)

    return model
