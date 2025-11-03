import json
import os
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from utils.logging import LOGGER_PREFIX, LazyLogger

logger = LazyLogger(f"{LOGGER_PREFIX}-Config")


# configs
def input_check(
    value_1,
    value_2,
    message,
    check: Literal["equal", "exclude", "within"] = "equal",
):
    try:
        if check == "equal":
            assert value_1 == value_2
        if check == "exclude":
            assert value_1 != value_2
        if check == "within":
            assert value_1 in value_2

    except AssertionError:
        logger.error(message)
        exit(1)


def load_config(config_dir: str = ".", config_name: str = "default", overrides: tuple[str, ...] = [], skip_validation: bool = False):

    if config_name.endswith((".yaml", ".yml")):
        logger.info(f"Loading complete config directly from {config_name}")
        config = OmegaConf.load(config_name)
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)
        if overrides:
            override_cfg = OmegaConf.from_dotlist(list(overrides))
            config = OmegaConf.merge(config, override_cfg)

    else:
        with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
            config = compose(
                config_name=config_name,
                overrides=list(overrides),
            )
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)

    if skip_validation:
        return config

    # some input check
    input_check(
        config.model.crop_param.bins_to_return,
        config.data.preprocess.n_window,
        "`model.bins_to_return` should be same as `data.preprocess.n_window`",
        check="equal",
    )
    # Check each training head is registered in model output_heads
    for head in config.training.use_head:
        input_check(
            head,
            config.model.output_heads.keys(),
            f"Training head `{head}` in `training.use_head` should be registered in `model.output_heads`",
            check="within",
        )
    if config.training.test_only:
        input_check(
            config.training.get("load_checkpoint", None),
            None,
            "When testing, you must load a checkpoint by specifying in `training.load_checkpoint`",
            check="exclude",
        )
        # Commented out to allow validation-only testing
        # input_check(
        #     "test",
        #     config.data.get("used_dataset", []),
        #     "When testing, you must use test dataset",
        #     check="within",
        # )
    else:
        input_check(
            "train",
            config.data.get("used_dataset", []),
            "When training, you must use train dataset",
            check="within",
        )
    if config.training.finetune:
        input_check(
            config.model.get("finetune_method", None),
            None,
            "When finetuning, you must specify the finetune method",
            check="exclude",
        )

    return config


def write_deepspeed_config(config, save_file):
    training_config = config.training
    deepspeed_setup = {}

    deepspeed_setup["optimizer"] = {
        "type": training_config.optimizer,
        "params": dict(training_config.optimizer_params),
    }
    deepspeed_setup["scheduler"] = {
        "type": training_config.scheduler,
        "params": dict(training_config.scheduler_params),
    }
    deepspeed_setup["train_micro_batch_size_per_gpu"] = training_config.batch_size
    deepspeed_setup["gradient_accumulation_steps"] = training_config.accum_step
    deepspeed_setup["gradient_clipping"] = training_config.clip_grad_norm

    with open(save_file, "w") as f:
        json.dump(deepspeed_setup, f)
