import os
from typing import Literal

from hydra import compose, initialize_config_dir
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


def load_config(config_dir: str, overrides: tuple[str, ...]):

    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
        config = compose(
            config_name="config",
            overrides=list(overrides),
        )

    input_check(
        config.model.bins_to_return,
        config.data.preprocess.n_window,
        "`model.bins_to_return` should be same as `data.preprocess.n_window`",
        check="equal",
    )
    input_check(
        config.model.use_head,
        config.model.output_heads.keys(),
        "Predict head (`model.use_head`) should be registered (`model.output_heads`)",
        check="within",
    )
    if config.model.model_name == "borzoi":
        input_check(
            config.data.preprocess.window_size, 128, "Enformer model only support 128 bp resolution", check="equal"
        )
    if config.training.test_only:
        input_check(
            config.training.load_checkpoint,
            None,
            "When testing, you must load a checkpoint by specifying in `training.load_checkpoint`",
            check="exclude",
        )

    return config
