import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s\n%(message)s\n", stream=sys.stdout)
LOGGER_PREFIX = "BICAN"

from hydra import compose, initialize_config_dir


# loggers
def set_logging_level(level, logger_prefix=LOGGER_PREFIX):
    logger_dict = logging.Logger.manager.loggerDict

    for name in logger_dict:
        if name.startswith(logger_prefix):
            logger = logging.getLogger(name)
            logger.setLevel(level)


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger(f"{LOGGER_PREFIX}-Config")


# configs
def input_check(value_1, value_2, message):
    try:
        assert value_1 == value_2
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
    )
    input_check(config.data.preprocess.window_size, 128, "Enformer model only support 128 bp resolution")

    return config
