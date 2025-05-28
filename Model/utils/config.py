import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOGGER_PREFIX = "BICAN"

from hydra import compose, initialize_config_dir


# loggers
def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    overwrite: bool = True,
    logger_prefix: str = LOGGER_PREFIX,
):
    for name in logging.Logger.manager.loggerDict:
        if name.startswith(logger_prefix):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler
    if log_file is not None:
        handler = logging.FileHandler(log_file, mode="w" if overwrite else "a")
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    handler.setLevel(level)

    # set to all Loggers
    for name in logging.Logger.manager.loggerDict:
        if name.startswith(logger_prefix):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(handler)
            logger.propagate = False


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
