import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(ROOT)

warnings.filterwarnings("ignore")

import click
from data.preprocess import preprocess
from utils.config import LOGGER_PREFIX, get_logger, load_config, setup_logging

# from utils.trainer import DNASeqModelTrainer

logger = get_logger(f"{LOGGER_PREFIX}-Main")


@click.command()
@click.option(
    "--config_dir",
    "-c",
    default="./Config",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the Hydra config directory (contains config.yaml + groups)",
)
@click.option(
    "--override_config",
    "-x",
    multiple=True,
    help="Hydra override string(s), e.g. 'model=no_flashatten' (change whole config profile), or '-x training=single_gpu -x training.gpu_id=3' (change profile, then change specific parameter)",
)
def main(config_dir, override_config):
    # read in config file
    myconfig = load_config(config_dir, override_config)
    logging_level = myconfig.logging.get("logging_level", "INFO")
    logging_file = myconfig.logging.get("log_file", None)
    overwrite = myconfig.logging.get("overwrite_log_file", True)
    setup_logging(
        level=eval(f"logging.{logging_level}"),
        log_file=logging_file,
        overwrite=overwrite,
        logger_prefix=LOGGER_PREFIX,
    )

    logger.debug(myconfig)

    # get data split and labels
    try:
        preprocess(**myconfig.data.preprocess)
    except Exception as e:
        logger.error("Please check preprocess setting")
        logger.exception(e)
        exit(1)

    # # get the trainer (load the model)
    # mytrainer = DNASeqModelTrainer(myconfig)
    # logger.debug(mytrainer.model)


if __name__ == "__main__":
    main()
