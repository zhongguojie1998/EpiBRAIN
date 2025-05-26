import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(ROOT)

warnings.filterwarnings("ignore")

import click
# from data.preprocess import preprocess
from utils.config import LOGGER_PREFIX, get_logger, load_config, set_logging_level
# from utils.trainer import DNASeqModelTrainer

logger = get_logger(f"{LOGGER_PREFIX}-Main")


@click.command()
@click.option(
    "--config_dir",
    "-c",
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
    logging_level = myconfig.logging.get("log_mode", "INFO")
    set_logging_level(eval(f"logging.{logging_level}"))

    logger.debug(myconfig)

    # # get data
    # if not os.path.exists(f"{myconfig.data.preprocess.storage_path}/test.pt"):
    #     logger.info("Start preprocess data")
    #     try:
    #         train, valid, test = preprocess(**myconfig.data.preprocess)
    #     except Exception as e:
    #         logger.error("Please check preprocess setting")
    #         logger.exception(e)
    #         exit(1)
    #     logger.info(f"Finish preprocess data\nSave at: {myconfig.data.preprocess.storage_path}")
    # else:
    #     logger.info(f"Read in preprocess data from: {myconfig.data.preprocess.storage_path}")

    # # get the trainer (load the model)
    # mytrainer = DNASeqModelTrainer(myconfig)
    # logger.debug(mytrainer.model)


if __name__ == "__main__":
    main()
