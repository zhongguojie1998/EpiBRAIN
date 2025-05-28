import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(ROOT)

warnings.filterwarnings("ignore")

import click
import torch
import torch.multiprocessing as mpt
from data.preprocess import preprocess
from model.pytorch_borzoi_model import Borzoi
from utils.config import load_config
from utils.logging import LOGGER_PREFIX, LazyLogger, setup_logging
from utils.trainer import DNASeqModelTrainer, mp_main
from utils.multi_gpu import find_free_port

logger = LazyLogger(f"{LOGGER_PREFIX}-Main")


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
    # read in config file and setup logging
    myconfig = load_config(config_dir, override_config)
    log_level = myconfig.logging.get("log_level", "INFO")
    log_dir = myconfig.logging.get("log_dir", "./logs/unknown")
    os.makedirs(log_dir, exist_ok=True)

    ## set up training devices (predefine the loggers for each deivce)
    world_size = myconfig.training.get("world_size", 1)
    world_size = world_size if world_size <= torch.cuda.device_count() else torch.cuda.device_count()
    if world_size > 1:
        if myconfig.training.get("gpu_id", None) is not None:
            logger.warning(f"World size is set, gpu_id will be ignored, using gpu 0 to {world_size - 1}")
        gpu_id = None
    else:
        gpu_id = myconfig.training.get("gpu_id", 0)
        gpu_id = gpu_id if torch.cuda.is_available() else "cpu"

    setup_logging(
        level=eval(f"logging.{log_level}"),
        log_dir=log_dir,
        redirect=myconfig.logging.get("write_log_to_file", False),
        overwrite=myconfig.logging.get("overwrite_log_file", False),
        use_tensorboard=myconfig.logging.get("use_tensorboard", False),
        world_size=world_size,
        gpu_id=gpu_id,
    )

    logger.debug(myconfig)

    # get data split and labels
    try:
        preprocess(**myconfig.data.preprocess)
    except Exception as e:
        logger.error("Please check preprocess setting")
        logger.exception(e)
        exit(1)

    # get the model
    model = Borzoi.from_hparams(**myconfig.model)
    # logger.debug(model)

    # get the trainer (load the model)
    mytrainer = DNASeqModelTrainer(myconfig)

    # set up multigpu training if needed
    if world_size > 1:
        logger.info(f"Multi-GPU training with {world_size} GPUs")

        myconfig.training.MASTER_PORT = find_free_port(myconfig.training.get("MASTER_PORT", 12320))
        myconfig.training.MASTER_ADDR = myconfig.training.get("MASTER_ADDR", "localhost")

        mpt.freeze_support()
        try:
            mpt.spawn(mp_main, args=(world_size, myconfig), nprocs=world_size, join=True)
        except Exception as e:
            logger.error(f"Exception: {e}")
            raise e

    else:
        logger.info("Single GPU training")
        if gpu_id == "cpu":
            logger.warning("No GPU specified/available. Training on CPU.")
        mp_main(rank=gpu_id, world_size=world_size, myconfig=myconfig)


if __name__ == "__main__":
    main()
