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
from omegaconf import OmegaConf
from data.preprocess import preprocess
from utils.config import load_config
from utils.logging import LOGGER_PREFIX, BaseLogger, setup_logging
from utils.multi_gpu import find_free_port
from utils.trainer import mp_main


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
    logger = BaseLogger(
        name=f"{LOGGER_PREFIX}-Main",
        level=eval(f"logging.{log_level}"),
        log_dir=myconfig.logging.log_dir,
        redirect=myconfig.logging.get("write_log_to_file", False),
        overwrite=myconfig.logging.get("overwrite_log_file", False),
    )

    ## set up working directory
    os.makedirs(myconfig.logging.log_dir, exist_ok=True)
    os.makedirs(myconfig.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(myconfig.logging.res_dir, exist_ok=True)
    os.makedirs(myconfig.data.preprocess.storage_path, exist_ok=True)

    ## set up training devices (predefine the loggers for each deivce)
    world_size = myconfig.training.get("world_size", 1)
    available_devices = max(1, torch.cuda.device_count())
    world_size = min(world_size, available_devices)
    if world_size > 1:
        if myconfig.training.get("gpu_id", None) is not None:
            logger.warning(f"World size is set, gpu_id will be ignored")
        gpu_id = None
        logger.info(f"Using gpu 0 to {world_size - 1} for training")
    else:
        gpu_id = myconfig.training.get("gpu_id", 0)
        gpu_id = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {gpu_id} for training")
    myconfig.training.world_size = world_size
    myconfig.training.gpu_id = gpu_id

    ## set up overall loggings
    setup_logging(
        level=eval(f"logging.{log_level}"),
        log_dir=myconfig.logging.log_dir,
        redirect=myconfig.logging.get("write_log_to_file", False),
        use_tensorboard=myconfig.logging.get("use_tensorboard", False),
        world_size=world_size,
        gpu_id=gpu_id,
    )

    ## save the configs
    logger.debug(myconfig)
    with open(f"{myconfig.logging.log_dir}/overall_setting.yaml", "w") as f:
        OmegaConf.save(myconfig, f)

    # get data split and labels
    try:
        preprocess(**myconfig.data.preprocess)
    except Exception as e:
        logger.error("Please check preprocess setting")
        logger.exception(e)
        exit(1)

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
        mp_main(rank=gpu_id, world_size=world_size, myconfig=myconfig)


if __name__ == "__main__":
    main()
