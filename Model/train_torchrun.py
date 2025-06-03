import logging
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(ROOT)

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
import torch.multiprocessing as mpt
from data.preprocess import preprocess
from omegaconf import OmegaConf
from utils.config import load_config
from utils.logging import LOGGER_PREFIX, BaseLogger, save_tensorboard_run_script, setup_logging
from utils.multi_gpu import find_free_port, cleanup
from utils.trainer import torchrun_main
from torch.distributed.elastic.multiprocessing.errors import record


def get_args():
    parser = argparse.ArgumentParser(description='BICAN Borzoi Training Script')
    # keep first
    parser.add_argument('--config_dir', '-c', default="./Config", type=str, help='Path to the Hydra config directory (contains config.yaml + groups)')
    parser.add_argument('--override_config', '-x', action='append', default=[], 
                        help='Hydra override string(s), e.g. "model=no_flashatten" (change whole config profile), or "-x training=single_gpu -x training.gpu_id=3" (change profile, then change specific parameter)')
    return parser.parse_args()

@record
def main(config_dir, override_config):
    # get world size, gpu id and rank from torchrun
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    # read in config file and setup logging
    myconfig = load_config(config_dir, override_config)

    ## set up logging
    log_level = myconfig.logging.get("log_level", "INFO")
    myconfig.logging.log_level = eval(f"logging.{log_level}") if isinstance(log_level, str) else log_level

    ## set up working directory
    os.makedirs(myconfig.logging.log_dir, exist_ok=True)
    os.makedirs(myconfig.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(myconfig.logging.res_dir, exist_ok=True)
    os.makedirs(myconfig.data.preprocess.storage_path, exist_ok=True)

    ## set up overall loggings
    logger = BaseLogger(
        name=f"{LOGGER_PREFIX}-Main",
        level=myconfig.logging.log_level,
        log_dir=myconfig.logging.log_dir,
        redirect=myconfig.logging.write_log_to_file,
        overwrite=myconfig.logging.overwrite_log_file,
    )
    setup_logging(
        level=myconfig.logging.log_level,
        log_dir=myconfig.logging.log_dir,
        redirect=myconfig.logging.write_log_to_file,
    )

    ## set up training devices (predefine the loggers for each deivce)
    myconfig.training.world_size = world_size

    ## set up random seeds
    myconfig.training.seed = myconfig.training.get("seed", 42)
    random.seed(myconfig.training.seed)
    np.random.seed(myconfig.training.seed)
    torch.manual_seed(myconfig.training.seed)
    torch.cuda.manual_seed_all(myconfig.training.seed)

    ## save the configs
    logger.debug(myconfig)
    with open(f"{myconfig.logging.log_dir}/overall_setting.yaml", "w") as f:
        OmegaConf.save(myconfig, f)

    ## write a tensorboard booting bash script
    if myconfig.logging.use_tensorboard:
        save_tensorboard_run_script(
            log_dir=f"{myconfig.logging.log_dir}/tb", save_dir=f"{myconfig.logging.log_dir}"
        )

    # get data split and labels
    try:
        preprocess(**myconfig.data.preprocess)
    except Exception as e:
        logger.error("Please check preprocess setting")
        logger.exception(e)
        exit(1)

    # set up multigpu training if needed
    logger.info(f"Torchrun with {world_size} GPUs")
    
    mpt.freeze_support()
    try:
        torchrun_main(local_rank=local_rank, rank=rank, world_size=world_size, myconfig=myconfig)
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e


if __name__ == "__main__":
    args = get_args()
    main(args.config_dir, args.override_config)
