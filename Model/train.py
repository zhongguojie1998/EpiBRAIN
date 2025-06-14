import logging
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore")

import click
import numpy as np
import torch
import torch.multiprocessing as mpt
from data.preprocess import preprocess
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from utils.config import load_config, write_deepspeed_config
from utils.logging import LOGGER_PREFIX, LazyLogger, save_tensorboard_run_script, setup_logging
from utils.multi_gpu import find_free_port
from utils.trainer import mp_main

logger = LazyLogger(f"{LOGGER_PREFIX}-Main")


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option(
    "--config_setting",
    "-c",
    default="default",
    required=True,
    help="The config setting",
)
@click.option(
    "--config_dir",
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
@click.option(
    "--torchrun",
    "-t",
    is_flag=True,
    help="If we are using torchrun to train the model",
)
@click.option(
    "--deepspeed",
    "-d",
    is_flag=True,
    help="If we are using deepspeed to train the model",
)
@click.option(
    "--only_data",
    is_flag=True,
    help="If we are only processing the data",
)
@record
def main(config_setting, config_dir, override_config, torchrun, deepspeed, only_data):

    if torchrun and deepspeed:
        raise ValueError("Cannot both enbale torchrun and deepspeed")

    # read in config file and setup logging
    myconfig = load_config(config_dir, config_setting, override_config)

    ## set up logging
    log_level = myconfig.logging.get("log_level", "INFO")
    myconfig.logging.log_level = eval(f"logging.{log_level}") if isinstance(log_level, str) else log_level
    myconfig.logging.diagnose = myconfig.logging.get("diagnose", False)

    ## set up working directory
    os.makedirs(myconfig.logging.log_dir, exist_ok=True)
    os.makedirs(myconfig.data.preprocess.storage_path, exist_ok=True)
    if not only_data:
        os.makedirs(myconfig.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(myconfig.logging.res_dir, exist_ok=True)

    ## set up training devices (and logging)
    myconfig.training.torchrun = torchrun
    myconfig.training.deepspeed = deepspeed
    if torchrun or deepspeed:
        ### get world size, gpu id and rank from torchrun (from environment) or deepspeed (from input parameter)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))

        setup_logging(
            level=myconfig.logging.log_level,
            log_dir=myconfig.logging.log_dir,
            redirect=myconfig.logging.write_log_to_file,
            rank=rank,  # when using torchrun, we need to repress other processes from writing log
            world_size=world_size,
            overwrite=myconfig.logging.overwrite_log_file,
        )
    else:
        world_size = myconfig.training.get("world_size", 1)
        available_devices = max(1, torch.cuda.device_count())
        world_size = min(world_size, available_devices)

        setup_logging(
            level=myconfig.logging.log_level,
            log_dir=myconfig.logging.log_dir,
            redirect=myconfig.logging.write_log_to_file,
            rank=0,  # when we are not using torchrun, the main process is regarded as (fake) rank 0, which writes the log
            world_size=world_size,
            overwrite=myconfig.logging.overwrite_log_file,
        )

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
    if deepspeed:
        myconfig.training.deepspeed_config = f"{myconfig.logging.log_dir}/deepspeed_setup.json"
        write_deepspeed_config(myconfig, f"{myconfig.logging.log_dir}/deepspeed_setup.json")

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
    if only_data:
        exit(0)

    # set up multigpu training if needed
    if torchrun or deepspeed:
        logger.info(f"Torchrun with {world_size} GPUs") if torchrun else logger.info(f"Deepspeed with {world_size} GPUs")

        mpt.freeze_support()
        try:
            mp_main(local_rank=local_rank, rank=rank, world_size=world_size, myconfig=myconfig)
        except Exception as e:
            logger.error(f"Exception: {e}")
            logger.exception(e)
            exit(1)
    else:
        if world_size > 1:
            logger.info(f"Multi-GPU training with {world_size} GPUs")

            myconfig.training.MASTER_PORT = find_free_port(myconfig.training.get("MASTER_PORT", 12320))
            myconfig.training.MASTER_ADDR = myconfig.training.get("MASTER_ADDR", "localhost")

            mpt.freeze_support()
            try:
                mpt.spawn(mp_main, args=(world_size, myconfig), nprocs=world_size, join=True)
            except Exception as e:
                logger.error(f"Exception: {e}")
                logger.exception(e)
                exit(1)

        else:
            logger.info("Single GPU training")
            try:
                mp_main(rank=gpu_id, world_size=world_size, myconfig=myconfig)
            except Exception as e:
                logger.error(f"Exception: {e}")
                logger.exception(e)
                exit(1)


if __name__ == "__main__":
    main()
