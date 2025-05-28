from utils.logging import LOGGER_PREFIX, LazyLogger
from utils.multi_gpu import cleanup, setup

logger = LazyLogger(f"{LOGGER_PREFIX}-Trainer")


class DNASeqModelTrainer:
    def __init__(self, config):
        if config.training.get("world_size", 1) > 1:
            for i in range(config.training.get("world_size", 1)):
                logger.debug(f"Initializing trainer for rank {i}")
                mp_main(i, config.training.get("world_size", 1))
        else:
            logger.debug("Initializing single GPU trainer")
            mp_main(config.training.get("gpu_id", "cpu"), 1)


def mp_main(rank, world_size, myconfig):
    # special train loggers which can also log to tensorboard
    logger = LazyLogger(f"{LOGGER_PREFIX}-Trainer:rank_{rank}")

    setup(rank, world_size, myconfig.training.MASTER_ADDR, myconfig.training.MASTER_PORT)

    cleanup()
