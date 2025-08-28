import logging
import os
import sys
import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

LOGGER_PREFIX = "BICAN"
LOGGING_MODULE = [
    f"{LOGGER_PREFIX}-{i}"
    for i in [
        "Main",
        "Config",
        "Data Preprocess",
        "Model",
        "MultiGPU Setup",
    ]
]


def check_rank(f):
    def wrapper(*args, **kwargs):
        # for model diagnose, will allow more logs
        force_diagnose = kwargs.get("diagnose", False)
        if args[0].world_size == 1 or args[0].rank == 0 or force_diagnose:
            return f(*args, **kwargs)

    return wrapper


# add rank information to the log message
def add_rank(f):
    def wrapper(obj, msg, *args, **kwargs):
        rank = obj.rank
        msg = f"[rank:{rank}] {msg}"
        return f(obj, msg, *args, **kwargs)

    return wrapper


class BaseLogger:

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_dir: str = "./logs",
        redirect: bool = False,
        overwrite: bool = False,
        rank=0,
        world_size=1,
    ):

        self.level_ = level
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.log_dir = log_dir
        # delete all existing handlers
        self.logger.handlers = []
        if redirect:
            handler = logging.FileHandler(f"{self.log_dir}/logs.log", mode="w" if overwrite else "a")
        else:
            handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

        self._format(self.logger)

        self.rank = rank
        self.world_size = world_size

    # for setting the format of the logger
    def _format(
        self,
        logger,
        fmt: str = "[%(asctime)s] [%(name)s:%(levelname)s] %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ):
        formatter = logging.Formatter(fmt, datefmt)

        for h in logger.handlers:
            h.setFormatter(formatter)
            # we only filter logging info on the logger, not on the handlers
            h.setLevel(logging.DEBUG)

    # set level interface
    @property
    def level(self):
        return self.level_

    @level.setter
    def level(self, level: int):
        self.level_ = level
        self.logger.setLevel(level)

    def set_level(self, level: int):
        self.level = level

    # basic logging methods
    @check_rank
    def info(self, msg: str):
        self.logger.info(msg)

    @check_rank
    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def exception(self, msg: str):
        self.logger.exception(msg)


# logger for DDP training, enable extra tensorboard logging
class TrainingLogger(BaseLogger):

    def __init__(
        self,
        name,
        level=logging.INFO,
        log_dir="./logs",
        redirect: bool = False,
        overwrite=False,
        rank=0,
        world_size=1,
        use_tensorboard=False,
        diagnose=False,
    ):
        super().__init__(name, level, log_dir, redirect, overwrite, rank, world_size)

        if use_tensorboard:
            self.add_tb(diagnose)
        else:
            os.makedirs(f"{self.log_dir}/metrics", exist_ok=True)

    def set_rank(self, rank: int):
        self.rank = rank

    # add tensorboard writer
    @check_rank
    def add_tb(self, diagnose=False):
        os.makedirs(f"{self.log_dir}/tb", exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{self.log_dir}/tb")

    # for infos, only log on rank 0 in DDP mode
    @check_rank
    @add_rank
    def info(self, msg: str, diagnose=False):
        self.logger.info(msg)

    @check_rank
    @add_rank
    def debug(self, msg: str, diagnose=False):
        self.logger.debug(msg)

    @check_rank
    @add_rank
    def warning(self, msg: str, diagnose=False):
        self.logger.warning(msg)

    @add_rank
    def error(self, msg: str):
        self.logger.error(msg)

    @add_rank
    def exception(self, msg: str):
        self.logger.exception(msg)

    # show training information
    @check_rank
    def metric(self, k: str, v: float, step: int, log_also=True, write_hist=False, diagnose=False):
        if log_also:
            self.info(f"[Metric] [Step {step}] {k} = {v:.6f}")
        if hasattr(self, "writer"):
            if not write_hist:
                self.writer.add_scalar(k, v, step or 0)
            else:
                self.writer.add_histogram(k, v, step or 0)


class LazyLogger:
    """A lazy logger that initializes the real logger only when needed."""

    def __init__(self, name):
        self._name = name
        self._real_logger = None
        self._backup_logger = BaseLogger(
            name=f"{LOGGER_PREFIX}-Backup Error Logger",
            level=logging.INFO,
            log_dir=None,
        )

    def _ensure_init(self):
        if self._real_logger is None:
            if "LOGGERS" not in globals():
                self._real_logger = self._backup_logger
            else:
                self._real_logger = LOGGERS.get(self._name, self._backup_logger)

    def __getattr__(self, name):
        self._ensure_init()
        return getattr(self._real_logger, name)

    def __call__(self, *args, **kwargs):
        self._ensure_init()
        return self._real_logger(*args, **kwargs)


logger = LazyLogger(f"{LOGGER_PREFIX}-Logger")


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "./logs",
    redirect: bool = False,
    rank: int = 0,
    world_size: int = 1,
    overwrite: bool = False,
):
    global LOGGING_MODULE
    global LOGGERS
    LOGGERS = {}

    for name in LOGGING_MODULE:
        LOGGERS[name] = BaseLogger(
            name=name,
            level=level,
            log_dir=log_dir,
            redirect=redirect,
            overwrite=False if name != "Main" else overwrite,
            rank=rank,
            world_size=world_size,
        )


class timer:

    def __init__(self, name: str, logg: Optional[BaseLogger] = None, rank: int = 0, world_size: int = 0):
        self.name = name
        self.logg = logg
        self.rank = rank
        self.world_size = world_size

    @check_rank
    def _write(self, msg: str):
        if self.logg is None:
            print(msg)
        else:
            self.logg.info(msg)

    def __enter__(self):
        self._write(f"{self.name} starting...")
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval = self.end - self.start
        msg = f"{self.name} elapsed time: {interval:4f} seconds"

        self._write(msg)
