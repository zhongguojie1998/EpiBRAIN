import logging
import os
import sys

from torch.utils.tensorboard import SummaryWriter

LOGGER_PREFIX = "BICAN"
LOGGING_MODULE = [
    f"{LOGGER_PREFIX}-{i}"
    for i in [
        "Main",
        "Config",
        "Data Preprocess",
        "Trainer",
        "MultiGPU Setup",
    ]
]


def check_rank(f):
    def wrapper(*args, **kwargs):
        if args[0].world_size == 1 or args[0].rank == 0:
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

    # for setting the format of the logger
    def _format(
        self,
        logger,
        fmt: str = "[%(asctime)s] [%(name)s:%(levelname)s]\n%(message)s\n",
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
    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


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
    ):
        super().__init__(name, level, log_dir, redirect, overwrite)

        self.rank = rank
        self.world_size = world_size

        if use_tensorboard and (self.world_size == 1 or self.rank == 0):
            self.add_tb()

    def set_rank(self, rank: int):
        self.rank = rank

    # add tensorboard writer, check rank for only writing info on rank 0 in DDP mod
    @check_rank
    def add_tb(self):
        os.makedirs(f"{self.log_dir}/tb", exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{self.log_dir}/tb")

    # for infos, only log on rank 0 in DDP mode
    @check_rank
    @add_rank
    def info(self, msg: str):
        self.logger.info(msg)

    @add_rank
    def debug(self, msg: str):
        self.logger.debug(msg)

    @add_rank
    def warning(self, msg: str):
        self.logger.warning(msg)

    @add_rank
    def error(self, msg: str):
        self.logger.error(msg)

    # show training information
    @check_rank
    def metric(self, k: str, v: float, step: int, log_also=True):
        if log_also:
            self.info(f"[Metric] [Step {step}] {k} = {v:.6f}")
        if hasattr(self, "writer"):
            self.writer.add_scalar(k, v, step or 0)


class LazyLogger:
    """A lazy logger that initializes the real logger only when needed."""

    def __init__(self, name):
        self._name = name
        self._real_logger = None
        self._backup_logger = BaseLogger(
            name=f"{LOGGER_PREFIX}-Backup Error Logger",
            level=logging.DEBUG,
            log_dir=None,
        )

    def _ensure_init(self):
        if self._real_logger is None:
            if "LOGGERS" not in globals():
                self._real_logger = self._backup_logger
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
    overwrite: bool = False,
    use_tensorboard: bool = False,
    world_size: int = 1,
    gpu_id: int = None,
):
    global LOGGING_MODULE
    if world_size > 1:
        LOGGING_MODULE += [f"{LOGGER_PREFIX}-Trainer:rank_{i}" for i in range(world_size)]
    else:
        LOGGING_MODULE += [f"{LOGGER_PREFIX}-Trainer:rank_{gpu_id}"]

    global LOGGERS
    LOGGERS = {}
    for name in LOGGING_MODULE:
        if name == "Main":
            LOGGERS[name] = BaseLogger(
                name=name,
                level=level,
                log_dir=log_dir,
                redirect=redirect,
                overwrite=overwrite,
            )
        elif "Trainer" in name and "rank" in name:
            try:
                rank = int(name.split("_")[-1])
            except ValueError:
                rank = 0
            LOGGERS[name] = TrainingLogger(
                name=name,
                level=level,
                log_dir=log_dir,
                redirect=redirect,
                overwrite=False,
                rank=rank,
                world_size=world_size,
                use_tensorboard=use_tensorboard,
            )
        else:
            LOGGERS[name] = BaseLogger(
                name=name,
                level=level,
                log_dir=log_dir,
                redirect=redirect,
                overwrite=False,
            )
