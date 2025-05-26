from model.pytorch_borzoi_model import Borzoi
from .config import get_logger, LOGGER_PREFIX

logger = get_logger(f"{LOGGER_PREFIX}-Trainer")


class DNASeqModelTrainer:
    def __init__(self, config):
        self.model = Borzoi.from_hparams(**config.model)
