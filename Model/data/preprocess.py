import os

import click
import pyBigWig
from utils.config import LOGGER_PREFIX, get_logger

logger = get_logger(f"{LOGGER_PREFIX}-Data preprocess")


def preprocess(storage_path, context_length, window_size, n_window, refer_genom, trial_path, return_data=True):
    os.makedirs(storage_path, exist_ok=True)

    logger.info(
        f"Preprocess config:\ncontext length: {context_length}\nwindow size: {window_size}\nkept central window num: {n_window}"
    )


    if return_data:
        return context_length, window_size, n_window


@click.command()
@click.option(
    "-s",
    "--storage_path",
    required=True,
    type=click.Path(writable=True, file_okay=False),
    help="The storage dir for precomputed data",
)
@click.option(
    "--context_length",
    required=True,
    default=196_608,
    type=int,
    help="The length of the DNA serving as one sample",
)
@click.option(
    "--window_size",
    required=True,
    default=128,
    type=int,
    help="The window size is restricted by the model architecture. For enformer, the number is 128",
)
@click.option(
    "--n_window",
    required=True,
    default=896,
    type=int,
    help="The central bin number to keep and calculate the label. For enformer, the number is 896 (196,608 / 128 (bins) - 320 (bins per side) * 2)",
)
@click.option(
    "--refer_genom",
    required=True,
    type=click.Path(dir_okay=False),
    help="The path to reference genome",
)
@click.option(
    "--trial_path",
    required=True,
    type=click.Path(writable=True, file_okay=True, dir_okay=True),
    help="The path to get the trials to serve as the label, only support bigwig files",
)
def command_line_preprocess(storage_path, context_length, window_size, n_window, refer_genom, trial_path):
    preprocess(storage_path, context_length, window_size, n_window, refer_genom, trial_path, return_data=False)


if __name__ == "__main__":
    command_line_preprocess()
