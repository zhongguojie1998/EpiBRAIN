# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import socket
import torch.distributed as dist

from utils.logging import LOGGER_PREFIX, LazyLogger

logger = LazyLogger(f"{LOGGER_PREFIX}-MultiGPU Setup")

MAXPORT = 12500


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_free_port(orig_port: int) -> int:
    port = orig_port
    while is_port_in_use(port) and port < MAXPORT:
        logger.debug(f"Port {port} in use, trying next port")
        port += 1
    logger.info(f"Port {orig_port} passed, using port {port}")
    return port


def setup(rank, world_size, MASTER_ADDR="localhost", MASTER_PORT=12320):

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
