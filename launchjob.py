import os
import subprocess
from pathlib import Path
from typing import List

import click

MACHINE_IP_MAP = {
    "euler": "192.168.255.9",
    "turing": "192.168.255.8",
    "neumann": "192.168.255.7",
    "shannon": "192.168.255.12",
}
WORK_PATH = Path(".").absolute()
EXECUTABLE_BASE = os.environ["CONDA_PREFIX"]


def validate_machine(ctx, param, value):
    invalid_machine = [m for m in value if m not in MACHINE_IP_MAP]
    if invalid_machine:
        raise click.BadParameter(f"Unknown machine {', '.join(invalid_machine)}")
    return value


@click.command()
@click.option("-e", "--exp_name", type=str, required=True, help="The exp name")
@click.option(
    "-m", "--machine", multiple=True, required=True, callback=validate_machine, help="The node(s) to use"
)
@click.option("-c", "--load_checkpoint", help="Whether to load the chk and continue training")
@click.option("--nproc-per-node", default=4, help="GPU available per machine")
@click.option("--master-port", default=1234, help="The master node port")
def launch_distributed(exp_name: str, machine: List[str], load_checkpoint: str, nproc_per_node: int, master_port: int):

    if not machine:
        raise click.UsageError("At least need one machine")

    master_addr = MACHINE_IP_MAP[machine[0]]
    nnodes = len(machine)

    for rank, machine in enumerate(machine):

        cmd = [
            f"{EXECUTABLE_BASE}/bin/torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--nnodes={nnodes}",
            f"--node_rank={rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "Model/train.py",
            "-t",
            "-x",
            f"logging=debug",
            "-x",
            f"logging.exp_name={exp_name}",

        ]

        if load_checkpoint is not None:
            cmd += [
                "-x",
                f"training.load_checkpoint=./Chk/{exp_name}/chk_epoch_{load_checkpoint}.pt",
                "-x",
                f"logging.overwrite_log_file=False",
            ]

        full_cmd = f"""
        cd {WORK_PATH}
        nohup {' '.join(cmd)} > ./logs/backup/backup_{machine}_{exp_name}.log 2>&1 &
        """

        # launch job using ssh
        ssh_cmd = ["ssh", MACHINE_IP_MAP[machine], "bash", "-c", f'"{full_cmd.strip()}"']
        subprocess.run(ssh_cmd, check=True)


if __name__ == "__main__":
    launch_distributed()
