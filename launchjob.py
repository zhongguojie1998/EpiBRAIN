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
CONDA_BASE = "/home/dl3738/soft/miniforge3"


def validate_machine(ctx, param, value):
    invalid_machine = [m for m in value if m not in MACHINE_IP_MAP]
    if invalid_machine:
        raise click.BadParameter(f"Unknown machine {', '.join(invalid_machine)}")
    return value


def validate_launch_method(ctx, param, value):
    invalid_method = value if value not in ["torchrun", "deepspeed"] else None
    if invalid_method is not None:
        raise click.BadParameter(f"Unknown launch method {invalid_method}")
    return value


@click.command()
@click.option("-e", "--exp_name", type=str, required=True, help="The exp name")
@click.option("-c", "--config", type=str, required=True, default="default", help="The config we want to use")
@click.option(
    "-m", "--machines", multiple=True, required=True, callback=validate_machine, help="The node(s) to use"
)
@click.option(
    "-l",
    "--launch_method",
    type=str,
    required=True,
    callback=validate_launch_method,
    default="torchrun",
    help="The multinode lauch method",
)
@click.option("--load_checkpoint", help="Whether to load the chk and continue training")
@click.option(
    "--override_config",
    "-x",
    multiple=True,
    help="Hydra override string(s), e.g. 'model=no_flashatten' (change whole config profile), or '-x training=single_gpu -x training.gpu_id=3' (change profile, then change specific parameter)",
)
@click.option("--nproc-per-node", default=4, help="GPU available per machine")
@click.option("--master-port", default=1234, help="The master node port")
def launch_distributed(
    exp_name: str,
    config: str,
    machines: List[str],
    launch_method: str,
    load_checkpoint: str,
    override_config: list,
    nproc_per_node: int,
    master_port: int,
):

    if not machines:
        raise click.UsageError("At least need one machine")

    master_addr = MACHINE_IP_MAP[machines[0]]
    nnodes = len(machines)

    with open(f"{WORK_PATH}/hostfile", "w") as f:
        for machine in machines:
            f.write(f"{MACHINE_IP_MAP[machine]} slots={nproc_per_node}\n")

    for rank, machine in enumerate(machines):
        if launch_method == "torchrun":
            cmd = [
                f"{EXECUTABLE_BASE}/bin/torchrun",
                f"--nproc_per_node={nproc_per_node}",
                f"--nnodes={nnodes}",
                f"--node_rank={rank}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                "Model/train.py",
                "-c",
                config,
                "-t",
                "-x",
                f"logging=debug",
                "-x",
                f"logging.exp_name={exp_name}",
            ]
        elif launch_method == "deepspeed":
            cmd = [
                f"{EXECUTABLE_BASE}/bin/deepspeed",
                f"--hostfile={WORK_PATH}/hostfile",
                f"--num_nodes={nnodes}",
                f"--num_gpus={nproc_per_node}",
                f"--node_rank={rank}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                "--no_ssh",
                # "/home/dl3738/work/BICAN/test/15_test_deepspeed.py",
                "Model/train.py",
                "-c",
                config,
                "-d",
                "-x",
                f"logging=debug",
                "-x",
                f"logging.exp_name={exp_name}",
            ]

        if load_checkpoint is not None:
            cmd += [
                "-x",
                f"training.load_checkpoint={load_checkpoint}",
                "-x",
                f"logging.overwrite_log_file=False",
            ]
        
        if override_config:
            for change in override_config:
                cmd = cmd + ["-x"] + [change]

        full_cmd = f"""
        cd {WORK_PATH}

        export PATH={EXECUTABLE_BASE}/bin:$PATH
        export LD_LIBRARY_PATH={EXECUTABLE_BASE}/lib:{EXECUTABLE_BASE}/lib64:$LD_LIBRARY_PATH
        export LIBRARY_PATH={EXECUTABLE_BASE}/lib:$LIBRARY_PATH

        nohup {' '.join(cmd)} > ./logs/backup/backup_{machine}_{exp_name}.log 2>&1 &
        """

        # launch job using ssh
        ssh_cmd = ["ssh", MACHINE_IP_MAP[machine], "bash", "-c", f"'{full_cmd.strip()}'"]
        subprocess.run(ssh_cmd, check=True)


if __name__ == "__main__":
    launch_distributed()
