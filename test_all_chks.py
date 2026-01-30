import subprocess
import os

import click
import numpy as np
from tqdm import tqdm


@click.command()
@click.option("-c", "--config", help="Path to the config")
@click.option("-chk", "--chk_base", help="Base path to the checkpoints")
@click.option("--start", type=int, help="Start from this checkpoint")
@click.option("--end", type=int, help="End to this checkpoint")
def main(config, chk_base, start, end):
    all_chks = [
        int(i.split(".")[0].split("_")[-1]) for i in os.listdir(chk_base) if i.split(".")[0].split("_")[-1].isdigit()
    ]
    start = 1 if start is None else start
    end = max(all_chks) if end is None else end
    all_chks = sorted([i for i in all_chks if i in np.arange(start, end + 1)])

    for chk in tqdm(all_chks):
        cmd = ["/gpfs/commons/home/guojiezhong/miniconda3/envs/BICAN/bin/python", "Model/train.py", "-c", config, "-x", f"training.load_checkpoint={chk_base}/chk_epoch_{chk}.pt", 
               "-x", "training.test_only=True", "-x", "logging.exp_name=basel_ganglia_complete_v2_test", 
               "-x", "training=default_nygc"]

        subprocess.run(cmd)


if __name__ == "__main__":
    main()
