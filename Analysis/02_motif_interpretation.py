import logging
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, IntegratedGradients
from modisco.visualization import viz_sequence
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from data.data_utils import STD_CHR
from data.tokenizer import FastaInterval, str_to_one_hot
from model.pytorch_borzoi_model import Borzoi
from model.pytorch_borzoi_utils import safe_state_dict_loader
from utils.config import load_config
from utils.logging import BaseLogger


def which_bins(s_idx: int, e_idx: int, window_size: int):

    bin_start = s_idx // window_size
    bin_end = (e_idx - 1) // window_size
    return np.array(range(bin_start, bin_end + 1))


@click.command()
@click.option("--rank", required=True, type=str, default="cuda", help="Device to use")
@click.option("--exp_name", "-e", required=True, type=str)
@click.option("--chk", required=True, type=str)
@click.option("--trial", "-t", required=True, type=str)
@click.option("--peak_file", "-p", required=True, type=str)
@click.option("--config", "-c", required=True, type=str, default="default")
@click.option("--config_base", required=True, type=str, default="./Config")
@click.option("--chk_base", required=True, type=str, default="./Chk")
@click.option("--res_base", required=True, type=str, default="./Res")
def main(rank, exp_name, chk, trial, peak_file, config, config_base, chk_base, res_base):
    CONFIG_DIR = os.path.abspath(config_base)
    CHK_BASE = os.path.abspath(chk_base)
    RES_BASE = os.path.abspath(res_base)

    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp", exist_ok=True)
    os.makedirs(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp", exist_ok=True)

    # read in the config
    myconfig = load_config(CONFIG_DIR, config, overrides=[])
    logger = BaseLogger(name="Interpretation", level=logging.DEBUG)

    # get label
    label_meta = pd.read_csv(f"{myconfig.data.storage_path}/label_meta.csv", index_col=1)
    trial_dim = label_meta.loc[trial, "dim"]
    peak_df = pd.read_csv(peak_file, header=None, sep="\t")
    assert peak_file.split("/")[-1].split(".")[0] + "_" + peak_file.split("/")[-1].split("_")[2] == trial

    # get model
    checkpoint = torch.load(f"{CHK_BASE}/{exp_name}/chk_epoch_{chk}.pt", map_location="cpu")

    model = Borzoi.from_hparams(**myconfig.model)
    if myconfig.training.finetune:
        # load the pretrained state dict
        pretrained_model = Borzoi.from_pretrained(myconfig.model.model_name)
        org_model_state_dict = model.state_dict()
        updated_model_state_dict = safe_state_dict_loader(
            org_model_state_dict=org_model_state_dict,
            load_model_state_dict=pretrained_model.state_dict(),
            logger=logger,
        )
        org_model_state_dict.update(updated_model_state_dict)
        model.load_state_dict(org_model_state_dict)

        # initialize the finetune model
        if myconfig.model.finetune_method == "lora":
            logger.info("LORA Finetune")
            finetune_config = LoraConfig(**myconfig.model.finetune_param)
            model = get_peft_model(model, finetune_config)
        elif myconfig.model.finetune_method == "finetune_layers":
            pass
        else:
            logger.error(f"Finetune method {myconfig.model.finetune_method} is not implemented yet.")
            exit(1)

    model.load_state_dict(checkpoint["model_state_dict"])
    dl_model = DeepLift(model.eval().to(rank), multiply_by_inputs=True)

    # get data
    ## baseline
    np.random.seed(myconfig.training.seed)
    baseline_seq = "".join(
        np.random.choice(["A", "T", "C", "G"], size=myconfig.data.context_length, p=[0.3, 0.3, 0.2, 0.2])
    )
    baseline_seq_onehot = str_to_one_hot(baseline_seq)

    ## test
    dna_tokenizer = FastaInterval(
        fasta_file=os.path.abspath(myconfig.data.refer_genom), context_length=myconfig.data.context_length
    )

    for idx in tqdm(range(len(peak_df))):
        chr_name, start, end = peak_df.iloc[idx, [0, 1, 2]]
        if not chr_name in STD_CHR:
            continue

        token_dict = dna_tokenizer(
                chr_name=chr_name, start=start, end=end, return_augs=False, return_rela_idx=True
            )

        s_idx, e_idx = token_dict["rela_idx"]
        test_seq_onehot = token_dict["one_hot"]
        test_seq_onehot.requires_grad = True

        # get the interested bin, given the raw idx
        # change the bin index to relative index, given the trimmed window
        bin_range = which_bins(s_idx, e_idx, myconfig.data.preprocess.window_size)
        trim = (
            myconfig.data.context_length // myconfig.data.preprocess.window_size
            - myconfig.data.preprocess.n_window
        ) // 2
        bin_range = bin_range - trim

        assert bin_range.min() >= 0
        assert bin_range.max() <= myconfig.data.preprocess.n_window

        if not os.path.exists(f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp/{chr_name}_{start}_{end}.pt"):
            
            all_attribution = []
            for bin in bin_range:
                attribution = dl_model.attribute(
                    inputs=test_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(rank),
                    baselines=baseline_seq_onehot.unsqueeze(0).permute(0, 2, 1).to(rank),
                    target=(trial_dim, bin),
                )
                all_attribution.append(attribution.squeeze(0).detach().cpu())
            all_attribution = torch.stack(all_attribution)

            torch.save(
                all_attribution,
                f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp/{chr_name}_{start}_{end}.pt",
            )
        else:
            all_attribution = torch.load(
                f"{RES_BASE}/{exp_name}/analysis_{chk}/raw_data/interp/{chr_name}_{start}_{end}.pt"
            )

        # plot
        viz_sequence.plot_weights(all_attribution.mean(dim=0)[:, s_idx:e_idx].T, subticks_frequency=20)
        plt.savefig(
            f"{RES_BASE}/{exp_name}/analysis_{chk}/plot/interp/{chr_name}_{start}_{end}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
