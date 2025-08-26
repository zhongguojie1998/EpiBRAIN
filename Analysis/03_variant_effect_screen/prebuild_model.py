import logging
import os
import pickle
import sys
from pathlib import Path

import click
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from data.tokenizer import FastaInterval
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger
from model.model_building_block import get_positional_embed

class ModelPackage:
    def __init__(self, model, dna_tokenizer, config):
        self.model = model
        self.dna_tokenizer = dna_tokenizer
        self.config = config

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def fix_meta_buffers_with_original_data(model, original_model=None):

    fixed_count = 0

    for name, buffer in model.named_buffers():
        if buffer.is_meta and "positions" in name:
            print(f"Fixing meta buffer: {name}")

            new_positions = None
            if original_model is not None:
                try:
                    for orig_name, orig_buffer in original_model.named_buffers():
                        if orig_name == name and not orig_buffer.is_meta:
                            new_positions = orig_buffer.detach().clone().cpu()
                            print(f"  Copied from original model: {orig_buffer.shape}")
                            break
                except:
                    pass

            if new_positions is None:
                seq_len = buffer.shape[0]  
                features = buffer.shape[1]  
                new_positions = get_positional_embed(seq_len, features, torch.device("cpu"))
                print(f"  Recreated: {new_positions.shape}")

            module_parts = name.split(".")
            parent_module = model
            for part in module_parts[:-1]: 
                if part.isdigit():
                    parent_module = parent_module[int(part)]
                else:
                    parent_module = getattr(parent_module, part)

            parent_module.register_buffer("positions", new_positions, persistent=False)
            fixed_count += 1

    if fixed_count > 0:
        print(f"Fixed {fixed_count} meta buffers (positions)")

    return model


@click.command()
@click.option("--config", "-c", required=True, help="Path to config file")
@click.option("--checkpoint", "-chk", help="Path to checkpoint file")
@click.option("--output", "-o", required=True, help="Output path for packaged model")
@click.option("--use_borzoi", is_flag=True, help="If set, use the borzoi implementation from borzoi-pytorch")
def main(config, checkpoint, output, use_borzoi):
    myconfig = load_config(config_name=config)
    logger = BaseLogger(name="Model packaging", level=logging.INFO)

    if not use_borzoi:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model = setup_model(myconfig, logger=logger)
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        from model.borzoi_pytorch.pytorch_borzoi_model import Borzoi
        model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
        model = fix_meta_buffers_with_original_data(model)
    model.eval()

    dna_tokenizer = FastaInterval(
        fasta_file=myconfig.data.refer_genom, context_length=myconfig.data.context_length
    )

    package = ModelPackage(model, dna_tokenizer, myconfig)

    with open(output, "wb") as f:
        pickle.dump(package, f)

    print(f"Model packaged to {output}")


if __name__ == "__main__":
    main()
