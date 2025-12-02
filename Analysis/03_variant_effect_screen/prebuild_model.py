import logging
import os
import pickle
import sys
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "Model"))
os.chdir(ROOT)

from data.tokenizer import FastaInterval
from model.model_utils import setup_model
from utils.config import load_config
from utils.logging import BaseLogger


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


@click.command()
@click.option("--config", "-c", required=True, help="Path to config file")
@click.option("--checkpoint", "-chk", required=True, help="Path to checkpoint file")
@click.option("--output", "-o", required=True, help="Output path for packaged model")
def main(config, checkpoint, output):
    # Skip validation since we're just packaging the model, not training/testing
    myconfig = load_config(config_name=config, skip_validation=True)
    logger = BaseLogger(name="Model packaging", level=logging.INFO)

    # Check if compilation is enabled
    use_compile = myconfig.model.get("use_compile", False)

    # Prevent setup_model from loading checkpoint automatically
    myconfig.training.load_checkpoint = None
    # Disable compilation during model setup
    myconfig.model.use_compile = False

    # Setup model - it will initialize without loading checkpoint
    model = setup_model(myconfig, logger=logger)

    # Manually load the checkpoint
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    # Now compile if it was enabled
    if use_compile:
        compile_mode = myconfig.model.get("compile_mode", "default")
        compile_backend = myconfig.model.get("compile_backend", "inductor")
        compile_fullgraph = myconfig.model.get("compile_fullgraph", False)
        logger.info(f"Compiling model with torch.compile (mode={compile_mode}, backend={compile_backend}, fullgraph={compile_fullgraph})")
        try:
            model = torch.compile(model, mode=compile_mode, backend=compile_backend, fullgraph=compile_fullgraph)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Proceeding with uncompiled model.")

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
