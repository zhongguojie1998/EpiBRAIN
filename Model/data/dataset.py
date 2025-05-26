from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from .toeknizer import FastaInterval


class GenomeIntervalDataset(Dataset):

    def __init__(
        self,
        dataset_type,  # train, valid, test
        storage_path,
        refer_genom,
        context_length=196_608,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()

        self.dataset_type = dataset_type

        df = pd.read_csv(f"{storage_path}/{dataset_type}.bed", separator="\t", has_header=False)
        self.df = df

        self.tokenizer = FastaInterval(
            fasta_file=refer_genom,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )
        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        return self.tokenizer(chr_name, start, end, return_augs=self.return_augs)
