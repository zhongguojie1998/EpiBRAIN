import math

import pandas as pd
import torch
import torch.distributed as dist
from data.tokenizer import FastaInterval
from torch.utils.data import Dataset, Sampler


class GenomeIntervalDataset(Dataset):

    def __init__(
        self,
        dataset_type,  # train, valid, test
        storage_path,
        refer_genom,
        logger,
        context_length=196_608,
        preload_data=True,
        shift_augs=None,
        rc_aug=False,
        return_augs=True,
    ):
        super().__init__()

        self.storage_path = storage_path
        self.logger = logger
        self.context_length = context_length

        # load meta data
        df = pd.read_csv(f"{storage_path}/sequences.bed", sep="\t", header=None)
        df.columns = ["chr", "start", "end", "split"]
        self.df = df[df["split"] == dataset_type].reset_index(drop=True)
        self.label_meta = pd.read_csv(f"{storage_path}/label_meta.csv", index_col=0)

        # load label
        self.preload_data = preload_data
        if preload_data:
            self.label = torch.load(f"{storage_path}/data/{dataset_type}.pt")["label"][self.df.index]

        # get tokenizer
        self.tokenizer = FastaInterval(
            fasta_file=refer_genom,
            context_length=context_length,
            return_seq_indices=False,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )
        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.iloc[ind, [0, 1, 2]]
        chrom, start, end = interval
        if self.preload_data:
            label = self.label[ind]
        else:
            label = torch.load(f"{self.storage_path}/data/{chrom}_{start}_{end}.pt")["label"]

        one_hot, shift, reverse = self.tokenizer(chrom, start, end, return_augs=self.return_augs)
        if self.return_augs:
            # here, if reverse, the sequence is still 5'->3', which means the corresponding label should be reversed
            if reverse:
                label = torch.flip(label, dims=[0])

        if one_hot.shape[0] != self.context_length:
            self.logger.error(f"Context length not match (expecting {self.context_length}, {one_hot.shape[0]} observed). Chr {chrom}, start {start}, end {end}, aug shift {shift}")
            exit(1)

        return one_hot, label


def safe_collate_fn(batch):
    one_hots, labels = zip(*batch)
    one_hots = torch.stack([x.clone() for x in one_hots])
    labels = torch.stack([x.clone() for x in labels])
    return one_hots, labels


class DumySampler:

    def __init__(self, **kwargs):
        pass

    def set_epoch(self, epoch):
        pass


class StrictDistributedSampler(Sampler):
    """
    strict allow once and only once appearance of each data point in the training loop
    """

    def __init__(
        self,
        dataset,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires torch.distributed")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires torch.distributed")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            # set the random seed based on the epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        # get the start and end based on the rank
        start = self.rank * self.num_samples
        end = min(start + self.num_samples, n)
        # we do not discard any sample, but it may cause the last card has less sample
        sub_indices = indices[start:end]

        return iter(sub_indices)

    def __len__(self):

        if self.rank == self.num_replicas - 1:
            return len(self.dataset) - self.num_samples * (self.num_replicas - 1)
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
