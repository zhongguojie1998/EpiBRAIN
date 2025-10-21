import math
import multiprocessing as mp

import pandas as pd
import torch
import torch.distributed as dist
from data.tokenizer import FastaInterval
from torch.utils.data import Dataset, Sampler, get_worker_info


class GenomeIntervalDataset(Dataset):

    def __init__(
        self,
        dataset_type,  # train, valid, test
        storage_path,
        refer_genom,
        context_length=196_608,
        preload_data=True,
        shift_augs=None,
        rc_aug=False,
        return_augs=True,
        return_token_dict=False,
        external_rand_seed=mp.Value("i", 0),
        load_task=set("regression"),
        **kwargs,
    ):
        super().__init__()

        self.storage_path = storage_path
        self.context_length = context_length

        assert return_augs if rc_aug else True, "If you want to use reverse complement augmentation, you must also return the augmentation information"

        # load meta data
        df = pd.read_csv(f"{storage_path}/sequences.bed", sep="\t", header=None)
        df.columns = ["chr", "start", "end", "split"]
        self.df = df[df["split"] == dataset_type].reset_index(drop=True)
        self.load_task = list(load_task)
        raw_label_meta = pd.read_csv(f"{storage_path}/raw_label_meta.csv")
        self.raw_label_meta = raw_label_meta
        
        # get the RNA_plus and RNA_minus index, reverse for each cell type
        self.reverse_complement_aug_index = None
        if "RNAplus" in self.raw_label_meta['modality'].values and "RNAminus" in self.raw_label_meta['modality'].values:
            # needs to reverse the RNAplus and RNAminus
            self.reverse_complement_aug_index = []
            for i, row in self.raw_label_meta.iterrows():
                if row['modality'] == 'RNAplus':
                    # find the index of RNAminus of corresponding cell type
                    minus_idx = self.raw_label_meta.index[(self.raw_label_meta['modality'] == 'RNAminus') & (self.raw_label_meta['cell_type'] == row['cell_type'])].values[0]
                    self.reverse_complement_aug_index.append(int(minus_idx))
                elif row['modality'] == 'RNAminus':
                    # find the index of RNAplus of corresponding cell type
                    plus_idx = self.raw_label_meta.index[(self.raw_label_meta['modality'] == 'RNAplus') & (self.raw_label_meta['cell_type'] == row['cell_type'])].values[0]
                    self.reverse_complement_aug_index.append(int(plus_idx))
                else:
                    # don't change
                    self.reverse_complement_aug_index.append(i)
        
        # load label
        self.preload_data = preload_data
        if preload_data:
            all_label = torch.load(f"{storage_path}/data/{dataset_type}.pt")["label"]
            self.all_label = {k: v[self.df.index] for k, v in all_label.items()}

        # get tokenizer
        self.tokenizer = FastaInterval(
            fasta_file=refer_genom,
            context_length=context_length,
            return_seq_indices=False,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )
        self.return_augs = return_augs
        self.return_token_dict = return_token_dict

        self.external_rand_seed = external_rand_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.iloc[ind, [0, 1, 2]]
        chrom, start, end = interval

        token_dict = self.tokenizer(
            chrom, start, end, return_augs=self.return_augs, seed=self.external_rand_seed.value + ind
        )
        one_hot = token_dict["one_hot"]
        if one_hot.shape[0] != self.context_length:
            raise ValueError(
                f"Context length not match (expecting {self.context_length}, {one_hot.shape[0]} observed). Chr {chrom}, start {start}, end {end}, aug shift {token_dict.get('rand_shift')}"
            )

        if self.preload_data:
            label = {k: v[ind] for k, v in all_label.items() if k in self.load_task}
        else:
            all_label = torch.load(f"{self.storage_path}/data/{chrom}_{start}_{end}.pt")["label"]
            label = {k: v for k, v in all_label.items() if k in self.load_task}
            # load transcripts if needed
            if "transcripts" in self.load_task:
                transcript_data = torch.load(f"{self.storage_path}/data/{chrom}_{start}_{end}_transcripts.pt")["transcripts_mask"]
                label["transcripts_mask"] = transcript_data

        if self.return_augs:
            # here, if reverse, the sequence is still 5'->3', which means the corresponding label should be reversed
            # if RNA_minus and RNA_plus are in the label, they should be swapped
            if token_dict["rand_reverse"]:
                label = {k: torch.flip(v, dims=[0]) for k, v in label.items()}
                if self.reverse_complement_aug_index:
                    # reverse the RNA_plus and RNA_minus index
                    label['regression'] = label['regression'][:, self.reverse_complement_aug_index]

        return one_hot if not self.return_token_dict else token_dict, label, ind


def collate_fn(batch):
    """
    Custom collate function to handle dictionary labels.
    """
    sequences, labels, indices = zip(*batch)
    
    # Stack sequences and indices
    sequences = torch.stack(sequences)
    indices = torch.tensor(indices)
    
    # Handle dictionary labels
    if isinstance(labels[0], dict):
        # Get all task keys from the first sample
        task_keys = labels[0].keys()
        batched_labels = {}
        
        for task_key in task_keys:
            if task_key == "transcripts_mask":
                # 1 x n_window x n_transcripts
                # Transform to n_transcripts_total x bsz x n_window
                transformed_labels = []
                for idx, label in enumerate(labels):
                    transcripts_mask = label[task_key]
                    # ensure 3D: 1 x n_window x n_transcripts
                    if transcripts_mask.dim() == 2:
                        transcripts_mask = transcripts_mask.unsqueeze(0)
                    # transpose to n_transcripts x 1 x n_window
                    transcripts_mask = transcripts_mask.permute(2, 0, 1)
                    # expand dim 1 to batch size
                    transcripts_mask = transcripts_mask.expand(-1, len(batch), -1)
                    # create mask to zero out all batch dims except idx-th
                    batch_mask = torch.zeros(len(batch), dtype=torch.bool)
                    batch_mask[idx] = True
                    transcripts_mask = transcripts_mask * batch_mask.view(1, -1, 1)
                    transformed_labels.append(transcripts_mask)
                # concatenate all transcripts across the batch
                batched_labels[task_key] = torch.cat(transformed_labels, dim=0)
            else:
                # Stack labels for this task across the batch
                task_labels = [label[task_key] for label in labels]
                batched_labels[task_key] = torch.stack(task_labels)
        
        return sequences, batched_labels, indices
    else:
        # Fallback for non-dictionary labels
        return sequences, torch.stack(labels), indices


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
