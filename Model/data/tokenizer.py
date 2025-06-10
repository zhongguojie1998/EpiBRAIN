# Adapted from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/data.py

from pathlib import Path
from random import random, randrange

import numpy as np
import torch
import torch.nn.functional as F
from pyfaidx import Fasta


# helper functions
def exists(val):
    return val is not None


def cast_list(t):
    return t if isinstance(t, list) else [t]


def coin_flip():
    return random() > 0.5


# genomic function transforms, define the tokens

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    # transform str to integer (ASCII), align with the predefined encoding matrix
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype=np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentation functions
def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


# processing bed files and get the one hot encoding
class FastaInterval:
    def __init__(
        self, *, fasta_file, context_length=None, return_seq_indices=False, shift_augs=None, rc_aug=False
    ):
        """
        fasta_file: reference genome
        context_length: the context window size, serving as one "data point"
        return_seq_indices: return single number encoded sequence (instead of one-hot encoded sequence)
        shift_augs: data augmentation by random shift the window, if None, skip
        rc_aug: data augmentation by randomly return reverse complement strand, if False, skip
        """

        self.fasta_file = fasta_file
        assert Path(fasta_file).exists(), "path to fasta file must exist"

        # self.seqs = Fasta(str(fasta_file))
        self.seqs = None  # Pyfaidx isn't fork-safe. Read and write can conflict in multi-process scenarios (dataloader num_workers>0)
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    def __call__(self, chr_name, start, end, return_augs=False):
        """
        Encode DNA sequence. Do seq shift and strand flip data augumentation. Extend the sequence to reach the context length if necessary. Padding if out of range.

        chr_name: chromosome name
        start: sample start idx
        end: sample end idx
        return_augs: whether to also return the aug metadata (the exact augmentation parameter used)
        """
        # the genome is delay initialized
        if self.seqs is None:
            self.seqs = Fasta(self.fasta_file)

        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = min(max(start + min_shift, 0) - start, 0)
            max_shift = max(min(end + max_shift, chromosome_length) - end, 1)

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift
        else:
            rand_shift = 0

        left_padding = right_padding = 0

        # when given interval in the bed file is shorter than the given context length, extend to the given context length
        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)

        should_rc_aug = self.rc_aug and coin_flip()

        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)

            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)

            return seq

        one_hot = str_to_one_hot(seq)

        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot, None, None

        # returns the shift integer as well as the bool (for whether reverse complement was activated)
        # for this particular genomic sequence

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return one_hot, rand_shift_tensor, rand_aug_bool_tensor
