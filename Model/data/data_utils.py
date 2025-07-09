# Adapted from https://github.com/calico/basenji/blob/master/bin/basenji_data.py, https://github.com/calico/basenji/blob/master/bin/basenji_data_read.py

import collections
import heapq
import math
import os
import subprocess
import tempfile

import h5py
import intervaltree
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import torch
from utils.logging import LOGGER_PREFIX, LazyLogger

logger = LazyLogger(f"{LOGGER_PREFIX}-Data Preprocess")

Contig = collections.namedtuple("Contig", ["chr", "start", "end"])
ModelSeq = collections.namedtuple("ModelSeq", ["chr", "start", "end", "label"])

STD_CHR = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def load_chromosomes(genome_file):
    """Load genome segments from either a FASTA file or
    chromosome length table."""

    # is genome_file FASTA or (chrom,start,end) table?
    file_fasta = open(genome_file).readline()[0] == ">"

    chrom_segments = {}

    if file_fasta:
        fasta_open = pysam.Fastafile(genome_file)
        for i in range(len(fasta_open.references)):
            chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
        fasta_open.close()

    else:
        for line in open(genome_file):
            a = line.split()
            chrom_segments[a[0]] = [(0, int(a[1]))]

    return chrom_segments


def split_contigs(chrom_segments, gaps_file):
    """Split the assembly up into contigs defined by the gaps.

    Args:
      chrom_segments: dict mapping chromosome names to lists of (start,end)
      gaps_file: file specifying assembly gaps

    Returns:
      chrom_segments: same, with segments broken by the assembly gaps.
    """

    chrom_events = {}

    # add known segments
    for chrom in chrom_segments:
        if len(chrom_segments[chrom]) > 1:
            logger.error("I've made a terrible mistake...regarding the length of chrom_segments[%s]" % chrom)
            exit(1)
        cstart, cend = chrom_segments[chrom][0]
        chrom_events.setdefault(chrom, []).append((cstart, "Cstart"))
        chrom_events[chrom].append((cend, "cend"))

    # add gaps
    for line in open(gaps_file):
        a = line.split()
        chrom = a[0]
        gstart = int(a[1])
        gend = int(a[2])

        # consider only if its in our genome
        if chrom in chrom_events:
            chrom_events[chrom].append((gstart, "gstart"))
            chrom_events[chrom].append((gend, "Gend"))

    for chrom in chrom_events:
        # sort
        chrom_events[chrom].sort()

        # read out segments
        chrom_segments[chrom] = []
        for i in range(len(chrom_events[chrom]) - 1):
            pos1, event1 = chrom_events[chrom][i]
            pos2, event2 = chrom_events[chrom][i + 1]

            event1 = event1.lower()
            event2 = event2.lower()

            shipit = False
            if event1 == "cstart" and event2 == "cend":
                shipit = True
            elif event1 == "cstart" and event2 == "gstart":
                shipit = True
            elif event1 == "gend" and event2 == "gstart":
                shipit = True
            elif event1 == "gend" and event2 == "cend":
                shipit = True
            elif event1 == "gstart" and event2 == "gend":
                pass
            else:
                logger.error("I'm confused by this event ordering: %s - %s" % (event1, event2))
                exit(1)

            if shipit and pos1 < pos2:
                chrom_segments[chrom].append((pos1, pos2))

    return chrom_segments


def break_large_contigs(contigs, break_t, verbose=False):
    """Break large contigs in half until all contigs are under
    the size threshold."""

    # initialize a heapq of contigs and lengths
    contig_heapq = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heapq, (-ctg_len, ctg))

    ctg_len = break_t + 1
    while ctg_len > break_t:

        # pop largest contig
        ctg_nlen, ctg = heapq.heappop(contig_heapq)
        ctg_len = -ctg_nlen

        # if too large
        if ctg_len > break_t:
            if verbose:
                logger.info("Breaking %s:%d-%d (%d nt)" % (ctg.chr, ctg.start, ctg.end, ctg_len))

            # break in two
            ctg_mid = ctg.start + ctg_len // 2

            try:
                ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
            except AttributeError:
                ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

            # add left
            ctg_left_len = ctg_left.end - ctg_left.start
            heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

            # add right
            ctg_right_len = ctg_right.end - ctg_right.start
            heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

    # return to list
    contigs = [len_ctg[1] for len_ctg in contig_heapq]

    return contigs


def contig_sequences(contigs, seq_length, stride, snap=1, label=None):
    """Break up a list of Contig's into a list of ModelSeq's."""
    mseqs = []
    for ctg in contigs:
        seq_start = int(np.ceil(ctg.start / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end <= ctg.end:
            # record sequence
            mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))

            # update
            seq_start += stride
            seq_end += stride

    return mseqs


def rejoin_large_contigs(contigs):
    """Rejoin large contigs that were broken up before alignment comparison."""

    # split list by chromosome
    chr_contigs = {}
    for ctg in contigs:
        chr_contigs.setdefault(ctg.chr, []).append(ctg)

    contigs = []
    for chrm in chr_contigs:
        # sort within chromosome
        chr_contigs[chrm].sort(key=lambda x: x.start)

        ctg_ongoing = chr_contigs[chrm][0]
        for i in range(1, len(chr_contigs[chrm])):
            ctg_this = chr_contigs[chrm][i]
            if ctg_ongoing.end == ctg_this.start:
                # join
                # ctg_ongoing.end = ctg_this.end
                ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
            else:
                # conclude ongoing
                contigs.append(ctg_ongoing)

                # move to next
                ctg_ongoing = ctg_this

        # conclude final
        contigs.append(ctg_ongoing)

    return contigs


def divide_contigs_folds(contigs, folds, seed=42):
    """Divide list of contigs into cross fold lists."""

    np.random.seed(seed)
    logger.info("Divide contigs into %d folds" % folds)

    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed fold nucleotides
    fold_nt_aim = int(np.ceil(total_nt / folds))

    # initialize current fold nucleotides
    fold_nt = np.zeros(folds)

    # initialize fold contig lists
    fold_contigs = []
    for fi in range(folds):
        fold_contigs.append([])

    # process contigs
    for ctg_len, ctg in length_contigs:

        # compute gap between current and aim
        fold_nt_gap = fold_nt_aim - fold_nt
        fold_nt_gap = np.clip(fold_nt_gap, 0, np.inf)

        # compute sample probability
        fold_prob = fold_nt_gap / fold_nt_gap.sum()

        # sample train/valid/test
        fi = np.random.choice(folds, p=fold_prob)
        fold_contigs[fi].append(ctg)
        fold_nt[fi] += ctg_len

    logger.info("Contigs divided into")
    for fi in range(folds):
        logger.info(
            " Fold%d: %5d contigs, %10d nt (%.4f)"
            % (fi, len(fold_contigs[fi]), fold_nt[fi], fold_nt[fi] / total_nt)
        )

    return fold_contigs


def divide_contigs_pct(contigs, test_pct, valid_pct, pct_abstain=0.2, seed=42):
    """Divide list of contigs into train/valid/test lists,
    aiming for the specified nucleotide percentages."""

    np.random.seed(seed)
    logger.info("Random divide contigs into train/valid/test based on nucleotide percentages")

    # sort contigs descending by length
    length_contigs = [(ctg.end - ctg.start, ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed train/valid/test nucleotides
    test_nt_aim = test_pct * total_nt
    valid_nt_aim = valid_pct * total_nt
    train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg_len, ctg in length_contigs:

        # compute gap between current and aim
        test_nt_gap = max(0, test_nt_aim - test_nt)
        valid_nt_gap = max(0, valid_nt_aim - valid_nt)
        train_nt_gap = max(1, train_nt_aim - train_nt)

        # skip if too large
        if ctg_len > pct_abstain * test_nt_gap:
            test_nt_gap = 0
        if ctg_len > pct_abstain * valid_nt_gap:
            valid_nt_gap = 0

        # compute remaining %
        gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
        test_pct_gap = test_nt_gap / gap_sum
        valid_pct_gap = valid_nt_gap / gap_sum
        train_pct_gap = train_nt_gap / gap_sum

        # sample train/valid/test
        ri = np.random.choice(range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap])[0]
        if ri == 0:
            train_contigs.append(ctg)
            train_nt += ctg_len
        elif ri == 1:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        elif ri == 2:
            test_contigs.append(ctg)
            test_nt += ctg_len
        else:
            logger.error("TVT random number beyond 0,1,2")
            exit(1)

    logger.info(
        f"""Contigs divided into
        Train: {len(train_contigs):5d} contigs, {train_nt:10d} nt ({train_nt / total_nt:.4f})
        Valid: {len(valid_contigs):5d} contigs, {valid_nt:10d} nt ({valid_nt / total_nt:.4f})
        Test: {len(test_contigs):5d} contigs, {test_nt:10d} nt ({test_nt / total_nt:.4f})"""
    )

    return [train_contigs, valid_contigs, test_contigs]


def divide_contigs_chr(contigs, test_chrs, valid_chrs):
    """Divide list of contigs into train/valid/test lists
    by chromosome."""

    logger.info("Divide contigs into train/valid/test by given chromosome")

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start

        if ctg.chr in test_chrs:
            test_contigs.append(ctg)
            test_nt += ctg_len
        elif ctg.chr in valid_chrs:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        else:
            train_contigs.append(ctg)
            train_nt += ctg_len

    total_nt = train_nt + valid_nt + test_nt

    logger.info(
        f"""Contigs divided into
        Train: {len(train_contigs):5d} contigs, {train_nt:10d} nt ({train_nt / total_nt:.4f})
        Valid: {len(valid_contigs):5d} contigs, {valid_nt:10d} nt ({valid_nt / total_nt:.4f})
        Test: {len(test_contigs):5d} contigs, {test_nt:10d} nt ({test_nt / total_nt:.4f})"""
    )

    return [train_contigs, valid_contigs, test_contigs]


def annotate_unmap(mseqs, unmap_bed, seq_length, pool_width):
    """Intersect the sequence segments with unmappable regions
         and annoate the segments as NaN to possible be ignored.

    Args:
      mseqs: list of ModelSeq's
      unmap_bed: unmappable regions BED file
      seq_length: sequence length (after cropping)
      pool_width: pooled bin width

    Returns:
      seqs_unmap: NxL binary NA indicators
    """

    # print sequence segments to file
    seqs_temp = tempfile.NamedTemporaryFile()
    seqs_bed_file = seqs_temp.name
    write_seqs_bed(seqs_bed_file, mseqs)

    # hash segments to indexes
    chr_start_indexes = {}
    for i in range(len(mseqs)):
        chr_start_indexes[(mseqs[i].chr, mseqs[i].start)] = i

    # initialize unmappable array
    pool_seq_length = seq_length // pool_width
    seqs_unmap = np.zeros((len(mseqs), pool_seq_length), dtype="bool")

    # intersect with unmappable regions
    p = subprocess.Popen(
        "bedtools intersect -wo -a %s -b %s" % (seqs_bed_file, unmap_bed), shell=True, stdout=subprocess.PIPE
    )
    for line in p.stdout:
        line = line.decode("utf-8")
        a = line.split()

        seq_chrom = a[0]
        seq_start = int(a[1])
        seq_end = int(a[2])
        seq_key = (seq_chrom, seq_start)

        unmap_start = int(a[4])
        unmap_end = int(a[5])

        overlap_start = max(seq_start, unmap_start)
        overlap_end = min(seq_end, unmap_end)

        pool_seq_unmap_start = math.floor((overlap_start - seq_start) / pool_width)
        pool_seq_unmap_end = math.ceil((overlap_end - seq_start) / pool_width)

        # skip minor overlaps to the first
        first_start = seq_start + pool_seq_unmap_start * pool_width
        first_end = first_start + pool_width
        first_overlap = first_end - overlap_start
        if first_overlap < 0.1 * pool_width:
            pool_seq_unmap_start += 1

        # skip minor overlaps to the last
        last_start = seq_start + (pool_seq_unmap_end - 1) * pool_width
        last_overlap = overlap_end - last_start
        if last_overlap < 0.1 * pool_width:
            pool_seq_unmap_end -= 1

        seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end] = True
        assert (
            seqs_unmap[chr_start_indexes[seq_key], pool_seq_unmap_start:pool_seq_unmap_end].sum()
            == pool_seq_unmap_end - pool_seq_unmap_start
        )

    return seqs_unmap


def write_seqs_bed(bed_file, seqs, labels=False, return_stats=False):
    """Write sequences to BED file."""
    bed_out = open(bed_file, "w")
    if return_stats:
        stats_dict = {}
    for i in range(len(seqs)):
        line = "%s\t%d\t%d" % (seqs[i].chr, seqs[i].start, seqs[i].end)
        if labels:
            line += "\t%s" % seqs[i].label
        if return_stats:
            if seqs[i].label not in stats_dict:
                stats_dict[seqs[i].label] = 0
            stats_dict[seqs[i].label] += 1
        print(line, file=bed_out)
    bed_out.close()

    if return_stats:
        return stats_dict


def read_blacklist(blacklist_bed, black_buffer=20):
    """Construct interval trees of blacklist
    regions for each chromosome."""
    black_chr_trees = {}

    if blacklist_bed is not None and os.path.isfile(blacklist_bed):
        for line in open(blacklist_bed):
            a = line.split()
            chrm = a[0]
            start = max(0, int(a[1]) - black_buffer)
            end = int(a[2]) + black_buffer

            if chrm not in black_chr_trees:
                black_chr_trees[chrm] = intervaltree.IntervalTree()

            black_chr_trees[chrm][start:end] = True

    return black_chr_trees


def get_labels(
    model_seqs,
    blacklist_bed,
    pool_width,
    kept_num_after_crop,
    seqs_cov_file,
    genome_cov_file,
    umap_npy_path=None,
    sum_stat="sum",
    baseline_pct=0.5,
    umap_pct=0.5,
    scale=1,
    extreme_clip_pct=None,
    offset=None,
    anchor_target=None,
    anchor_pct=0.999,
    clip=None,
    clip_soft=None,
):
    """Get coverage labels for model sequences."""

    logger.debug(
        f"""Setting for {seqs_cov_file.split('/')[-1].split('.')[0]}:
        sum_stat {sum_stat}, baseline_pct {baseline_pct}, umap_pct {umap_pct}, scale {scale}, extreme_clip_pct {extreme_clip_pct}, offset {offset}, anchor_target {anchor_target}, anchor_pct {anchor_pct}, clip_threshold {clip}, softclip_threshold {clip_soft}"""
    )

    # read blacklist regions
    black_chr_trees = read_blacklist(blacklist_bed)

    # compute dimensions
    num_seqs = len(model_seqs)
    seq_len_nt = model_seqs[0].end - model_seqs[0].start
    target_length = seq_len_nt // pool_width
    assert target_length > 0

    # initialize sequences coverage file
    seqs_cov_open = h5py.File(seqs_cov_file, "w")
    # seqs_cov_open.create_dataset('targets', shape=(num_seqs, target_length), dtype='float16')
    targets = []

    # open genome coverage file
    genome_cov_open = CovFace(genome_cov_file)

    # open unmap files if applicable
    if umap_npy_path is not None:
        unmap_mask = np.load(umap_npy_path)

    # for each model sequence
    for si in range(num_seqs):
        mseq = model_seqs[si]

        # read coverage
        seq_cov_nt = genome_cov_open.read(mseq.chr, mseq.start, mseq.end)
        seq_cov_nt = seq_cov_nt.astype("float32")

        # determine baseline coverage
        if target_length >= 8:
            baseline_cov = np.percentile(seq_cov_nt, 100 * baseline_pct)
            baseline_cov = np.nan_to_num(baseline_cov)
        else:
            baseline_cov = 0

        # set blacklist to baseline
        if mseq.chr in black_chr_trees:
            for black_interval in black_chr_trees[mseq.chr][mseq.start : mseq.end]:
                # adjust for sequence indexes
                black_seq_start = black_interval.begin - mseq.start
                black_seq_end = black_interval.end - mseq.start
                black_seq_values = seq_cov_nt[black_seq_start:black_seq_end]
                seq_cov_nt[black_seq_start:black_seq_end] = np.clip(black_seq_values, -baseline_cov, baseline_cov)
                # seq_cov_nt[black_seq_start:black_seq_end] = baseline_cov

        # set NaN's to baseline
        nan_mask = np.isnan(seq_cov_nt)
        seq_cov_nt[nan_mask] = baseline_cov

        # sum pool
        seq_cov = seq_cov_nt.reshape(target_length, pool_width)

        if sum_stat == "sum":
            seq_cov = seq_cov.sum(axis=1, dtype="float32")
        elif sum_stat == "sum_sqrt":
            seq_cov = seq_cov.sum(axis=1, dtype="float32")
            # seq_cov = -1 + (1+seq_cov)**0.75
            seq_cov = -1 + np.sqrt(1 + seq_cov)
        elif sum_stat in ["mean", "avg"]:
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
        elif sum_stat in ["mean_sqrt", "avg_sqrt"]:
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
            # seq_cov = -1 + (1+seq_cov)**0.75
            seq_cov = -1 + np.sqrt(1 + seq_cov)
        elif sum_stat == "median":
            seq_cov = seq_cov.median(axis=1)
        elif sum_stat == "max":
            seq_cov = seq_cov.max(axis=1)
        elif sum_stat == "peak":
            seq_cov = seq_cov.mean(axis=1, dtype="float32")
            seq_cov = np.clip(np.sqrt(seq_cov * 4), 0, 1)
        else:
            logger.error('ERROR: Unrecognized summary statistic "%s".' % sum_stat)
            exit(1)

        if umap_npy_path is not None:
            umap_cov = np.percentile(seq_cov, 100 * umap_pct)
            seq_cov[unmap_mask[si, :]] = np.minimum(seq_cov[unmap_mask[si, :]], umap_cov)

        # crop to final central size
        trim = (kept_num_after_crop - target_length) // 2
        if trim < 0:
            seq_cov = seq_cov[-trim:trim]

        # save
        targets.append(seq_cov)

    targets = np.array(targets, dtype="float32")
    assert targets.shape[0] == unmap_mask.shape[0]

    # clip extreme values
    if extreme_clip_pct is not None:
        extreme_clip = np.quantile(targets, extreme_clip_pct, method="lower")
        targets = np.clip(targets, -extreme_clip, extreme_clip)

    # substract the offset
    if offset is not None:
        targets = np.maximum(targets - offset, 0)

    # scale the quantile value to an anchor value
    if anchor_target is not None:
        org_quantile_value = np.quantile(targets, anchor_pct, method="lower")
        anchor_scale = anchor_target / org_quantile_value

        targets = targets * anchor_scale

    # soft/hard clip values to the wanted scale
    if clip_soft is not None:
        clip_mask = targets > clip_soft
        targets[clip_mask] = clip_soft - 1 + np.sqrt(targets[clip_mask] - clip_soft + 1)
    if clip is not None:
        targets = np.clip(targets, -clip, clip)

    # scale (we follow the Borzoi org implementation, put the scale at the end)
    targets = scale * targets

    # clip float16 min/max
    # targets = np.clip(targets, np.finfo(np.float16).min, np.finfo(np.float16).max).astype("float16")

    if not np.isfinite(targets).all():
        raise ValueError("Non-finite values (NaN or Inf) found in targets.")

    # write all
    seqs_cov_open.create_dataset("targets", data=targets, dtype="float32", compression="gzip")

    # close genome coverage file
    genome_cov_open.close()

    # close sequences coverage file
    seqs_cov_open.close()


class CovFace:
    """Coverage file I/O interface."""

    def __init__(self, cov_file):
        self.cov_file = cov_file
        self.bigwig = False
        self.bed = False

        cov_ext = os.path.splitext(self.cov_file)[1].lower()
        if cov_ext == ".gz":
            cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

        if cov_ext in [".bed", ".narrowpeak"]:
            self.bed = True
            self.preprocess_bed()

        elif cov_ext in [".bw", ".bigwig"]:
            self.cov_open = pyBigWig.open(self.cov_file, "r")
            self.bigwig = True

        elif cov_ext in [".h5", ".hdf5", ".w5", ".wdf5"]:
            self.cov_open = h5py.File(self.cov_file, "r")

        else:
            logger.error('Cannot identify coverage file extension "%s".' % cov_ext)
            exit(1)

    def preprocess_bed(self):
        # read BED
        bed_df = pd.read_csv(self.cov_file, sep="\t", usecols=range(3), names=["chr", "start", "end"])

        # for each chromosome
        self.cov_open = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm]

            # find max pos
            pos_max = bed_chr_df.end.max()

            # initialize array
            self.cov_open[chrm] = np.zeros(pos_max, dtype="bool")

            # set peaks
            for peak in bed_chr_df.itertuples():
                self.cov_open[peak.chr][peak.start : peak.end] = 1

    def read(self, chrm, start, end):
        if self.bigwig:
            cov = self.cov_open.values(chrm, start, end, numpy=True).astype("float16")

        else:
            if chrm in self.cov_open:
                cov = self.cov_open[chrm][start:end]

                # handle mysterious inf's
                cov = np.clip(cov, np.finfo(np.float16).min, np.finfo(np.float16).max)

                # pad
                pad_zeros = end - start - len(cov)
                if pad_zeros > 0:
                    cov_pad = np.zeros(pad_zeros, dtype="bool")
                    cov = np.concatenate([cov, cov_pad])

            else:
                logger.warning(
                    "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % (self.cov_file, chrm, start, end),
                )
                cov = np.zeros(end - start, dtype="float16")

        return cov

    def close(self):
        if not self.bed:
            self.cov_open.close()


def aggregate_data(storage_path, preload_data, ref_order, task=None, precision="float32"):
    """Aggregate the label data from multiple files into a single file."""

    separate_label_file = [
        f"{storage_path}/labels/{i}" for i in os.listdir(f"{storage_path}/labels") if i.endswith(".h5")
    ]
    separate_label_file = [
        f"{storage_path}/labels/{i}.h5"
        for i in ref_order
        if f"{storage_path}/labels/{i}.h5" in separate_label_file
    ]
    if not separate_label_file:
        logger.error("No label files found in the specified directory.")
        exit(1)
    else:
        logger.info(f"Found {len(separate_label_file)} label files to aggregate.")

    # get all the label data
    label_data = []
    label_meta = pd.DataFrame()
    for dim, label_file in enumerate(separate_label_file):
        with h5py.File(label_file, "r") as f:
            label_data.append(f["targets"][:])
        label_meta.loc[dim, "trial"] = label_file.split("/")[-1].split(".")[0]
    label_meta.index.name = "dim"
    label_meta.to_csv(f"{storage_path}/label_meta.csv", index=True)

    label_data = np.stack(label_data, axis=-1)
    try:
        precision = getattr(torch, precision)
    except AttributeError:
        logger.warning(f"Specified precision {precision} is not accept, save as float32")
        precision = getattr(torch, "float32")
    label_data = torch.tensor(label_data, dtype=precision)

    if preload_data:
        # save the aggregated data
        # TODO: if the data is too large (if we add the pre tokenization step), save it in chunks given how many GPUs are used, so that each dataset will only load the data for one GPU
        for dataset_type, tmp in task.groupby(3):
            torch.save({"label": label_data[tmp.index]}, f"{storage_path}/data/{dataset_type}.pt")
        torch.save(label_data, f"{storage_path}/data/all_label.pt")

    else:
        # save per data point separately
        for i in task.index:
            chrom, start, end = task.iloc[i, [0, 1, 2]]
            torch.save(label_data[i].clone(), f"{storage_path}/data/{chrom}_{start}_{end}.pt")
