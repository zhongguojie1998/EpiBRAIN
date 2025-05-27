# Adapted from https://github.com/calico/basenji/blob/master/bin/basenji_data.py, https://github.com/calico/basenji/blob/master/bin/basenji_data_read.py
import json
import multiprocessing as mp
import os
import random
from functools import partial

import numpy as np
import pandas as pd
from utils.config import LOGGER_PREFIX, get_logger
from utils.utils import split_tasks

from .data_utils import (
    Contig,
    annotate_unmap,
    break_large_contigs,
    contig_sequences,
    divide_contigs_chr,
    divide_contigs_folds,
    divide_contigs_pct,
    get_labels,
    load_chromosomes,
    rejoin_large_contigs,
    split_contigs,
    write_seqs_bed,
)

logger = get_logger(f"{LOGGER_PREFIX}-Data preprocess")


def get_trail_value_mp(trial_files, storage_path, mseqs, blacklist_bed, window_size, n_window):
    for trial_file in trial_files.index:
        genome_cov_file = trial_files.loc[trial_file, "file"]
        seqs_cov_file = f"{storage_path}/labels/{trial_file}.h5"

        clip_ti = None
        if "clip" in trial_files.columns:
            clip_ti = trial_files.loc[trial_file, "clip"]

        clipsoft_ti = None
        if "clip_soft" in trial_files.columns:
            clipsoft_ti = trial_files.loc[trial_file, "clip_soft"]

        scale_ti = 1
        if "scale" in trial_files.columns:
            scale_ti = trial_files.loc[trial_file, "scale"]

        if os.path.exists(seqs_cov_file):
            logger.info("Skipping existing %s" % seqs_cov_file)

        get_labels(
            model_seqs=mseqs,
            blacklist_bed=blacklist_bed,
            pool_width=window_size,
            kept_num_after_crop=n_window,
            scale=scale_ti,
            clip=clip_ti,
            clip_soft=clipsoft_ti,
            sum_stat=trial_files.loc[trial_file, "sum_stat"],
            seqs_cov_file=seqs_cov_file,
            genome_cov_file=genome_cov_file,
        )


def preprocess(
    storage_path,
    refer_genom,
    trial_path,
    gap_bed=None,
    blacklist_bed=None,
    umap_bed=None,
    break_threshold=1_179_648,
    stride_train=1.0,
    stride_test=1.0,
    snap=1,
    context_length=196_608,
    window_size=128,
    n_window=896,
    folds=None,
    valid_pct_or_chr=0.1,
    test_pct_or_chr=0.1,
    sample_pct=1.0,
    umap_threshold=0.5,
    seed=42,
    num_worker=16,
):
    #################################################
    # basic settings
    #################################################

    ## output setting
    if os.path.isdir(storage_path):
        os.makedirs(storage_path, exist_ok=True)
    else:
        logger.error(f"The storage path {storage_path} should be a directory.")
        exit(1)

    ## random seed setting
    random.seed(seed)
    np.random.seed(seed)

    ## check the contig break down is at least longer than the context length
    if not break_threshold is None and break_threshold < context_length:
        logger.error(
            f"The break threshold {break_threshold} should be larger than the context length {context_length}."
        )
        exit(1)

    ## transform proportion strides to base pairs
    if stride_train <= 1:
        stride_train = int(np.round(stride_train * context_length))
    if stride_test <= 1:
        stride_test = int(np.round(stride_test * context_length))

    ## check snap
    if snap is not None:
        if np.mod(context_length, snap) != 0:
            raise ValueError("seq_length must be a multiple of snap")
        if np.mod(stride_train, snap) != 0:
            raise ValueError("stride_train must be a multiple of snap")
        if np.mod(stride_test, snap) != 0:
            raise ValueError("stride_test must be a multiple of snap")

    logger.info(
        f"""
        Preprocess config:
        context length: {context_length} bp
        window size: {window_size}
        kept central window num: {n_window}
        stride train: {stride_train} bp
        stride test: {stride_test} bp
        """
    )

    #################################################
    # read in / curate the reference genome
    #################################################

    # load the reference data
    chrom_contigs = load_chromosomes(refer_genom)

    # if gap bed file is provided, split the contig and remove the gaps
    if gap_bed is not None:
        if not os.path.isfile(gap_bed):
            logger.error(f"The gap bed file {gap_bed} does not exist.")
            exit(1)
        chrom_contigs = split_contigs(chrom_contigs, gap_bed)

    # ditch the chromosomes for contigs
    contigs = []
    for chrom in chrom_contigs:
        contigs += [Contig(chrom, ctg_start, ctg_end) for ctg_start, ctg_end in chrom_contigs[chrom]]

    # filter for large enough contigs
    contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= context_length]

    # break up large contigs
    if break_threshold is not None:
        contigs = break_large_contigs(contigs, break_threshold)

    ################################################################
    # divide between train/valid/test (contigs)
    ################################################################\

    if folds is not None:
        fold_labels = ["fold%d" % fi for fi in range(folds)]
        num_folds = folds
    else:
        fold_labels = ["train", "valid", "test"]
        num_folds = 3

    if folds is not None:
        # divide by fold pct
        fold_contigs = divide_contigs_folds(contigs, folds)

    else:
        try:
            # convert to float pct
            valid_pct = float(valid_pct_or_chr)
            test_pct = float(test_pct_or_chr)
            assert 0 <= valid_pct <= 1
            assert 0 <= test_pct <= 1

            # divide by pct
            fold_contigs = divide_contigs_pct(contigs, test_pct, valid_pct)

        except (ValueError, AssertionError):
            # divide by chr
            valid_chrs = valid_pct_or_chr.split(",")
            test_chrs = test_pct_or_chr.split(",")
            fold_contigs = divide_contigs_chr(contigs, test_chrs, valid_chrs)

    # rejoin broken contigs within set
    for fi in range(len(fold_contigs)):
        fold_contigs[fi] = rejoin_large_contigs(fold_contigs[fi])

    # write labeled contigs to BED file
    ctg_bed_file = f"{storage_path}/contigs.bed"
    with open(ctg_bed_file, "w") as f:
        for fi in range(len(fold_contigs)):
            for ctg in fold_contigs[fi]:
                line = "%s\t%d\t%d\t%s" % (ctg.chr, ctg.start, ctg.end, fold_labels[fi])
                f.write(line + "\n")

    ################################################################
    # define model sequences (in each contig, stride across contig)
    ################################################################
    fold_mseqs = []
    for fi in range(num_folds):
        if fold_labels[fi] in ["valid", "test"]:
            stride_fold = stride_test
        else:
            stride_fold = stride_train

        # stride sequences across contig
        fold_mseqs_fi = contig_sequences(fold_contigs[fi], context_length, stride_fold, snap, fold_labels[fi])
        fold_mseqs.append(fold_mseqs_fi)

        # shuffle
        random.shuffle(fold_mseqs[fi])

        # down-sample
        if sample_pct < 1.0:
            fold_mseqs[fi] = random.sample(fold_mseqs[fi], int(sample_pct * len(fold_mseqs[fi])))

    # merge into one list
    mseqs = [ms for fm in fold_mseqs for ms in fm]

    # mappability filter
    if umap_bed is not None:
        # annotate unmappable positions
        mseqs_unmap = annotate_unmap(mseqs, umap_bed, context_length, window_size)

        # filter unmappable
        mseqs_map_mask = mseqs_unmap.mean(axis=1, dtype="float64") < umap_threshold
        mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
        mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

        # write to file
        unmap_npy = f"{storage_path}/mseqs_unmap.npy"
        np.save(unmap_npy, mseqs_unmap)

    # write all the sequences to BED
    seqs_bed_file = f"{storage_path}/sequences.bed"
    write_seqs_bed(seqs_bed_file, mseqs, labels=True)

    ################################################################
    # generate labels for the model sequences
    ################################################################

    # read in the trail bigwig summary files
    try:
        trial_files = pd.read_csv(trial_path, index_col=0)
    except FileNotFoundError:
        logger.error(
            "The trial files should be provided in a summary csv file, with at least the file path (`file`) and summary statistics (`sum_stat`) provided."
        )
        exit(1)
    logger.info(f"Found {len(trial_files)} trial files.")

    os.makedirs(f"{storage_path}/labels", exist_ok=True)

    # get summary statistics
    get_trail_value_mp_prebound = partial(
        get_trail_value_mp,
        storage_path=storage_path,
        mseqs=mseqs,
        blacklist_bed=blacklist_bed,
        window_size=window_size,
        n_window=n_window,
    )
    tasks = split_tasks(trial_files, num_worker)
    with mp.Pool(processes=num_worker) as pool:
        pool.map(get_trail_value_mp_prebound, tasks)

    ################################################################
    # stats
    ################################################################
    stats_dict = {}
    stats_dict["num_targets"] = trial_files.shape[0]
    stats_dict["context_length"] = context_length
    stats_dict["window_size"] = window_size
    stats_dict["kept_bin_num"] = n_window

    for fi in range(num_folds):
        stats_dict["%s_seqs" % fold_labels[fi]] = len(fold_mseqs[fi])

    with open(f"{storage_path}/statistics.json", "w") as stats_json_out:
        json.dump(stats_dict, stats_json_out, indent=4)
