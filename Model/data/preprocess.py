# Adapted from https://github.com/calico/basenji/blob/master/bin/basenji_data.py, https://github.com/calico/basenji/blob/master/bin/basenji_data_read.py
import json
import logging
import multiprocessing as mp
import os
import random
import time
from functools import partial

import numpy as np
import pandas as pd
from utils.logging import LOGGER_PREFIX, LazyLogger

from .data_utils import (
    STD_CHR,
    Contig,
    aggregate_data,
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

logger = LazyLogger(f"{LOGGER_PREFIX}-Data Preprocess")

PARA_CHECK = [
    "refer_genom",
    "gap_bed",
    "unmap_bed",
    "unmap_threshold",
    "blacklist_bed",
    "baseline_pct",
    "break_threshold",
    "stride_train",
    "stride_test",
    "snap",
    "context_length",
    "window_size",
    "n_window",
    "folds",
    "valid_pct_or_chr",
    "test_pct_or_chr",
    "sample_pct",
    "seed",
]


def get_trail_value_mp(
    trial, storage_path, mseqs, blacklist_bed, baseline_pct, window_size, n_window, restart=False
):

    exp = trial["exp"]
    genome_cov_file = trial["file"]
    seqs_cov_file = f"{storage_path}/labels/{exp}.h5"
    sum_stat = trial["sum_stat"]

    clip_ti = trial.get("clip", None)
    clipsoft_ti = trial.get("clip_soft", None)
    scale_ti = trial.get("scale", 1)

    if os.path.exists(seqs_cov_file) and not restart:
        logger.info("Skipping existing %s" % seqs_cov_file)
        return None

    try:
        get_labels(
            model_seqs=mseqs,
            blacklist_bed=blacklist_bed,
            baseline_pct=baseline_pct,
            pool_width=window_size,
            kept_num_after_crop=n_window,
            scale=scale_ti,
            clip=clip_ti,
            clip_soft=clipsoft_ti,
            sum_stat=sum_stat,
            seqs_cov_file=seqs_cov_file,
            genome_cov_file=genome_cov_file,
            clip_pct=0.9999999,
        )
    except Exception as e:
        logger.error(f"Fail to process {exp}. Manually check.")
        if logger.getEffectiveLevel() > logging.DEBUG:
            logger.error(e)
        else:
            # If the logging level is DEBUG, log the full traceback
            logger.exception(e)
        return exp

    return None


def preprocess(
    storage_path,
    refer_genom,
    trial_summary_path,
    gap_bed=None,
    unmap_bed=None,
    unmap_threshold=0.5,
    blacklist_bed=None,
    baseline_pct=0.25,
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
    seed=42,
    num_worker=16,
    force_restart=False,
    preload_data=True,
):
    #################################################
    # basic settings
    #################################################

    ## output setting
    os.makedirs(storage_path, exist_ok=True)

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
        stride_train_int = int(np.round(stride_train * context_length))
    else:
        stride_train_int = stride_train
    if stride_test <= 1:
        stride_test_int = int(np.round(stride_test * context_length))
    else:
        stride_test_int = stride_test

    ## check snap
    if snap is not None:
        if np.mod(context_length, snap) != 0:
            raise ValueError("seq_length must be a multiple of snap")
        if np.mod(stride_train_int, snap) != 0:
            raise ValueError("stride_train must be a multiple of snap")
        if np.mod(stride_test_int, snap) != 0:
            raise ValueError("stride_test must be a multiple of snap")

    logger.info(
        f"""Preprocess config:
        context length: {context_length} bp
        window size: {window_size}
        kept central window num: {n_window}
        stride train: {stride_train_int} bp
        stride test: {stride_train_int} bp"""
    )

    if force_restart or not os.path.exists(f"{storage_path}/statistics.json"):
        logger.info("Start preprocess data")

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

        # filter for specified chromosomes
        contigs = [ctg for ctg in contigs if ctg.chr in STD_CHR]

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
                stride_fold = stride_test_int
            else:
                stride_fold = stride_train_int

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
        if unmap_bed is not None:
            # annotate unmappable positions
            mseqs_unmap = annotate_unmap(mseqs, unmap_bed, context_length, window_size)

            # filter unmappable
            mseqs_map_mask = mseqs_unmap.mean(axis=1, dtype="float64") < unmap_threshold
            mseqs = [mseqs[i] for i in range(len(mseqs)) if mseqs_map_mask[i]]
            mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

            # save the mappability mask for kept model sequences
            unmap_npy = f"{storage_path}/mseqs_unmap.npy"
            np.save(unmap_npy, mseqs_unmap)

        # write all the sequences to BED
        seqs_bed_file = f"{storage_path}/sequences.bed"
        write_seqs_bed(seqs_bed_file, mseqs, labels=True)

        logger.info(
            "Get\n" + "\n".join(f"{fold_labels[fi]}_num: {len(fold_mseqs[fi])}" for fi in range(num_folds))
        )

        ################################################################
        # generate labels for the model sequences
        ################################################################

        # read in the trail bigwig summary files
        try:
            trial_files = pd.read_csv(trial_summary_path)
            assert "file" in trial_files.columns
            assert "exp" in trial_files.columns
            assert "sum_stat" in trial_files.columns
        except FileNotFoundError or AssertionError:
            logger.error(
                "The trial files should be provided in a summary csv file, with at least the file path (`file`), the trial name (`exp`) and summary statistics (`sum_stat`) provided."
            )
            exit(1)
        logger.info(f"Found {len(trial_files)} trial files. Start generate labels.")

        os.makedirs(f"{storage_path}/labels", exist_ok=True)

        # get summary statistics
        get_trail_value_mp_prebound = partial(
            get_trail_value_mp,
            storage_path=storage_path,
            mseqs=mseqs,
            blacklist_bed=blacklist_bed,
            baseline_pct=baseline_pct,
            window_size=window_size,
            n_window=n_window,
            restart=force_restart,
        )

        start_time = time.perf_counter()
        with mp.Pool(processes=num_worker) as pool:
            failed_targets = pool.map(get_trail_value_mp_prebound, trial_files.to_dict(orient="records"))
        end_time = time.perf_counter()

        failed_targets = [item for item in failed_targets if item is not None]
        failed_num = sum(failed_targets)
        if failed_num > 0:
            logger.warning(
                f"Failed to process {failed_num} trial files. Please check the log for details. The failed trials are saved in {storage_path}/failed_trials.txt"
            )
            with open(f"{storage_path}/failed_trials.txt", "w") as f:
                for trial in failed_targets:
                    f.write(f"{trial}\n")

        ################################################################
        # stats
        ################################################################
        stats_dict = {}
        stats_dict["total_targets"] = trial_files.shape[0]
        stats_dict["succeeded_targets_num"] = trial_files.shape[0] - failed_num
        stats_dict["failed_targets_num"] = failed_num
        stats_dict["context_length"] = context_length
        stats_dict["window_size"] = window_size
        stats_dict["kept_bin_num"] = n_window
        stats_dict["time"] = end_time - start_time

        for fi in range(num_folds):
            stats_dict["%s_seqs" % fold_labels[fi]] = len(fold_mseqs[fi])

        with open(f"{storage_path}/statistics.json", "w") as stats_json_out:
            json.dump(stats_dict, stats_json_out, indent=4)

        with open(f"{storage_path}/para.json", "w") as para_json_out:
            para = {k: v for k, v in locals().items() if k in PARA_CHECK}
            json.dump(para, para_json_out, indent=4)

        logger.info(
            f"Finish preprocess data in {(end_time - start_time) / 60:.2f} minutes\nSave at: {storage_path}"
        )

    else:
        # check if the preprocess para setting is the same
        with open(f"{storage_path}/para.json", "r") as para_json_in:
            para_dict = json.load(para_json_in)
        for para in PARA_CHECK:
            if para not in para_dict:
                logger.error(
                    f"The parameter {para} is missing in the preprocessed data. Please re-run preprocess."
                )
                exit(1)
            if locals()[para] != para_dict[para]:
                logger.error(
                    f"The parameter {para} is different from the preprocessed data. Please re-run preprocess."
                )
                exit(1)

        # load statistics
        with open(f"{storage_path}/statistics.json", "r") as stats_json_in:
            stats_dict = json.load(stats_json_in)
        logger.info(
            "Get\n"
            + "\n".join(
                f"{stats.split('_')[0]}_num: {stats_dict[stats]}" for stats in stats_dict if "_seqs" in stats
            )
        )
        logger.info(f"Preprocess data already exists at {storage_path}. Skipping preprocess step.")

    # start to aggregate the preprocessed data
    os.makedirs(f"{storage_path}/data", exist_ok=True)

    if preload_data:
        if force_restart or not os.path.exists(f"{storage_path}/data/all_label.pt"):
            logger.info("Start to aggregate data")
            data = pd.read_csv(f"{storage_path}/sequences.bed", sep="\t", header=None)
            aggregate_data(storage_path=storage_path, preload_data=True, task=data)
            logger.info("Finish aggregation")
    else:
        data = pd.read_csv(f"{storage_path}/sequences.bed", sep="\t", header=None)
        data["generate"] = data.apply(
            lambda x: not os.path.exists(f"{storage_path}/data/{x[0]}_{x[1]}_{x[2]}.pt") or force_restart, axis=1
        )

        if data["generate"].any():
            logger.info("Start to aggregate data")
            aggregate_data(storage_path=storage_path, preload_data=False, task=data[data["generate"]])
            logger.info("Finish aggregation")
